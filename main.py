import os
import secrets
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from itsdangerous import URLSafeSerializer, BadSignature

from fastapi import Form
from database import (
    init_db, save_user, get_user_by_id, save_bookmarks, get_bookmarks,
    get_bookmark, get_similar_bookmarks, save_deep_analysis,
    toggle_read, save_notes, save_roadmap_progress, update_bookmark_meta,
    save_mvp_prompt,
)
from ai_service import categorize_bookmarks, deep_analyze, generate_mvp_prompt, generate_mvp_questions
from x_auth import get_auth_url, exchange_code_for_token, get_user_info

load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_hex(32))
s = URLSafeSerializer(SECRET_KEY)


@app.on_event("startup")
async def startup():
    await init_db()


def get_session(request: Request):
    token = request.cookies.get("session")
    if not token:
        return None
    try:
        return s.loads(token)
    except BadSignature:
        return None


# ── Routes ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    session = get_session(request)
    return templates.TemplateResponse(
        "index.html", {"request": request, "logged_in": bool(session)}
    )


@app.get("/auth/login")
async def auth_login():
    auth_url, state, code_verifier = get_auth_url()
    response = RedirectResponse(url=auth_url)
    response.set_cookie(
        "oauth_state",
        s.dumps({"state": state, "cv": code_verifier}),
        max_age=600,
        httponly=True,
        samesite="lax",
    )
    return response


@app.get("/auth/callback")
async def auth_callback(request: Request, code: str = None, state: str = None):
    oauth_cookie = request.cookies.get("oauth_state")
    if not oauth_cookie:
        raise HTTPException(400, "OAuth state eksik")

    try:
        oauth_data = s.loads(oauth_cookie)
    except BadSignature:
        raise HTTPException(400, "Geçersiz OAuth state")

    if state != oauth_data["state"]:
        raise HTTPException(400, "State uyuşmuyor")

    token_data = await exchange_code_for_token(code, oauth_data["cv"])
    access_token = token_data["access_token"]

    user_info = await get_user_info(access_token)
    user_id = await save_user(user_info["id"], user_info["username"], access_token)

    response = RedirectResponse(url="/dashboard")
    response.set_cookie(
        "session",
        s.dumps({"user_id": user_id}),
        max_age=86400 * 30,
        httponly=True,
        samesite="lax",
    )
    response.delete_cookie("oauth_state")
    return response


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    session = get_session(request)
    if not session:
        return RedirectResponse(url="/")

    bookmarks = await get_bookmarks(session["user_id"])

    categories: dict[str, list] = {}
    for bm in bookmarks:
        cat = bm["category"] or "Other"
        categories.setdefault(cat, []).append(bm)

    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "categories": categories, "total": len(bookmarks)},
    )


@app.post("/sync")
async def sync_bookmarks(request: Request):
    session = get_session(request)
    if not session:
        raise HTTPException(401, "Giriş yapılmamış")

    user = await get_user_by_id(session["user_id"])
    if not user:
        raise HTTPException(404, "Kullanıcı bulunamadı")

    headers = {"Authorization": f"Bearer {user['access_token']}"}
    tweets = []

    async with httpx.AsyncClient() as http:
        me_res = await http.get("https://api.twitter.com/2/users/me", headers=headers)
        me_res.raise_for_status()
        user_x_id = me_res.json()["data"]["id"]

        bm_res = await http.get(
            f"https://api.twitter.com/2/users/{user_x_id}/bookmarks",
            headers=headers,
            params={"max_results": 10, "tweet.fields": "created_at,entities"},
        )
        bm_res.raise_for_status()
        bm_data = bm_res.json()

        if bm_data.get("data"):
            tweets = [{"id": str(t["id"]), "text": t["text"]} for t in bm_data["data"]]

    if not tweets:
        return JSONResponse({"message": "Bookmark bulunamadı", "count": 0})
    categorized = await categorize_bookmarks(tweets)
    await save_bookmarks(session["user_id"], categorized)

    return JSONResponse({"message": "Tamamlandı!", "count": len(categorized)})


@app.get("/bookmark/{tweet_id}", response_class=HTMLResponse)
async def bookmark_detail(request: Request, tweet_id: str):
    session = get_session(request)
    if not session:
        return RedirectResponse(url="/")

    bookmark = await get_bookmark(session["user_id"], tweet_id)
    if not bookmark:
        raise HTTPException(404, "Bookmark bulunamadı")

    similar = await get_similar_bookmarks(
        session["user_id"], tweet_id,
        bookmark["category"], bookmark["tags"]
    )

    # Deep analysis: generate if not cached
    if not bookmark.get("deep_analysis"):
        analysis = await deep_analyze(bookmark, similar)
        await save_deep_analysis(session["user_id"], tweet_id, analysis)
        bookmark["deep_analysis"] = analysis

    return templates.TemplateResponse("detail.html", {
        "request": request,
        "bm": bookmark,
        "similar": similar,
        "analysis": bookmark["deep_analysis"],
    })


@app.post("/bookmark/{tweet_id}/read")
async def mark_read(request: Request, tweet_id: str):
    session = get_session(request)
    if not session:
        raise HTTPException(401)
    is_read = await toggle_read(session["user_id"], tweet_id)
    return JSONResponse({"is_read": is_read})


@app.post("/bookmark/{tweet_id}/notes")
async def update_notes(request: Request, tweet_id: str, notes: str = Form("")):
    session = get_session(request)
    if not session:
        raise HTTPException(401)
    await save_notes(session["user_id"], tweet_id, notes)
    return JSONResponse({"ok": True})


@app.post("/bookmark/{tweet_id}/progress")
async def update_progress(request: Request, tweet_id: str):
    session = get_session(request)
    if not session:
        raise HTTPException(401)
    body = await request.json()
    await save_roadmap_progress(session["user_id"], tweet_id, body)
    return JSONResponse({"ok": True})


@app.post("/bookmark/{tweet_id}/reanalyze")
async def reanalyze(request: Request, tweet_id: str):
    session = get_session(request)
    if not session:
        raise HTTPException(401)
    bookmark = await get_bookmark(session["user_id"], tweet_id)
    if not bookmark:
        raise HTTPException(404)
    similar = await get_similar_bookmarks(session["user_id"], tweet_id, bookmark["category"], bookmark["tags"])
    analysis = await deep_analyze(bookmark, similar)
    await save_deep_analysis(session["user_id"], tweet_id, analysis)
    return JSONResponse({"ok": True})


@app.post("/bookmark/{tweet_id}/edit")
async def edit_bookmark(
    request: Request, tweet_id: str,
    category: str = Form(...),
    difficulty: str = Form(...),
    priority: int = Form(...),
):
    session = get_session(request)
    if not session:
        raise HTTPException(401)
    await update_bookmark_meta(session["user_id"], tweet_id, category, difficulty, priority)
    return JSONResponse({"ok": True})


@app.post("/bookmark/{tweet_id}/mvp-questions")
async def create_mvp_questions(request: Request, tweet_id: str):
    session = get_session(request)
    if not session:
        raise HTTPException(401)
    bookmark = await get_bookmark(session["user_id"], tweet_id)
    if not bookmark:
        raise HTTPException(404)
    result = await generate_mvp_questions(bookmark)
    return JSONResponse(result)


@app.post("/bookmark/{tweet_id}/mvp-prompt")
async def create_mvp_prompt(request: Request, tweet_id: str):
    session = get_session(request)
    if not session:
        raise HTTPException(401)
    bookmark = await get_bookmark(session["user_id"], tweet_id)
    if not bookmark:
        raise HTTPException(404)
    body = await request.json()
    answers = body.get("answers", {})
    project_type = body.get("project_type", "mobile")
    prompt = await generate_mvp_prompt(bookmark, answers, project_type)
    await save_mvp_prompt(session["user_id"], tweet_id, prompt)
    return JSONResponse({"prompt": prompt})


@app.get("/bookmark/{tweet_id}/export")
async def export_roadmap(request: Request, tweet_id: str):
    from fastapi.responses import PlainTextResponse
    session = get_session(request)
    if not session:
        raise HTTPException(401)
    bookmark = await get_bookmark(session["user_id"], tweet_id)
    if not bookmark or not bookmark.get("deep_analysis"):
        raise HTTPException(404, "Analiz bulunamadı")

    analysis = bookmark["deep_analysis"]
    lines = [
        f"# {bookmark['summary'] or bookmark['text'][:80]}",
        f"\n**Kategori:** {bookmark['category']} / {bookmark['subcategory']}",
        f"**Zorluk:** {bookmark['difficulty']}  |  **Tür:** {bookmark['content_type']}",
        f"\n## Özet\n{analysis.get('detailed_summary', '')}",
        f"\n## Neden Önemli?\n{analysis.get('why_it_matters', '')}",
    ]

    if analysis.get("prerequisites"):
        lines.append("\n## Önkoşullar")
        for p in analysis["prerequisites"]:
            lines.append(f"- {p}")

    if analysis.get("what_to_do"):
        lines.append("\n## Ne Yapmalısın?")
        for step in analysis["what_to_do"]:
            lines.append(f"\n**{step['step']}. {step['action']}**")
            lines.append(f"{step['detail']}")

    if analysis.get("roadmap"):
        lines.append("\n## Öğrenme Yol Haritası")
        for phase in analysis["roadmap"]:
            lines.append(f"\n### {phase['phase']} ({phase.get('duration', '')})")
            for step in phase.get("steps", []):
                lines.append(f"- [ ] {step}")

    if analysis.get("resources"):
        lines.append("\n## Kaynaklar")
        for r in analysis["resources"]:
            lines.append(f"- {r}")

    if bookmark.get("notes"):
        lines.append(f"\n## Notlarım\n{bookmark['notes']}")

    content = "\n".join(lines)
    return PlainTextResponse(
        content,
        headers={"Content-Disposition": f"attachment; filename=roadmap-{tweet_id}.md"}
    )


@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/")
    response.delete_cookie("session")
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
