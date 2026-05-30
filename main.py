import os
import secrets
import hmac
import hashlib
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from itsdangerous import URLSafeSerializer, BadSignature
from urllib.parse import urlencode

from fastapi import Form
from database import (
    init_db, save_user, get_user_by_id, save_bookmarks, get_bookmarks,
    get_bookmark, get_similar_bookmarks, save_deep_analysis,
    toggle_read, save_notes, save_roadmap_progress, update_bookmark_meta,
    save_mvp_prompt, delete_user,
    get_user_by_paddle_customer, get_user_by_paddle_subscription,
    update_subscription, update_subscription_by_sub_id,
    count_bookmarks, get_existing_tweet_ids, update_last_synced,
)
from ai_service import categorize_bookmarks, deep_analyze, generate_mvp_prompt, generate_mvp_questions
from x_auth import generate_code_verifier, generate_code_challenge, exchange_code_for_token, get_user_info

load_dotenv()

# ── Paddle client ────────────────────────────────────────────────────────────
from paddle_billing import Client, Environment, Options

_paddle_env = os.getenv("PADDLE_ENVIRONMENT", "production")
paddle = Client(
    os.getenv("PADDLE_API_SECRET_KEY", ""),
    options=Options(Environment.SANDBOX if _paddle_env == "sandbox" else Environment.PRODUCTION),
)
PADDLE_CLIENT_TOKEN = os.getenv("PADDLE_CLIENT_TOKEN", "")
PADDLE_WEBHOOK_SECRET = os.getenv("PADDLE_WEBHOOK_SECRET", "")
PADDLE_PRICE_MONTHLY = os.getenv("PADDLE_PRICE_MONTHLY", "")
PADDLE_PRICE_ANNUAL = os.getenv("PADDLE_PRICE_ANNUAL", "")
PADDLE_SANDBOX = _paddle_env == "sandbox"


def _verify_paddle_signature(sig_header: str, raw_body: bytes) -> bool:
    """Verify Paddle webhook HMAC-SHA256 signature."""
    try:
        parts = dict(p.split("=", 1) for p in sig_header.split(";"))
        timestamp = parts.get("ts", "")
        paddle_sig = parts.get("h1", "")
        payload = f"{timestamp}:{raw_body.decode('utf-8')}".encode()
        computed = hmac.new(
            PADDLE_WEBHOOK_SECRET.encode(),
            msg=payload,
            digestmod=hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(computed, paddle_sig)
    except Exception:
        return False

app = FastAPI()
app.mount("/assets", StaticFiles(directory="assets"), name="assets")
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


# ── Helpers ─────────────────────────────────────────────────────────────────

async def _do_sync(user_id: int, access_token: str) -> dict:
    """Fetch bookmarks from X and categorize only NEW ones with AI."""
    headers = {"Authorization": f"Bearer {access_token}"}

    async with httpx.AsyncClient() as http:
        me_res = await http.get("https://api.twitter.com/2/users/me", headers=headers)
        if me_res.status_code != 200:
            raise RuntimeError(f"Twitter /users/me failed ({me_res.status_code}): {me_res.text}")
        user_x_id = me_res.json()["data"]["id"]

        bm_res = await http.get(
            f"https://api.twitter.com/2/users/{user_x_id}/bookmarks",
            headers=headers,
            params={"max_results": 100, "tweet.fields": "created_at,entities"},
        )
        if bm_res.status_code != 200:
            raise RuntimeError(f"Twitter /bookmarks failed ({bm_res.status_code}): {bm_res.text}")

        bm_data = bm_res.json()
        all_tweets = []
        if bm_data.get("data"):
            all_tweets = [{"id": str(t["id"]), "text": t["text"]} for t in bm_data["data"]]

    if not all_tweets:
        return {"count": 0, "new": 0}

    # Only send NEW tweets to the AI — saves cost on re-syncs
    existing_ids = await get_existing_tweet_ids(user_id)
    new_tweets = [t for t in all_tweets if t["id"] not in existing_ids]

    if new_tweets:
        categorized = await categorize_bookmarks(new_tweets)
        await save_bookmarks(user_id, categorized)

    await update_last_synced(user_id)
    return {"count": len(all_tweets), "new": len(new_tweets)}


# ── Routes ──────────────────────────────────────────────────────────────────

_SECURE_COOKIE = os.getenv("REDIRECT_URI", "").startswith("https")


@app.get("/health")
async def health_check():
    """Lightweight endpoint for uptime monitoring — no DB/AI calls."""
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    session = get_session(request)
    if session:
        return RedirectResponse(url="/dashboard", status_code=302)
    return templates.TemplateResponse(
        "index.html", {"request": request, "logged_in": False}
    )


@app.get("/auth/login")
async def auth_login():
    state = secrets.token_urlsafe(16)
    code_verifier = generate_code_verifier()
    code_challenge = generate_code_challenge(code_verifier)

    # Embed code_verifier inside the signed state — no cookie needed
    signed_state = s.dumps({"state": state, "cv": code_verifier})

    params = {
        "response_type": "code",
        "client_id": os.getenv("CLIENT_ID"),
        "redirect_uri": os.getenv("REDIRECT_URI"),
        "scope": "tweet.read users.read bookmark.read offline.access",
        "state": signed_state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    auth_url = f"https://twitter.com/i/oauth2/authorize?{urlencode(params)}"
    return RedirectResponse(url=auth_url)


@app.get("/auth/callback")
async def auth_callback(request: Request, code: str = None, state: str = None):
    if not state:
        raise HTTPException(400, "OAuth state missing")

    try:
        oauth_data = s.loads(state)
    except BadSignature:
        raise HTTPException(400, "Invalid OAuth state")

    token_data = await exchange_code_for_token(code, oauth_data["cv"])
    access_token = token_data["access_token"]

    user_info = await get_user_info(access_token)
    user_id = await save_user(user_info["id"], user_info["username"], access_token)

    # Auto-sync if this is a new account (no bookmarks yet)
    existing_count = await count_bookmarks(user_id)
    if existing_count == 0:
        try:
            await _do_sync(user_id, access_token)
        except Exception:
            pass  # Don't block login if sync fails

    response = RedirectResponse(url="/dashboard")
    response.set_cookie(
        "session",
        s.dumps({"user_id": user_id}),
        max_age=86400 * 30,
        httponly=True,
        samesite="lax",
        secure=_SECURE_COOKIE,
    )
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

    user = await get_user_by_id(session["user_id"])
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "categories": categories,
            "total": len(bookmarks),
            "subscription_status": user.get("subscription_status", "free") if user else "free",
        },
    )


@app.post("/sync")
async def sync_bookmarks(request: Request):
    session = get_session(request)
    if not session:
        return JSONResponse({"error": "Not logged in"}, status_code=401)

    try:
        user = await get_user_by_id(session["user_id"])
        if not user:
            return JSONResponse({"error": "User not found"}, status_code=404)

        result = await _do_sync(user["id"], user["access_token"])
        msg = f"Done! {result['new']} new bookmark(s) added." if result["new"] else "Already up to date."
        return JSONResponse({"message": msg, "count": result["count"], "new": result["new"]})

    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {str(e)}"}, status_code=500)


@app.get("/bookmark/{tweet_id}", response_class=HTMLResponse)
async def bookmark_detail(request: Request, tweet_id: str):
    session = get_session(request)
    if not session:
        return RedirectResponse(url="/")

    bookmark = await get_bookmark(session["user_id"], tweet_id)
    if not bookmark:
        raise HTTPException(404, "Bookmark not found")

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
        raise HTTPException(404, "Analysis not found")

    analysis = bookmark["deep_analysis"]
    lines = [
        f"# {bookmark['summary'] or bookmark['text'][:80]}",
        f"\n**Category:** {bookmark['category']} / {bookmark['subcategory']}",
        f"**Difficulty:** {bookmark['difficulty']}  |  **Type:** {bookmark['content_type']}",
        f"\n## Summary\n{analysis.get('detailed_summary', '')}",
        f"\n## Why It Matters\n{analysis.get('why_it_matters', '')}",
    ]

    if analysis.get("prerequisites"):
        lines.append("\n## Prerequisites")
        for p in analysis["prerequisites"]:
            lines.append(f"- {p}")

    if analysis.get("what_to_do"):
        lines.append("\n## What To Do")
        for step in analysis["what_to_do"]:
            lines.append(f"\n**{step['step']}. {step['action']}**")
            lines.append(f"{step['detail']}")

    if analysis.get("roadmap"):
        lines.append("\n## Learning Roadmap")
        for phase in analysis["roadmap"]:
            lines.append(f"\n### {phase['phase']} ({phase.get('duration', '')})")
            for step in phase.get("steps", []):
                lines.append(f"- [ ] {step}")

    if analysis.get("resources"):
        lines.append("\n## Resources")
        for r in analysis["resources"]:
            lines.append(f"- {r}")

    if bookmark.get("notes"):
        lines.append(f"\n## My Notes\n{bookmark['notes']}")

    content = "\n".join(lines)
    return PlainTextResponse(
        content,
        headers={"Content-Disposition": f"attachment; filename=roadmap-{tweet_id}.md"}
    )


@app.get("/pricing", response_class=HTMLResponse)
async def pricing(request: Request):
    session = get_session(request)
    user = None
    if session:
        user = await get_user_by_id(session["user_id"])
    return templates.TemplateResponse("pricing.html", {
        "request": request,
        "logged_in": bool(session),
        "user_id": session["user_id"] if session else None,
        "subscription_status": user.get("subscription_status", "free") if user else "free",
        "paddle_client_token": PADDLE_CLIENT_TOKEN,
        "paddle_price_monthly": PADDLE_PRICE_MONTHLY,
        "paddle_price_annual": PADDLE_PRICE_ANNUAL,
        "paddle_sandbox": PADDLE_SANDBOX,
    })


@app.post("/webhooks/paddle")
async def paddle_webhook(request: Request):
    sig_header = request.headers.get("Paddle-Signature", "")
    raw_body = await request.body()

    if PADDLE_WEBHOOK_SECRET and not _verify_paddle_signature(sig_header, raw_body):
        raise HTTPException(status_code=403, detail="Invalid Paddle signature")

    payload = await request.json()
    event_type = payload.get("event_type", "")
    data = payload.get("data", {})

    # Extract common fields
    sub_id = data.get("id") if event_type.startswith("subscription") else data.get("subscription_id")
    customer_id = data.get("customer_id", "")
    status = data.get("status", "")
    next_billed_at = data.get("next_billed_at") or data.get("current_billing_period", {}).get("ends_at")

    # Map Paddle status to our internal status
    STATUS_MAP = {
        "active": "active",
        "trialing": "trialing",
        "past_due": "past_due",
        "paused": "paused",
        "canceled": "free",
    }

    if event_type in ("subscription.created", "subscription.activated", "subscription.updated", "subscription.trialing"):
        # Find user by customer_id
        user = await get_user_by_paddle_customer(customer_id)
        if not user:
            # Try to find by custom_data user_id passed at checkout
            custom_data = data.get("custom_data") or {}
            user_id = custom_data.get("user_id")
            if user_id:
                user = await get_user_by_id(int(user_id))

        if user:
            mapped_status = STATUS_MAP.get(status, "active")
            # Determine plan from items
            items = data.get("items", [])
            plan = None
            if items:
                price_id = items[0].get("price", {}).get("id", "")
                if price_id == PADDLE_PRICE_ANNUAL:
                    plan = "annual"
                elif price_id == PADDLE_PRICE_MONTHLY:
                    plan = "monthly"
            await update_subscription(
                user["id"], customer_id, sub_id or "",
                mapped_status, plan, next_billed_at,
            )

    elif event_type in ("subscription.paused", "subscription.past_due"):
        mapped = "paused" if event_type == "subscription.paused" else "past_due"
        if sub_id:
            await update_subscription_by_sub_id(sub_id, mapped, next_billed_at)

    elif event_type == "subscription.canceled":
        if sub_id:
            await update_subscription_by_sub_id(sub_id, "free", None)

    elif event_type == "subscription.resumed":
        if sub_id:
            await update_subscription_by_sub_id(sub_id, "active", next_billed_at)

    elif event_type == "transaction.completed":
        # Ensure subscription is marked active after successful payment
        sub_id_from_tx = data.get("subscription_id")
        if sub_id_from_tx:
            await update_subscription_by_sub_id(sub_id_from_tx, "active", next_billed_at)

    return {"status": "ok"}


@app.get("/account/portal")
async def customer_portal(request: Request):
    """Redirect logged-in user to their Paddle customer portal."""
    session = get_session(request)
    if not session:
        return RedirectResponse(url="/pricing")

    user = await get_user_by_id(session["user_id"])
    if not user or not user.get("paddle_customer_id"):
        return RedirectResponse(url="/pricing")

    paddle_api_base = (
        "https://sandbox-api.paddle.com" if PADDLE_SANDBOX else "https://api.paddle.com"
    )
    customer_id = user["paddle_customer_id"]
    sub_id = user.get("paddle_subscription_id")

    body = {"subscription_ids": [sub_id]} if sub_id else {}
    async with httpx.AsyncClient() as http:
        resp = await http.post(
            f"{paddle_api_base}/customers/{customer_id}/portal-sessions",
            headers={"Authorization": f"Bearer {os.getenv('PADDLE_API_SECRET_KEY', '')}"},
            json=body,
        )

    if resp.status_code != 201:
        raise HTTPException(500, "Could not generate portal session")

    portal_url = resp.json()["data"]["urls"]["general"]["overview"]
    return RedirectResponse(url=portal_url)


@app.get("/subscription/status")
async def subscription_status(request: Request):
    session = get_session(request)
    if not session:
        return JSONResponse({"status": "free"})
    user = await get_user_by_id(session["user_id"])
    return JSONResponse({
        "status": user.get("subscription_status", "free") if user else "free",
        "plan": user.get("subscription_plan") if user else None,
        "ends_at": user.get("subscription_ends_at") if user else None,
    })


@app.get("/privacy", response_class=HTMLResponse)
async def privacy(request: Request):
    return templates.TemplateResponse("privacy.html", {"request": request})


@app.get("/terms", response_class=HTMLResponse)
async def terms(request: Request):
    return templates.TemplateResponse("terms.html", {"request": request})


@app.get("/refund", response_class=HTMLResponse)
async def refund(request: Request):
    return templates.TemplateResponse("refund.html", {"request": request})


@app.post("/account/delete")
async def delete_account(request: Request):
    session = get_session(request)
    if not session:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    
    await delete_user(session["user_id"])
    
    response = JSONResponse({"ok": True, "message": "Account deleted successfully"})
    response.delete_cookie("session")
    return response


@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/")
    response.delete_cookie("session")
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
