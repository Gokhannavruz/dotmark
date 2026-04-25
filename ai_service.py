import anthropic
import asyncio
import json
import os
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

CATEGORIES = [
    "iOS", "Android", "Flutter", "React Native",
    "Backend", "DevOps", "AI/ML", "Design",
    "Career", "Tools", "Web", "Security", "Other"
]

CONTENT_TYPES = ["Tutorial", "Tool", "Article", "Thread", "Tip", "Resource", "Opinion", "Job"]
DIFFICULTIES = ["Beginner", "Intermediate", "Advanced"]
ACTIONS = ["Read", "Try", "Watch", "Practice", "Save"]


async def deep_analyze(bookmark: dict, similar: list) -> dict:
    similar_text = "\n".join([
        f"- {b['summary'] or b['text'][:100]}" for b in similar[:5]
    ]) if similar else "Yok"

    response = await asyncio.to_thread(
        client.messages.create,
        model="claude-sonnet-4-6",
        max_tokens=3000,
        messages=[{
            "role": "user",
            "content": f"""Bir yazılım geliştirici bu tweet'i bookmark'lamış. Derinlemesine analiz yap.

Tweet:
{bookmark['text']}

Mevcut Analiz:
- Kategori: {bookmark.get('category')} / {bookmark.get('subcategory')}
- Tür: {bookmark.get('content_type')}
- Zorluk: {bookmark.get('difficulty')}

Benzer Bookmark'lar:
{similar_text}

Şu JSON formatında döndür (başka hiçbir şey yazma):
{{
  "detailed_summary": "Bu içeriğin 3-4 cümlelik kapsamlı Türkçe özeti",
  "why_it_matters": "Bu neden önemli? Pratikte ne işe yarar? (2-3 cümle)",
  "prerequisites": ["Önkoşul 1", "Önkoşul 2"],
  "what_to_do": [
    {{"step": 1, "action": "Yapılacak ilk şey", "detail": "Nasıl yapılacak"}},
    {{"step": 2, "action": "Yapılacak ikinci şey", "detail": "Nasıl yapılacak"}},
    {{"step": 3, "action": "Yapılacak üçüncü şey", "detail": "Nasıl yapılacak"}}
  ],
  "roadmap": [
    {{"phase": "Başlangıç", "steps": ["Adım 1", "Adım 2"], "duration": "1 hafta"}},
    {{"phase": "Orta Seviye", "steps": ["Adım 3", "Adım 4"], "duration": "2 hafta"}},
    {{"phase": "İleri Seviye", "steps": ["Adım 5", "Adım 6"], "duration": "1 ay"}}
  ],
  "resources": ["İlgili kaynak veya araç önerisi 1", "İlgili kaynak önerisi 2"],
  "connection_to_similar": "Bu bookmark diğer bookmark'larla nasıl bağlantılı? (varsa)"
}}"""
        }]
    )

    try:
        text = response.content[0].text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except (json.JSONDecodeError, IndexError):
        return {}


BASE_QUESTIONS = [
    {
        "id": "prompt_lang",
        "question": "Prompt hangi dilde olsun?",
        "why": "Oluşturulacak promptun dili",
        "type": "radio",
        "options": ["Türkçe", "English"],
        "default": "Türkçe",
    },
    {
        "id": "platform",
        "question": "Hangi platform(lar) için geliştiriyorsun?",
        "why": "Tech stack seçimini belirler",
        "type": "radio",
        "options": ["Sadece Mobil", "Sadece Web", "Mobil + Web"],
        "default": "Sadece Mobil",
    },
    {
        "id": "prog_lang",
        "question": "Hangi programlama dili / framework?",
        "why": "Proje iskeleti bu dile göre oluşturulur",
        "type": "select",
        "options": ["Flutter (Dart)", "Swift (iOS)", "Kotlin (Android)", "React Native", "Next.js (TypeScript)", "Vue.js", "FastAPI (Python)", "Node.js + Express"],
        "default": "Flutter (Dart)",
    },
    {
        "id": "database",
        "question": "Hangi veritabanı kullanmak istersin?",
        "why": "Veri modeli ve backend mimarisi buna göre şekillenir",
        "type": "select",
        "options": ["Supabase (PostgreSQL)", "Firebase Firestore", "SQLite (lokal)", "PostgreSQL (kendi sunucu)", "MongoDB", "PocketBase"],
        "default": "Supabase (PostgreSQL)",
    },
]

async def generate_mvp_questions(bookmark: dict) -> dict:
    response = await asyncio.to_thread(
        client.messages.create,
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=[{
            "role": "user",
            "content": f"""Bir yazılımcı bu bookmark'tan bir ürün geliştirmek istiyor.

Bookmark:
{bookmark['text']}

Özet: {bookmark.get('summary', '')}
Kategori: {bookmark.get('category', '')}

Görevin:
1. Bu ürünün ne tür bir proje olduğunu belirle (mobile / web / both)
2. Bu projeye ÖZEL, aşağıdakilerden FARKLI maksimum 3 soru üret
3. Zaten sorulacak sorular: prompt dili, platform, programlama dili, veritabanı — bunları TEKRARLAMA

Sadece JSON döndür:
{{
  "project_type": "mobile | web | both",
  "project_summary": "Projenin 1 cümlelik Türkçe özeti",
  "extra_questions": [
    {{
      "id": "benzersiz_id",
      "question": "Türkçe soru metni",
      "why": "Neden soruyorsun (kısa)",
      "type": "select | radio | text",
      "options": ["Seçenek 1", "Seçenek 2"],
      "default": "Varsayılan"
    }}
  ]
}}

Proje özelinde sorulabilecek örnekler:
- Kullanıcı girişi (auth) gerekiyor mu? → Evet / Hayır
- Offline çalışma gerekiyor mu? → Evet / Hayır
- Backend API gerekiyor mu? → Evet (kendi server) / Hayır (Supabase/Firebase yeterli)
- Bildirim (push notification) olacak mı? → Evet / Hayır
- Ücretli abonelik / in-app purchase? → Evet / Hayır
- Gerçek zamanlı güncelleme gerekiyor mu? → Evet / Hayır"""
        }]
    )

    try:
        text = response.content[0].text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text)
    except (json.JSONDecodeError, IndexError):
        data = {"project_type": "mobile", "project_summary": bookmark.get("summary", ""), "extra_questions": []}

    return {
        "project_type": data.get("project_type", "mobile"),
        "project_summary": data.get("project_summary", bookmark.get("summary", "")),
        "questions": BASE_QUESTIONS + data.get("extra_questions", []),
    }


async def generate_mvp_prompt(bookmark: dict, answers: dict, project_type: str) -> str:
    answers_text = "\n".join([f"- {k}: {v}" for k, v in answers.items()])
    lang = answers.get("Prompt hangi dilde olsun?", "Türkçe")
    prog_lang = answers.get("Hangi programlama dili / framework?", "Flutter (Dart)")
    database = answers.get("Hangi veritabanı kullanmak istersin?", "Supabase (PostgreSQL)")
    is_mobile = project_type in ("mobile", "both")
    is_web = project_type in ("web", "both")

    prompt_lang_note = "Write the entire prompt and all explanations in English." if lang == "English" else "Promptu ve tüm açıklamaları Türkçe yaz."

    response = await asyncio.to_thread(
        client.messages.create,
        model="claude-sonnet-4-6",
        max_tokens=4000,
        messages=[{
            "role": "user",
            "content": f"""Bir yazılımcı bu fikri geliştirmek istiyor.

Bookmark:
{bookmark['text']}

Özet: {bookmark.get('summary', '')}
Proje tipi: {project_type}
Ana dil/framework: {prog_lang}
Veritabanı: {database}

Kullanıcının tüm tercihleri:
{answers_text}

{prompt_lang_note}

Doğrudan Claude Code veya başka bir AI kodlama asistanına yapıştırılacak, eksiksiz bir MVP geliştirme promptu oluştur.

Format:
---
## 🚀 MVP Geliştirme Promptu

### Proje Tanımı
[Ne yapıyor, kime hitap ediyor, temel değer önerisi]

### Platform & Tech Stack
[Kullanıcı tercihlerine göre kesinleşmiş stack - her şeyi ver, belirsizlik bırakma]

### MVP Özellikleri (v1.0)
[Sadece core, scope creep yok]

### Ekranlar / Sayfalar
[Her biri için: ne gösteriyor, kullanıcı ne yapabilir]

### Veri Modelleri
[Kod bloğu içinde]

### Klasör & Mimari Yapısı
[Kod bloğu içinde - clean architecture]

### Geliştirme Sırası
[Önce ne, sonra ne - sıralı adımlar]

---
## 🤖 Claude Code'a Yapıştırılacak Prompt

[Tek seferlik, her şeyi içeren, belirsizlik bırakmayan, direkt çalıştırılabilir prompt.
State management, navigation, tüm ekranlar, auth (isteniyorsa), veritabanı entegrasyonu dahil.]
---"""
        }]
    )

    return response.content[0].text.strip()


async def categorize_bookmarks(tweets: list) -> list:
    results = []
    batch_size = 10
    for i in range(0, len(tweets), batch_size):
        batch = tweets[i:i + batch_size]
        batch_results = await _categorize_batch(batch)
        results.extend(batch_results)
    return results


async def _categorize_batch(tweets: list) -> list:
    tweets_text = "\n\n".join([f"[{t['id']}] {t['text'][:600]}" for t in tweets])

    response = await asyncio.to_thread(
        client.messages.create,
        model="claude-sonnet-4-6",
        max_tokens=4000,
        messages=[{
            "role": "user",
            "content": f"""Sen bir yazılım geliştirici asistanısın. Aşağıdaki Twitter bookmark'larını derinlemesine analiz et.

Kategoriler: {', '.join(CATEGORIES)}
İçerik Türleri: {', '.join(CONTENT_TYPES)}
Zorluk Seviyeleri: {', '.join(DIFFICULTIES)}
Eylem Türleri: {', '.join(ACTIONS)}

Tweet'ler:
{tweets_text}

Her tweet için şu JSON formatında analiz yap. Sadece JSON array döndür, başka hiçbir şey yazma:
[
  {{
    "id": "tweet_id",
    "category": "ana kategori",
    "subcategory": "alt kategori (ör: SwiftUI, Jetpack Compose, FastAPI)",
    "content_type": "içerik türü",
    "difficulty": "zorluk seviyesi",
    "action": "önerilen eylem",
    "summary": "1 cümle net Türkçe özet - ne öğrettiğini veya ne olduğunu açıkla",
    "key_points": [
      "Ana fikir veya öğrenilecek şey 1",
      "Ana fikir veya öğrenilecek şey 2"
    ],
    "tags": ["tag1", "tag2", "tag3"],
    "priority": 3,
    "is_evergreen": true
  }}
]

Kurallar:
- priority: 1=genel bilgi, 3=faydalı, 5=mutlaka öğrenilmeli
- is_evergreen: false ise zaman hassas içerik (haber, duyuru, indirim vs)
- key_points: somut, uygulanabilir maddeler yaz
- subcategory: mümkün olduğunca spesifik tut"""
        }]
    )

    try:
        text = response.content[0].text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        categorized = json.loads(text)
        tweet_map = {t["id"]: t for t in tweets}
        for item in categorized:
            if item["id"] in tweet_map:
                item["text"] = tweet_map[item["id"]]["text"]
        return categorized
    except (json.JSONDecodeError, IndexError):
        return [
            {
                "id": t["id"],
                "text": t["text"],
                "category": "Other",
                "subcategory": "",
                "content_type": "Article",
                "difficulty": "Intermediate",
                "action": "Read",
                "summary": "",
                "key_points": [],
                "tags": [],
                "priority": 3,
                "is_evergreen": True,
            }
            for t in tweets
        ]
