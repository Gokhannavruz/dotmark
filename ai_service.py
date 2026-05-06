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
    ]) if similar else "None"

    response = await asyncio.to_thread(
        client.messages.create,
        model="claude-sonnet-4-6",
        max_tokens=3000,
        messages=[{
            "role": "user",
            "content": f"""A software developer bookmarked this tweet. Perform a deep analysis.

Tweet:
{bookmark['text']}

Current Analysis:
- Category: {bookmark.get('category')} / {bookmark.get('subcategory')}
- Type: {bookmark.get('content_type')}
- Difficulty: {bookmark.get('difficulty')}

Similar Bookmarks:
{similar_text}

Return only JSON in this format (nothing else):
{{
  "detailed_summary": "A comprehensive 3-4 sentence summary of this content in English",
  "why_it_matters": "Why is this important? What practical value does it have? (2-3 sentences)",
  "prerequisites": ["Prerequisite 1", "Prerequisite 2"],
  "what_to_do": [
    {{"step": 1, "action": "First thing to do", "detail": "How to do it"}},
    {{"step": 2, "action": "Second thing to do", "detail": "How to do it"}},
    {{"step": 3, "action": "Third thing to do", "detail": "How to do it"}}
  ],
  "roadmap": [
    {{"phase": "Beginner", "steps": ["Step 1", "Step 2"], "duration": "1 week"}},
    {{"phase": "Intermediate", "steps": ["Step 3", "Step 4"], "duration": "2 weeks"}},
    {{"phase": "Advanced", "steps": ["Step 5", "Step 6"], "duration": "1 month"}}
  ],
  "resources": ["Related resource or tool suggestion 1", "Related resource suggestion 2"],
  "connection_to_similar": "How does this bookmark connect to other bookmarks? (if applicable)"
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
        "id": "platform",
        "question": "Which platform(s) are you building for?",
        "why": "Determines the tech stack selection",
        "type": "radio",
        "options": ["Mobile only", "Web only", "Mobile + Web"],
        "default": "Mobile only",
    },
    {
        "id": "prog_lang",
        "question": "Which programming language / framework?",
        "why": "The project scaffold will be generated for this language",
        "type": "select",
        "options": ["Flutter (Dart)", "Swift (iOS)", "Kotlin (Android)", "React Native", "Next.js (TypeScript)", "Vue.js", "FastAPI (Python)", "Node.js + Express"],
        "default": "Flutter (Dart)",
    },
    {
        "id": "database",
        "question": "Which database would you like to use?",
        "why": "Shapes the data model and backend architecture",
        "type": "select",
        "options": ["Supabase (PostgreSQL)", "Firebase Firestore", "SQLite (local)", "PostgreSQL (self-hosted)", "MongoDB", "PocketBase"],
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
            "content": f"""A developer wants to build a product based on this bookmark.

Bookmark:
{bookmark['text']}

Summary: {bookmark.get('summary', '')}
Category: {bookmark.get('category', '')}

Your task:
1. Determine what type of project this is (mobile / web / both)
2. Generate up to 3 project-SPECIFIC questions DIFFERENT from the ones already asked
3. Questions already being asked: platform, programming language, database — DO NOT repeat these

Return only JSON:
{{
  "project_type": "mobile | web | both",
  "project_summary": "One-sentence English summary of the project",
  "extra_questions": [
    {{
      "id": "unique_id",
      "question": "Question text in English",
      "why": "Why you are asking this (brief)",
      "type": "select | radio | text",
      "options": ["Option 1", "Option 2"],
      "default": "Default"
    }}
  ]
}}

Examples of project-specific questions:
- Does the app need user authentication? → Yes / No
- Should it work offline? → Yes / No
- Does it need a custom backend API? → Yes (own server) / No (Supabase/Firebase is enough)
- Will there be push notifications? → Yes / No
- Paid subscription / in-app purchase? → Yes / No
- Does it need real-time updates? → Yes / No"""
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
    prog_lang = answers.get("Which programming language / framework?", "Flutter (Dart)")
    database = answers.get("Which database would you like to use?", "Supabase (PostgreSQL)")
    is_mobile = project_type in ("mobile", "both")
    is_web = project_type in ("web", "both")

    response = await asyncio.to_thread(
        client.messages.create,
        model="claude-sonnet-4-6",
        max_tokens=4000,
        messages=[{
            "role": "user",
            "content": f"""A developer wants to build this idea.

Bookmark:
{bookmark['text']}

Summary: {bookmark.get('summary', '')}
Project type: {project_type}
Primary language/framework: {prog_lang}
Database: {database}

All user preferences:
{answers_text}

Write the entire prompt and all explanations in English.

Generate a complete MVP development prompt to be pasted directly into Claude Code or another AI coding assistant.

Format:
---
## 🚀 MVP Development Prompt

### Project Definition
[What it does, who it's for, core value proposition]

### Platform & Tech Stack
[Finalized stack based on user preferences — be specific, leave no ambiguity]

### MVP Features (v1.0)
[Core only, no scope creep]

### Screens / Pages
[For each: what it shows, what the user can do]

### Data Models
[In a code block]

### Folder & Architecture Structure
[In a code block — clean architecture]

### Development Order
[What to build first, what next — sequential steps]

---
## 🤖 Prompt to Paste into Claude Code

[Single-use, all-inclusive, unambiguous, ready-to-run prompt.
Include state management, navigation, all screens, auth (if requested), database integration.]
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
            "content": f"""You are a software developer assistant. Deeply analyze the following Twitter bookmarks.

Categories: {', '.join(CATEGORIES)}
Content Types: {', '.join(CONTENT_TYPES)}
Difficulty Levels: {', '.join(DIFFICULTIES)}
Action Types: {', '.join(ACTIONS)}

Tweets:
{tweets_text}

Analyze each tweet in the following JSON format. Return only a JSON array, nothing else:
[
  {{
    "id": "tweet_id",
    "category": "main category",
    "subcategory": "subcategory (e.g. SwiftUI, Jetpack Compose, FastAPI)",
    "content_type": "content type",
    "difficulty": "difficulty level",
    "action": "recommended action",
    "summary": "1 clear English sentence — explain what it teaches or what it is",
    "key_points": [
      "Main idea or thing to learn 1",
      "Main idea or thing to learn 2"
    ],
    "tags": ["tag1", "tag2", "tag3"],
    "priority": 3,
    "is_evergreen": true
  }}
]

Rules:
- priority: 1=general info, 3=useful, 5=must learn
- is_evergreen: false means time-sensitive content (news, announcements, discounts, etc.)
- key_points: write concrete, actionable items
- subcategory: keep as specific as possible"""
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
