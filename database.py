import aiosqlite
import json

DB_PATH = "bookmarks.db"


async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                x_user_id TEXT UNIQUE,
                username TEXT,
                access_token TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS bookmarks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                tweet_id TEXT,
                text TEXT,
                category TEXT,
                subcategory TEXT,
                content_type TEXT,
                difficulty TEXT,
                action TEXT,
                summary TEXT,
                key_points TEXT,
                tags TEXT,
                priority INTEGER DEFAULT 3,
                is_evergreen INTEGER DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, tweet_id),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        # Migrate existing table if columns missing
        for col, definition in [
            ("content_type", "TEXT"),
            ("difficulty", "TEXT"),
            ("action", "TEXT"),
            ("key_points", "TEXT"),
            ("is_evergreen", "INTEGER DEFAULT 1"),
            ("deep_analysis", "TEXT"),
            ("is_read", "INTEGER DEFAULT 0"),
            ("read_at", "DATETIME"),
            ("notes", "TEXT"),
            ("roadmap_progress", "TEXT"),
            ("mvp_prompt", "TEXT"),
        ]:
            try:
                await db.execute(f"ALTER TABLE bookmarks ADD COLUMN {col} {definition}")
            except Exception:
                pass
        await db.commit()


async def save_user(x_user_id: str, username: str, access_token: str) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO users (x_user_id, username, access_token)
            VALUES (?, ?, ?)
            ON CONFLICT(x_user_id) DO UPDATE SET
                access_token = excluded.access_token,
                username = excluded.username
            """,
            (x_user_id, username, access_token),
        )
        await db.commit()
        cursor = await db.execute(
            "SELECT id FROM users WHERE x_user_id = ?", (x_user_id,)
        )
        row = await cursor.fetchone()
        return row[0]


async def get_user_by_id(user_id: int) -> dict:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None


async def save_bookmarks(user_id: int, bookmarks: list):
    async with aiosqlite.connect(DB_PATH) as db:
        for bm in bookmarks:
            await db.execute(
                """
                INSERT INTO bookmarks
                    (user_id, tweet_id, text, category, subcategory,
                     content_type, difficulty, action, summary, key_points,
                     tags, priority, is_evergreen)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, tweet_id) DO UPDATE SET
                    category = excluded.category,
                    subcategory = excluded.subcategory,
                    content_type = excluded.content_type,
                    difficulty = excluded.difficulty,
                    action = excluded.action,
                    summary = excluded.summary,
                    key_points = excluded.key_points,
                    tags = excluded.tags,
                    priority = excluded.priority,
                    is_evergreen = excluded.is_evergreen
                """,
                (
                    user_id,
                    bm["id"],
                    bm["text"],
                    bm.get("category", "Other"),
                    bm.get("subcategory", ""),
                    bm.get("content_type", "Article"),
                    bm.get("difficulty", "Intermediate"),
                    bm.get("action", "Read"),
                    bm.get("summary", ""),
                    json.dumps(bm.get("key_points", [])),
                    json.dumps(bm.get("tags", [])),
                    bm.get("priority", 3),
                    1 if bm.get("is_evergreen", True) else 0,
                ),
            )
        await db.commit()


async def get_bookmark(user_id: int, tweet_id: str) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM bookmarks WHERE user_id = ? AND tweet_id = ?",
            (user_id, tweet_id),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        d = dict(row)
        d["tags"] = json.loads(d["tags"]) if d["tags"] else []
        d["key_points"] = json.loads(d["key_points"]) if d.get("key_points") else []
        d["deep_analysis"] = json.loads(d["deep_analysis"]) if d.get("deep_analysis") else None
        d["roadmap_progress"] = json.loads(d["roadmap_progress"]) if d.get("roadmap_progress") else {}
        d["mvp_prompt"] = d.get("mvp_prompt") or None
        return d


async def get_similar_bookmarks(user_id: int, tweet_id: str, category: str, tags: list) -> list:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """
            SELECT * FROM bookmarks
            WHERE user_id = ? AND tweet_id != ? AND category = ?
            ORDER BY priority DESC
            LIMIT 6
            """,
            (user_id, tweet_id, category),
        )
        rows = await cursor.fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["tags"] = json.loads(d["tags"]) if d["tags"] else []
            d["key_points"] = json.loads(d["key_points"]) if d.get("key_points") else []
            result.append(d)
        return result


async def save_deep_analysis(user_id: int, tweet_id: str, analysis: dict):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE bookmarks SET deep_analysis = ? WHERE user_id = ? AND tweet_id = ?",
            (json.dumps(analysis), user_id, tweet_id),
        )
        await db.commit()


async def toggle_read(user_id: int, tweet_id: str) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "SELECT is_read FROM bookmarks WHERE user_id = ? AND tweet_id = ?",
            (user_id, tweet_id),
        )
        row = await cursor.fetchone()
        new_val = 0 if row and row[0] else 1
        read_at = "CURRENT_TIMESTAMP" if new_val else "NULL"
        await db.execute(
            f"UPDATE bookmarks SET is_read = ?, read_at = {read_at} WHERE user_id = ? AND tweet_id = ?",
            (new_val, user_id, tweet_id),
        )
        await db.commit()
        return bool(new_val)


async def save_notes(user_id: int, tweet_id: str, notes: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE bookmarks SET notes = ? WHERE user_id = ? AND tweet_id = ?",
            (notes, user_id, tweet_id),
        )
        await db.commit()


async def save_roadmap_progress(user_id: int, tweet_id: str, progress: dict):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE bookmarks SET roadmap_progress = ? WHERE user_id = ? AND tweet_id = ?",
            (json.dumps(progress), user_id, tweet_id),
        )
        await db.commit()


async def save_mvp_prompt(user_id: int, tweet_id: str, prompt: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE bookmarks SET mvp_prompt = ? WHERE user_id = ? AND tweet_id = ?",
            (prompt, user_id, tweet_id),
        )
        await db.commit()


async def update_bookmark_meta(user_id: int, tweet_id: str, category: str, difficulty: str, priority: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """UPDATE bookmarks SET category = ?, difficulty = ?, priority = ?
               WHERE user_id = ? AND tweet_id = ?""",
            (category, difficulty, priority, user_id, tweet_id),
        )
        await db.commit()


async def get_bookmarks(user_id: int) -> list:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """
            SELECT * FROM bookmarks WHERE user_id = ?
            ORDER BY priority DESC, created_at DESC
            """,
            (user_id,),
        )
        rows = await cursor.fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["tags"] = json.loads(d["tags"]) if d["tags"] else []
            d["key_points"] = json.loads(d["key_points"]) if d.get("key_points") else []
            result.append(d)
        return result


async def delete_user(user_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        # First delete all user's bookmarks
        await db.execute("DELETE FROM bookmarks WHERE user_id = ?", (user_id,))
        # Then delete the user
        await db.execute("DELETE FROM users WHERE id = ?", (user_id,))
        await db.commit()
