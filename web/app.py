import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Dict, Any

import requests
from flask import Flask, flash, redirect, render_template, request, url_for


APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.abspath(os.path.join(APP_DIR, "..", "moderation.db"))
DB_PATH = os.getenv("MOD_DB_PATH", DEFAULT_DB_PATH)
DEFAULT_TOKEN_STORE_PATH = os.path.abspath(os.path.join(APP_DIR, "..", ".bot_token"))
TOKEN_STORE_PATH = os.getenv("TOKEN_STORE_PATH", DEFAULT_TOKEN_STORE_PATH)

app = Flask(__name__)
app.secret_key = os.getenv("WEB_SECRET_KEY", "change-me-in-production")


def resolve_telegram_token() -> str:
    env_token = (os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TOKEN") or "").strip()
    if env_token:
        return env_token
    try:
        if os.path.exists(TOKEN_STORE_PATH):
            with open(TOKEN_STORE_PATH, "r", encoding="utf-8") as f:
                return f.read().strip()
    except Exception:
        return ""
    return ""


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_web_columns() -> None:
    conn = get_db_connection()
    columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(moderation_events)").fetchall()
    }
    if "message_status" not in columns:
        conn.execute("ALTER TABLE moderation_events ADD COLUMN message_status TEXT")
    if "deleted_at" not in columns:
        conn.execute("ALTER TABLE moderation_events ADD COLUMN deleted_at TEXT")
    if "delete_error" not in columns:
        conn.execute("ALTER TABLE moderation_events ADD COLUMN delete_error TEXT")
    conn.commit()
    conn.close()


def parse_date(date_str: str | None) -> str | None:
    if not date_str:
        return None
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return dt.strftime("%d.%m.%Y %H:%M:%S")
    except Exception:
        return date_str


def get_risk_color(risk_level: str | None) -> str:
    level = (risk_level or "low").lower()
    if level == "high":
        return "danger"
    if level == "med":
        return "warning"
    return "success"


def chat_kind(chat_id: int | None) -> str:
    if chat_id is None:
        return "unknown"
    if chat_id > 0:
        return "private"
    cid = str(chat_id)
    if cid.startswith("-100"):
        return "channel/supergroup"
    return "group"


def telegram_message_link(chat_id: int | None, message_id: int | None) -> str | None:
    if chat_id is None or message_id is None:
        return None
    cid = str(chat_id)
    if cid.startswith("-100"):
        # Формат для supergroup/channel: https://t.me/c/<internal_chat_id>/<message_id>
        internal_id = cid[4:]
        return f"https://t.me/c/{internal_id}/{message_id}"
    return None


def enrich_event(event: sqlite3.Row) -> Dict[str, Any]:
    event_dict = dict(event)

    try:
        categories = json.loads(event_dict.get("categories_json") or "{}")
    except Exception:
        categories = {}

    try:
        scores = json.loads(event_dict.get("scores_json") or "{}")
    except Exception:
        scores = {}

    event_dict["categories"] = categories
    event_dict["scores"] = scores
    event_dict["created_at_formatted"] = parse_date(event_dict.get("created_at"))
    event_dict["review_at_formatted"] = parse_date(event_dict.get("review_at"))
    event_dict["deleted_at_formatted"] = parse_date(event_dict.get("deleted_at"))
    event_dict["risk_color"] = get_risk_color(event_dict.get("decision"))
    event_dict["chat_kind"] = chat_kind(event_dict.get("chat_id"))
    event_dict["chat_link"] = telegram_message_link(event_dict.get("chat_id"), event_dict.get("message_id"))
    status = (event_dict.get("message_status") or "").strip().lower()
    if not status:
        status = "present" if event_dict.get("message_id") else "unknown"
    event_dict["message_status"] = status
    return event_dict


def get_bot_status() -> Dict[str, Any]:
    conn = get_db_connection()
    try:
        rows = conn.execute("SELECT key, value FROM bot_status").fetchall()
    except sqlite3.OperationalError:
        rows = []
    conn.close()

    raw = {row["key"]: row["value"] for row in rows}
    now_epoch = int(datetime.now(timezone.utc).timestamp())

    try:
        next_run_epoch = int(raw.get("next_run_epoch", "0") or 0)
    except ValueError:
        next_run_epoch = 0
    try:
        queue_size = int(raw.get("queue_size", "0") or 0)
    except ValueError:
        queue_size = 0
    try:
        batch_interval_seconds = int(raw.get("batch_interval_seconds", "30") or 30)
    except ValueError:
        batch_interval_seconds = 30

    seconds_left = max(0, next_run_epoch - now_epoch) if next_run_epoch else None
    return {
        "next_run_epoch": next_run_epoch,
        "queue_size": queue_size,
        "batch_interval_seconds": batch_interval_seconds,
        "seconds_left": seconds_left,
    }


@app.context_processor
def inject_global_status() -> Dict[str, Any]:
    return {"bot_status": get_bot_status()}


def delete_telegram_message(chat_id: int, message_id: int) -> tuple[bool, str]:
    telegram_bot_token = resolve_telegram_token()
    if not telegram_bot_token:
        return False, "Не задан TELEGRAM_BOT_TOKEN (или TOKEN)"

    url = f"https://api.telegram.org/bot{telegram_bot_token}/deleteMessage"
    try:
        response = requests.post(
            url,
            json={"chat_id": chat_id, "message_id": message_id},
            timeout=15,
        )
    except requests.RequestException as exc:
        return False, f"Ошибка сети: {exc}"

    try:
        payload = response.json()
    except Exception:
        payload = {"ok": False, "description": response.text[:200]}

    if response.status_code != 200 or not payload.get("ok"):
        description = payload.get("description", f"HTTP {response.status_code}")
        return False, f"Telegram API: {description}"

    return True, "Сообщение удалено в Telegram"


def update_message_status(event_id: int, status: str, error: str | None = None) -> None:
    conn = get_db_connection()
    if status == "deleted":
        conn.execute(
            """
            UPDATE moderation_events
            SET message_status = ?, deleted_at = ?, delete_error = NULL
            WHERE id = ?
            """,
            (status, datetime.now(timezone.utc).isoformat(), event_id),
        )
    else:
        conn.execute(
            """
            UPDATE moderation_events
            SET message_status = ?, delete_error = ?
            WHERE id = ?
            """,
            (status, error, event_id),
        )
    conn.commit()
    conn.close()


@app.route("/")
def index():
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 50, type=int)
    if per_page > 200:
        per_page = 200

    offset = (page - 1) * per_page

    conn = get_db_connection()
    total_count = conn.execute("SELECT COUNT(*) as count FROM moderation_events").fetchone()["count"]

    events = conn.execute(
        """
        SELECT
            id,
            created_at,
            chat_id,
            message_id,
            username,
            text_excerpt,
            categories_json,
            scores_json,
            content_type,
            decision,
            flagged,
            reason,
            review_label,
            message_status,
            deleted_at,
            delete_error
        FROM moderation_events
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?
        """,
        (per_page, offset),
    ).fetchall()
    conn.close()

    events_list = [enrich_event(event) for event in events]
    total_pages = (total_count + per_page - 1) // per_page

    return render_template(
        "index.html",
        events=events_list,
        page=page,
        per_page=per_page,
        total_pages=total_pages,
        total_count=total_count,
    )


@app.route("/event/<int:event_id>")
def event_detail(event_id: int):
    conn = get_db_connection()
    event = conn.execute("SELECT * FROM moderation_events WHERE id = ?", (event_id,)).fetchone()
    conn.close()

    if event is None:
        return "Событие не найдено", 404

    return render_template("event_detail.html", event=enrich_event(event))


@app.route("/search")
def search():
    query = request.args.get("q", "").strip()
    if not query:
        return index()

    conn = get_db_connection()
    events = conn.execute(
        """
        SELECT
            id,
            created_at,
            chat_id,
            message_id,
            username,
            text_excerpt,
            categories_json,
            scores_json,
            content_type,
            decision,
            flagged,
            reason,
            review_label,
            message_status,
            deleted_at,
            delete_error
        FROM moderation_events
        WHERE
            username LIKE ? OR
            text_excerpt LIKE ? OR
            reason LIKE ? OR
            categories_json LIKE ?
        ORDER BY created_at DESC
        LIMIT 200
        """,
        (f"%{query}%", f"%{query}%", f"%{query}%", f"%{query}%"),
    ).fetchall()
    conn.close()

    events_list = [enrich_event(event) for event in events]
    return render_template("search.html", events=events_list, query=query, count=len(events_list))


@app.post("/event/<int:event_id>/delete-message")
def delete_message(event_id: int):
    conn = get_db_connection()
    event = conn.execute(
        "SELECT id, chat_id, message_id FROM moderation_events WHERE id = ?",
        (event_id,),
    ).fetchone()
    conn.close()

    next_url = request.form.get("next") or request.referrer or url_for("index")

    if event is None:
        flash("Событие не найдено", "danger")
        return redirect(next_url)

    chat_id = event["chat_id"]
    message_id = event["message_id"]
    if chat_id is None or message_id is None:
        update_message_status(event_id, "unknown", "missing chat_id/message_id")
        flash("Нельзя удалить: у события нет chat_id/message_id", "warning")
        return redirect(next_url)

    ok, message = delete_telegram_message(chat_id, message_id)
    if ok:
        update_message_status(event_id, "deleted")
    else:
        update_message_status(event_id, "delete_failed", message)
    flash(message, "success" if ok else "danger")
    return redirect(next_url)


ensure_web_columns()


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
