import json
import logging
import os
import re
import sqlite3
import time
import base64
import subprocess
import threading
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import requests
import telebot
from telebot import types


DB_PATH = os.getenv("MOD_DB_PATH", "moderation.db")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llava:7b")
AUTO_START_OLLAMA = os.getenv("AUTO_START_OLLAMA", "1") == "1"
AUTO_PULL_OLLAMA_MODELS = os.getenv("AUTO_PULL_OLLAMA_MODELS", "0") == "1"
OLLAMA_WAIT_SECONDS = int(os.getenv("OLLAMA_WAIT_SECONDS", "20"))
RECHECK_ENABLED = os.getenv("RECHECK_ENABLED", "1") == "1"
RECHECK_INTERVAL_SECONDS = int(os.getenv("RECHECK_INTERVAL_SECONDS", "30"))
RECHECK_LOOKBACK_DAYS = int(os.getenv("RECHECK_LOOKBACK_DAYS", "3"))
TOKEN_STORE_PATH = os.getenv(
    "TOKEN_STORE_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".bot_token"),
)
AUTO_SAVE_TOKEN = os.getenv("AUTO_SAVE_TOKEN", "1") == "1"

# Классы для few-shot классификации.
LABELS = [
    x.strip()
    for x in os.getenv(
        "LABELS",
        "allowed,sexual,violence,gore,illegal,drugs,spam",
    ).split(",")
    if x.strip()
]

# Список запрещенных классов.
BLOCKED_LABELS = {
    x.strip().lower()
    for x in os.getenv(
        "BLOCKED_LABELS",
        "sexual,violence,gore,illegal,drugs",
    ).split(",")
    if x.strip()
}

BLOCK_THRESHOLD = float(os.getenv("BLOCK_THRESHOLD", "0.80"))
REVIEW_THRESHOLD = float(os.getenv("REVIEW_THRESHOLD", "0.55"))
# Новая шкала риска: low / med / high.
# Для обратной совместимости используем старые env-имена как пороги.
HIGH_THRESHOLD = BLOCK_THRESHOLD
MED_THRESHOLD = REVIEW_THRESHOLD

# Few-shot примеры: JSON строка вида
# [{"text":"...","label":"allowed"}, {"text":"...","label":"violence"}]
SAMPLES_JSON = os.getenv("SAMPLES_JSON", "")

# Маркеры "обычной торговли", чтобы не ловить авто/телефон как запрещенку.
SAFE_COMMERCE_HINTS = [
    x.strip().lower()
    for x in os.getenv(
        "SAFE_COMMERCE_HINTS",
        (
            "авто,машина,телефон,смартфон,айфон,iphone,android,ноутбук,ноут,"
            "планшет,компьютер,пк,видеокарта,квартира,дом,велосипед,коляска,"
            "продам,продаю,цена,торг,состояние,гарантия,чек,доставка,самовывоз"
        ),
    ).split(",")
    if x.strip()
]

# Явные маркеры запрещенного сбыта/употребления.
ILLICIT_HINTS = [
    x.strip().lower()
    for x in os.getenv(
        "ILLICIT_HINTS",
        (
            "закладк,кладмен,меф,амф,гашиш,героин,кокаин,марихуан,"
            "соль,спайс,доза,куплю наркот,продам наркот,вещества,травка"
        ),
    ).split(",")
    if x.strip()
]

# Админы для ручной разметки.
ADMIN_IDS = {
    int(x.strip())
    for x in os.getenv("ADMIN_IDS", "").split(",")
    if x.strip().isdigit()
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_saved_token() -> str:
    try:
        if os.path.exists(TOKEN_STORE_PATH):
            with open(TOKEN_STORE_PATH, "r", encoding="utf-8") as f:
                return f.read().strip()
    except Exception as exc:
        logging.warning("Cannot read saved token from %s: %s", TOKEN_STORE_PATH, exc)
    return ""


def save_token(token: str) -> None:
    try:
        with open(TOKEN_STORE_PATH, "w", encoding="utf-8") as f:
            f.write(token.strip())
        os.chmod(TOKEN_STORE_PATH, 0o600)
        logging.info("Telegram token saved to %s", TOKEN_STORE_PATH)
    except Exception as exc:
        logging.warning("Cannot save token to %s: %s", TOKEN_STORE_PATH, exc)


def resolve_telegram_token() -> str:
    token = (os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TOKEN") or "").strip()
    if token and AUTO_SAVE_TOKEN:
        save_token(token)
        return token
    if token:
        return token

    token = load_saved_token()
    if token:
        return token

    if os.isatty(0):
        entered = input("Enter TELEGRAM_BOT_TOKEN: ").strip()
        if entered:
            if AUTO_SAVE_TOKEN:
                save_token(entered)
            return entered
    raise RuntimeError("Set TELEGRAM_BOT_TOKEN/TOKEN or save token to TOKEN_STORE_PATH")


def _get_ollama_tags() -> Dict[str, Any]:
    url = f"{OLLAMA_BASE_URL}/api/tags"
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    return response.json()


def ensure_ollama_ready() -> None:
    try:
        _get_ollama_tags()
    except Exception:
        if not AUTO_START_OLLAMA:
            raise RuntimeError("Ollama is not reachable and AUTO_START_OLLAMA=0")
        logging.info("Ollama is not reachable, starting 'ollama serve'...")
        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to start ollama serve: {exc}") from exc

        deadline = time.time() + OLLAMA_WAIT_SECONDS
        last_error = ""
        while time.time() < deadline:
            try:
                _get_ollama_tags()
                break
            except Exception as exc:
                last_error = str(exc)
                time.sleep(1)
        else:
            raise RuntimeError(f"Ollama did not start in time: {last_error}")

    if not AUTO_PULL_OLLAMA_MODELS:
        return

    try:
        tags = _get_ollama_tags()
        names = {str(item.get("name", "")).strip() for item in tags.get("models", [])}
    except Exception as exc:
        logging.warning("Cannot fetch Ollama tags for auto-pull: %s", exc)
        return

    needed = [OLLAMA_MODEL]
    if OLLAMA_VISION_MODEL:
        needed.append(OLLAMA_VISION_MODEL)

    for model in needed:
        if model and model not in names:
            logging.info("Model '%s' not found locally, pulling...", model)
            try:
                subprocess.run(["ollama", "pull", model], check=True)
            except Exception as exc:
                logging.warning("Auto-pull failed for model '%s': %s", model, exc)


class MskFormatter(logging.Formatter):
    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:
        dt = datetime.fromtimestamp(record.created, tz=ZoneInfo("Europe/Moscow"))
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S")


def setup_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(MskFormatter("[%(asctime)s] %(levelname)s %(message)s"))

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    root.addHandler(handler)


def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS moderation_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            chat_id INTEGER NOT NULL,
            user_id INTEGER,
            username TEXT,
            message_id INTEGER,
            content_type TEXT NOT NULL,
            text_excerpt TEXT,
            file_id TEXT,
            flagged INTEGER NOT NULL,
            decision TEXT NOT NULL,
            reason TEXT,
            categories_json TEXT,
            scores_json TEXT,
            reviewer_id INTEGER,
            review_label TEXT,
            review_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bot_status (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def set_bot_status(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        """
        INSERT INTO bot_status (key, value, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
        """,
        (key, value, utc_now_iso()),
    )
    conn.commit()


def excerpt(text: Optional[str], limit: int = 300) -> Optional[str]:
    if not text:
        return None
    cleaned = text.strip().replace("\n", " ")
    return cleaned[:limit]


def is_admin(user_id: Optional[int]) -> bool:
    return bool(user_id) and user_id in ADMIN_IDS


def parse_samples(samples_json: str) -> List[Dict[str, str]]:
    if not samples_json.strip():
        return []
    try:
        raw = json.loads(samples_json)
        if not isinstance(raw, list):
            return []
        out: List[Dict[str, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            label = str(item.get("label", "")).strip()
            if text and label:
                out.append({"text": text, "label": label})
        return out
    except Exception:
        return []


def _extract_json_object(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    if not raw:
        return {}

    # 1) Пробуем как есть.
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # 2) Пробуем вытащить первый JSON-объект из текста.
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return {}
    return {}


def _post_ollama_chat(payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{OLLAMA_BASE_URL}/api/chat"
    response = requests.post(url, json=payload, timeout=60)
    if response.status_code >= 400:
        # Показываем тело ответа Ollama, иначе причину 500 не понять.
        raise RuntimeError(f"Ollama HTTP {response.status_code}: {response.text[:500]}")
    return response.json()


def _has_safety_context(text: str) -> bool:
    lowered = text.lower()
    patterns = [
        "не употребля",
        "не использую наркот",
        "против наркот",
        "вред наркот",
        "борьб",
        "профилактик",
        "осужда",
        "не продаю",
        "не покупаю",
    ]
    return any(p in lowered for p in patterns)


def _has_safe_commerce_context(text: str) -> bool:
    lowered = text.lower()
    return any(p in lowered for p in SAFE_COMMERCE_HINTS)


def _has_illicit_context(text: str) -> bool:
    lowered = text.lower()
    return any(p in lowered for p in ILLICIT_HINTS)


def _risk_level(label: str, confidence: float, text: str) -> str:
    is_blocked_label = label.lower() in BLOCKED_LABELS
    if not is_blocked_label:
        return "low"

    if confidence >= HIGH_THRESHOLD:
        level = "high"
    elif confidence >= MED_THRESHOLD:
        level = "med"
    else:
        level = "low"

    # Контекстный понижающий коэффициент: если сообщение явно про отказ/осуждение.
    if _has_safety_context(text):
        if level == "high" and confidence < 0.92:
            return "med"
        if level == "med":
            return "low"

    # Бизнес-контекст: продажа техники/авто без признаков запрещенного оборота.
    if label.lower() in {"drugs", "illegal"} and _has_safe_commerce_context(text) and not _has_illicit_context(text):
        if level == "high" and confidence < 0.95:
            return "med"
        if level == "med":
            return "low"
    return level


def classify_text_ollama(text: str) -> Dict[str, Any]:
    if not OLLAMA_MODEL:
        raise RuntimeError("Set OLLAMA_MODEL")
    if not LABELS:
        raise RuntimeError("Set LABELS")
    if "allowed" not in [x.lower() for x in LABELS]:
        raise RuntimeError("LABELS must include 'allowed'")

    samples = parse_samples(SAMPLES_JSON)
    samples_block = ""
    if samples:
        lines = []
        for item in samples[:12]:
            lines.append(f'- text: "{item["text"]}" => label: "{item["label"]}"')
        samples_block = "Examples:\n" + "\n".join(lines)

    schema = '{"label":"one_of_labels","confidence":0.0,"reason":"short_reason"}'
    system_prompt = (
        "You are a strict content moderation classifier for Telegram chats.\n"
        f"Allowed labels: {LABELS}\n"
        f"Blocked labels: {sorted(BLOCKED_LABELS)}\n"
        "Classify the user message into exactly one label.\n"
        "Read semantic context, not keywords only. If user condemns illegal content,\n"
        "describes prevention, or says they do NOT use/sell drugs, do not escalate risk.\n"
        "Do not classify normal marketplace messages (car/phone/laptop sales) as drugs/illegal\n"
        "unless there are explicit illegal indicators.\n"
        "Return only JSON with fields: label, confidence, reason.\n"
        "confidence must be number in [0,1].\n"
        "No markdown, no extra text."
    )
    user_prompt = (
        f"{samples_block}\n\n"
        f"JSON schema: {schema}\n"
        f"Message: {text}"
    )

    payload_strict = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {"temperature": 0},
    }
    payload_fallback = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"{user_prompt}\n"
                    'Return only one JSON object, e.g. {"label":"allowed","confidence":0.93,"reason":"safe"}'
                ),
            },
        ],
        "options": {"temperature": 0},
    }

    try:
        data = _post_ollama_chat(payload_strict)
    except Exception as strict_exc:
        logging.warning("Ollama strict-json request failed, fallback mode enabled: %s", strict_exc)
        data = _post_ollama_chat(payload_fallback)

    content = (
        data.get("message", {}).get("content", "")
        if isinstance(data.get("message"), dict)
        else ""
    )
    parsed = _extract_json_object(content)

    top_label = str(parsed.get("label", "")).strip()
    try:
        top_confidence = float(parsed.get("confidence", 0.0) or 0.0)
    except Exception:
        top_confidence = 0.0

    top_confidence = max(0.0, min(1.0, top_confidence))
    if not top_label:
        return {
            "flagged": False,
            "decision": "review",
            "reason": "empty_or_invalid_model_output",
            "categories": {},
            "scores": {},
        }

    top_blocked = top_label.lower() in BLOCKED_LABELS
    risk_level = _risk_level(top_label, top_confidence, text)
    reason = f"{top_label}:{top_confidence:.3f}"

    return {
        "flagged": top_blocked,
        "decision": risk_level,
        "risk_level": risk_level,
        "risk_score": top_confidence,
        "reason": reason,
        "categories": {"top_label": top_label},
        "scores": {top_label: top_confidence},
    }


def classify_image_ollama(image_bytes: bytes, caption_text: str = "") -> Dict[str, Any]:
    if not OLLAMA_VISION_MODEL:
        raise RuntimeError("Set OLLAMA_VISION_MODEL")
    if not LABELS:
        raise RuntimeError("Set LABELS")

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    schema = '{"label":"one_of_labels","confidence":0.0,"reason":"short_reason"}'

    system_prompt = (
        "You are a strict image moderation classifier for Telegram chats.\n"
        f"Allowed labels: {LABELS}\n"
        f"Blocked labels: {sorted(BLOCKED_LABELS)}\n"
        "Analyze visual content. Do not classify by keyword only.\n"
        "Classify into exactly one label and return only JSON.\n"
        "Fields: label, confidence, reason.\n"
        "confidence must be number in [0,1].\n"
        "No markdown, no extra text."
    )
    user_prompt = (
        f"JSON schema: {schema}\n"
        f"Caption context: {caption_text[:300] if caption_text else 'none'}\n"
        "Classify this image."
    )

    payload = {
        "model": OLLAMA_VISION_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt, "images": [b64]},
        ],
        "options": {"temperature": 0},
    }

    data = _post_ollama_chat(payload)
    content = (
        data.get("message", {}).get("content", "")
        if isinstance(data.get("message"), dict)
        else ""
    )
    parsed = _extract_json_object(content)

    top_label = str(parsed.get("label", "")).strip()
    try:
        top_confidence = float(parsed.get("confidence", 0.0) or 0.0)
    except Exception:
        top_confidence = 0.0
    top_confidence = max(0.0, min(1.0, top_confidence))

    if not top_label:
        return {
            "flagged": False,
            "decision": "med",
            "risk_level": "med",
            "risk_score": 0.5,
            "reason": "empty_or_invalid_image_model_output",
            "categories": {},
            "scores": {},
        }

    top_blocked = top_label.lower() in BLOCKED_LABELS
    risk_level = _risk_level(top_label, top_confidence, caption_text)
    reason = f"{top_label}:{top_confidence:.3f}"

    return {
        "flagged": top_blocked,
        "decision": risk_level,
        "risk_level": risk_level,
        "risk_score": top_confidence,
        "reason": reason,
        "categories": {"top_label": top_label, "source": "image_vision_model"},
        "scores": {top_label: top_confidence},
    }


def save_event(
    conn: sqlite3.Connection,
    *,
    chat_id: int,
    user_id: Optional[int],
    username: Optional[str],
    message_id: Optional[int],
    content_type: str,
    text_excerpt_value: Optional[str],
    file_id: Optional[str],
    result: Dict[str, Any],
) -> int:
    cur = conn.execute(
        """
        INSERT INTO moderation_events (
            created_at, chat_id, user_id, username, message_id, content_type,
            text_excerpt, file_id, flagged, decision, reason,
            categories_json, scores_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            utc_now_iso(),
            chat_id,
            user_id,
            username,
            message_id,
            content_type,
            text_excerpt_value,
            file_id,
            int(bool(result.get("flagged", False))),
            str(result.get("decision", "review")),
            str(result.get("reason", "")),
            json.dumps(result.get("categories", {}), ensure_ascii=False),
            json.dumps(result.get("scores", {}), ensure_ascii=False),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def update_event_moderation(conn: sqlite3.Connection, event_id: int, result: Dict[str, Any]) -> None:
    conn.execute(
        """
        UPDATE moderation_events
        SET flagged = ?, decision = ?, reason = ?, categories_json = ?, scores_json = ?
        WHERE id = ?
        """,
        (
            int(bool(result.get("flagged", False))),
            str(result.get("decision", "low")),
            str(result.get("reason", "")),
            json.dumps(result.get("categories", {}), ensure_ascii=False),
            json.dumps(result.get("scores", {}), ensure_ascii=False),
            event_id,
        ),
    )
    conn.commit()


def set_review_label(conn: sqlite3.Connection, event_id: int, reviewer_id: int, label: str) -> None:
    conn.execute(
        """
        UPDATE moderation_events
        SET review_label = ?, reviewer_id = ?, review_at = ?
        WHERE id = ?
        """,
        (label, reviewer_id, utc_now_iso(), event_id),
    )
    conn.commit()


def maybe_send_review_controls(
    bot: telebot.TeleBot,
    chat_id: int,
    event_id: int,
    result: Dict[str, Any],
) -> None:
    if not ADMIN_IDS:
        return

    keyboard = types.InlineKeyboardMarkup()
    keyboard.add(
        types.InlineKeyboardButton("mark_allow", callback_data=f"review:{event_id}:allow"),
        types.InlineKeyboardButton("mark_block", callback_data=f"review:{event_id}:block"),
    )

    bot.send_message(
        chat_id,
        (
            f"[review] event_id={event_id} decision={result.get('decision')} "
            f"reason={result.get('reason')}\n"
            "Выберите label для датасета"
        ),
        reply_markup=keyboard,
    )


def log_detection(
    *,
    chat_id: int,
    message_id: Optional[int],
    user_id: Optional[int],
    username: Optional[str],
    content_type: str,
    text_excerpt_value: Optional[str],
    result: Dict[str, Any],
) -> None:
    logging.warning(
        (
            "MODERATION_MATCH chat_id=%s message_id=%s user_id=%s username=%s "
            "type=%s risk=%s risk_score=%s reason=%s categories=%s scores=%s text_excerpt=%s"
        ),
        chat_id,
        message_id,
        user_id,
        username,
        content_type,
        result.get("risk_level", result.get("decision")),
        result.get("risk_score"),
        result.get("reason"),
        json.dumps(result.get("categories", {}), ensure_ascii=False),
        json.dumps(result.get("scores", {}), ensure_ascii=False),
        text_excerpt_value,
    )


def main() -> None:
    setup_logging()

    ensure_ollama_ready()
    telegram_token = resolve_telegram_token()

    conn = init_db()
    bot = telebot.TeleBot(telegram_token)
    set_bot_status(conn, "batch_interval_seconds", str(RECHECK_INTERVAL_SECONDS if RECHECK_ENABLED else 0))
    set_bot_status(conn, "next_run_epoch", "0")
    set_bot_status(conn, "queue_size", "0")

    def process_message(message: telebot.types.Message) -> None:
        content_type = message.content_type
        chat_id = message.chat.id
        message_id = message.message_id
        user_id = message.from_user.id if message.from_user else None
        username = message.from_user.username if message.from_user else None
        text = (message.text or "").strip()
        caption = (message.caption or "").strip()
        file_id = None

        if content_type == "text":
            if not text:
                return
            result = classify_text_ollama(text)
            text_excerpt_value = excerpt(text)
        elif content_type == "photo":
            if not message.photo:
                return
            file_id = message.photo[-1].file_id
            try:
                file_info = bot.get_file(file_id)
                image_bytes = bot.download_file(file_info.file_path)
                result = classify_image_ollama(image_bytes, caption)
            except Exception as exc:
                logging.exception("Image moderation failed: %s", exc)
                result = {
                    "flagged": False,
                    "decision": "med",
                    "risk_level": "med",
                    "risk_score": 0.5,
                    "reason": "image_model_error_manual_review_required",
                    "categories": {"source": "ollama_vision_fallback"},
                    "scores": {},
                }
            text_excerpt_value = excerpt(caption or "")
        elif content_type in {"video", "video_note"}:
            if content_type == "video" and message.video:
                file_id = message.video.file_id
            elif content_type == "video_note" and message.video_note:
                file_id = message.video_note.file_id
            result = {
                "flagged": False,
                "decision": "med",
                "risk_level": "med",
                "risk_score": 0.5,
                "reason": "manual_media_review_required",
                "categories": {"source": "video_manual_review"},
                "scores": {},
            }
            text_excerpt_value = excerpt(caption or "")
        else:
            return

        event_id = save_event(
            conn,
            chat_id=chat_id,
            user_id=user_id,
            username=username,
            message_id=message_id,
            content_type=content_type,
            text_excerpt_value=text_excerpt_value,
            file_id=file_id,
            result=result,
        )

        risk = result.get("risk_level", "low")
        if risk in {"med", "high"}:
            log_detection(
                chat_id=chat_id,
                message_id=message_id,
                user_id=user_id,
                username=username,
                content_type=content_type,
                text_excerpt_value=text_excerpt_value,
                result=result,
            )
            maybe_send_review_controls(bot, chat_id, event_id, result)
        else:
            logging.info(
                "MODERATION_LOW chat_id=%s message_id=%s user_id=%s risk=%s reason=%s",
                chat_id,
                message_id,
                user_id,
                risk,
                result.get("reason"),
            )

    def periodic_recheck_worker() -> None:
        if not RECHECK_ENABLED:
            return
        while True:
            next_run = int(datetime.now(timezone.utc).timestamp()) + RECHECK_INTERVAL_SECONDS
            set_bot_status(conn, "next_run_epoch", str(next_run))
            time.sleep(RECHECK_INTERVAL_SECONDS)

            cutoff_iso = (datetime.now(timezone.utc) - timedelta(days=RECHECK_LOOKBACK_DAYS)).isoformat()
            rows = conn.execute(
                """
                SELECT id, chat_id, message_id, user_id, username, content_type, text_excerpt, file_id
                FROM moderation_events
                WHERE created_at >= ?
                ORDER BY id DESC
                """,
                (cutoff_iso,),
            ).fetchall()

            set_bot_status(conn, "queue_size", str(len(rows)))
            logging.info(
                "Periodic recheck started: interval=%ss lookback_days=%s events=%s",
                RECHECK_INTERVAL_SECONDS,
                RECHECK_LOOKBACK_DAYS,
                len(rows),
            )

            for row in rows:
                event_id = int(row[0])
                chat_id = int(row[1]) if row[1] is not None else None
                message_id = row[2]
                user_id = row[3]
                username = row[4]
                content_type = row[5]
                text_excerpt_value = row[6] or ""
                file_id = row[7]

                try:
                    if content_type == "text":
                        if not text_excerpt_value:
                            continue
                        result = classify_text_ollama(text_excerpt_value)
                    elif content_type == "photo":
                        if not file_id:
                            continue
                        file_info = bot.get_file(file_id)
                        image_bytes = bot.download_file(file_info.file_path)
                        result = classify_image_ollama(image_bytes, text_excerpt_value)
                    elif content_type in {"video", "video_note"}:
                        result = {
                            "flagged": False,
                            "decision": "med",
                            "risk_level": "med",
                            "risk_score": 0.5,
                            "reason": "periodic_recheck_manual_video_review_required",
                            "categories": {"source": "video_manual_review"},
                            "scores": {},
                        }
                    else:
                        continue

                    update_event_moderation(conn, event_id, result)
                    risk = result.get("risk_level", result.get("decision", "low"))
                    if risk in {"med", "high"} and chat_id is not None:
                        log_detection(
                            chat_id=chat_id,
                            message_id=message_id,
                            user_id=user_id,
                            username=username,
                            content_type=content_type,
                            text_excerpt_value=excerpt(text_excerpt_value, 200),
                            result=result,
                        )
                except Exception as exc:
                    logging.exception("Periodic recheck failed for event_id=%s: %s", event_id, exc)

            set_bot_status(conn, "queue_size", "0")

    recheck_thread = threading.Thread(
        target=periodic_recheck_worker,
        name="periodic-recheck-worker",
        daemon=True,
    )
    recheck_thread.start()

    @bot.message_handler(commands=["start", "help"])
    def on_start(message: telebot.types.Message) -> None:
        bot.reply_to(
            message,
            "Модератор активен (Ollama, text classifier).\n"
            "Удаление отключено: бот только логирует и ставит риск low/med/high.\n"
            "Проверка каждого нового сообщения включена.\n"
            f"Периодический recheck: {'on' if RECHECK_ENABLED else 'off'} "
            f"(каждые {RECHECK_INTERVAL_SECONDS}с, глубина {RECHECK_LOOKBACK_DAYS} дн.).\n"
            "Фото/видео сейчас идут в review.\n"
            "Команда для админов: /export_dataset",
        )

    @bot.message_handler(commands=["check_ollama"])
    def on_check(message: telebot.types.Message) -> None:
        try:
            result = classify_text_ollama("проверка соединения")
            bot.reply_to(
                message,
                (
                    f"OK\nurl={OLLAMA_BASE_URL}\nmodel={OLLAMA_MODEL}\n"
                    f"risk={result.get('risk_level')} score={result.get('risk_score')} "
                    f"reason={result.get('reason')}"
                ),
            )
        except Exception as exc:
            bot.reply_to(
                message,
                f"Ollama API error\nurl={OLLAMA_BASE_URL}\nmodel={OLLAMA_MODEL}\n{exc}",
            )

    @bot.message_handler(commands=["export_dataset"])
    def on_export_dataset(message: telebot.types.Message) -> None:
        if not is_admin(message.from_user.id if message.from_user else None):
            bot.reply_to(message, "Недостаточно прав")
            return

        output = os.path.abspath("training_feedback.jsonl")
        cur = conn.execute(
            """
            SELECT id, content_type, text_excerpt, categories_json, scores_json, review_label
            FROM moderation_events
            WHERE review_label IS NOT NULL
            ORDER BY id ASC
            """
        )
        rows = cur.fetchall()

        with open(output, "w", encoding="utf-8") as f:
            for row in rows:
                payload = {
                    "event_id": row[0],
                    "content_type": row[1],
                    "text": row[2],
                    "categories": json.loads(row[3]) if row[3] else {},
                    "scores": json.loads(row[4]) if row[4] else {},
                    "label": row[5],
                }
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        bot.reply_to(message, f"Экспортировано {len(rows)} кейсов в {output}")

    @bot.callback_query_handler(func=lambda call: call.data.startswith("review:"))
    def on_review(call: telebot.types.CallbackQuery) -> None:
        if not call.from_user or not is_admin(call.from_user.id):
            bot.answer_callback_query(call.id, "Нет прав")
            return

        try:
            _, event_id_raw, label = call.data.split(":", 2)
            event_id = int(event_id_raw)
            if label not in {"allow", "block"}:
                raise ValueError("invalid label")
        except Exception:
            bot.answer_callback_query(call.id, "Некорректные данные")
            return

        set_review_label(conn, event_id, call.from_user.id, label)
        bot.answer_callback_query(call.id, f"Сохранено: {label}")

    @bot.message_handler(content_types=["text"])
    def on_text(message: telebot.types.Message) -> None:
        if not (message.text or "").strip():
            return
        try:
            process_message(message)
        except Exception as exc:
            logging.exception("Text moderation failed: %s", exc)

    @bot.message_handler(content_types=["photo", "video", "video_note"])
    def on_media(message: telebot.types.Message) -> None:
        try:
            process_message(message)
        except Exception as exc:
            logging.exception("Media moderation failed: %s", exc)

    logging.info(
        "Bot started. ollama_url=%s model=%s labels=%s blocked=%s block=%.2f review=%.2f",
        OLLAMA_BASE_URL,
        OLLAMA_MODEL,
        LABELS,
        sorted(BLOCKED_LABELS),
        BLOCK_THRESHOLD,
        REVIEW_THRESHOLD,
    )

    while True:
        try:
            bot.polling(non_stop=True, skip_pending=True, timeout=20)
        except Exception as exc:
            logging.exception("Polling crash: %s", exc)
            time.sleep(3)


if __name__ == "__main__":
    main()
