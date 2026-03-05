"""
Gemini Watermark Remover + Meta AI Video Downloader Bot
- Send a Gemini image → get back watermark-free version
- Send a meta.ai/media-share URL → get back the downloaded video
"""

import logging
import os
import re
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BOT_TOKEN       = os.environ["BOT_TOKEN"]
ALLOWED_CHAT_ID = int(os.environ["CHAT_ID"])

META_AI_URL_PATTERN = re.compile(r"https?://(?:www\.)?meta\.ai/media-share/\S+")

# ─── TUNABLE OFFSET VARIABLES ─────────────────────────────────────────────────
# If the removal patch is in the wrong place, adjust these:
#   OFFSET_X  — shift patch left (negative) or right (positive)
#   OFFSET_Y  — shift patch up (negative) or down (positive)
#
# Gemini places watermark at bottom-right of the CONTENT area (ignoring black padding)
# Actual measured margins from content edge: ~16px right, ~15px bottom

MARGIN_RIGHT  = 16   # pixels from right edge of content to right edge of watermark
MARGIN_BOTTOM = 15   # pixels from bottom edge of content to bottom edge of watermark
OFFSET_X      = 44    # fine-tune horizontal position
OFFSET_Y      = 44    # fine-tune vertical position

# ─── ALPHA MAP LOADING ────────────────────────────────────────────────────────

ALPHA_MAP_URLS = {
    48: "https://raw.githubusercontent.com/journey-ad/gemini-watermark-remover/main/src/assets/bg_48.png",
    96: "https://raw.githubusercontent.com/journey-ad/gemini-watermark-remover/main/src/assets/bg_96.png",
}

def load_alpha_map(size: int) -> np.ndarray:
    url = ALPHA_MAP_URLS[size]
    log.info(f"Downloading alpha map {size}px...")
    with urllib.request.urlopen(url) as r:
        data = r.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to decode alpha map {size}px")
    return np.max(img.astype(np.float32), axis=2) / 255.0

log.info("Loading alpha maps...")
ALPHA_48 = load_alpha_map(48)
ALPHA_96 = load_alpha_map(96)
log.info("Alpha maps ready.")

# ─── WATERMARK REMOVAL ────────────────────────────────────────────────────────

def get_content_bottom(image: np.ndarray) -> int:
    """Find last non-black row to handle Gemini's bottom padding."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h    = image.shape[0]
    for y in range(h - 1, 0, -1):
        if gray[y].max() > 10:
            return y
    return h - 1


def remove_gemini_watermark(image: np.ndarray) -> np.ndarray:
    h, w           = image.shape[:2]
    content_bottom = get_content_bottom(image)
    content_h      = content_bottom  # approximate content height

    # Choose watermark size: 96px when both dims > 1024, else 48px
    if w > 1024 and content_h > 1024:
        size = 96
    else:
        size = 48

    alpha_map = ALPHA_48 if size == 48 else ALPHA_96

    # Watermark position with tunable offsets
    x   = w - MARGIN_RIGHT - size + OFFSET_X
    y_wm = content_bottom - MARGIN_BOTTOM - size + OFFSET_Y

    log.info(f"Image {w}x{h}, content_bottom={content_bottom}, {size}px watermark at ({x}, {y_wm})")

    ALPHA_THRESHOLD = 0.002
    MAX_ALPHA       = 0.99
    LOGO_VALUE      = 255.0

    result = image.astype(np.float32).copy()
    a      = np.clip(alpha_map, 0, MAX_ALPHA)
    mask   = a >= ALPHA_THRESHOLD

    for c in range(3):
        patch  = result[y_wm:y_wm+size, x:x+size, c]
        orig   = np.where(mask, (patch - a * LOGO_VALUE) / (1.0 - a), patch)
        result[y_wm:y_wm+size, x:x+size, c] = np.clip(orig, 0, 255)

    return result.astype(np.uint8)


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def is_authorized(update: Update) -> bool:
    return update.effective_chat.id == ALLOWED_CHAT_ID


# ─── GEMINI IMAGE HANDLER ─────────────────────────────────────────────────────

async def handle_image(update: Update, context):
    if not is_authorized(update):
        return

    if update.message.photo:
        file = await update.message.photo[-1].get_file()
        ext  = ".jpg"
    elif update.message.document and update.message.document.mime_type.startswith("image/"):
        file = await update.message.document.get_file()
        ext  = Path(update.message.document.file_name).suffix or ".jpg"
    else:
        return

    status = await update.message.reply_text("⏳ Removing watermark...")

    with tempfile.TemporaryDirectory() as tmp:
        input_path  = Path(tmp) / f"input{ext}"
        output_path = Path(tmp) / f"clean{ext}"

        await file.download_to_drive(input_path)

        try:
            image   = cv2.imread(str(input_path))
            cleaned = remove_gemini_watermark(image)
            cv2.imwrite(str(output_path), cleaned, [cv2.IMWRITE_JPEG_QUALITY, 100])

            await status.delete()
            with open(output_path, "rb") as f:
                await update.message.reply_document(
                    document=f,
                    filename=f"clean{ext}",
                    caption="✅ Watermark removed!",
                )
        except Exception as e:
            log.error(f"Image processing failed: {e}")
            await status.edit_text(f"❌ Failed to process image: {e}")


# ─── META AI URL HANDLER ──────────────────────────────────────────────────────

async def handle_text(update: Update, context):
    if not is_authorized(update):
        return

    text  = update.message.text.strip()
    match = META_AI_URL_PATTERN.search(text)

    if not match:
        await update.message.reply_text(
            "👋 Send me:\n"
            "• A Gemini image → I'll remove the watermark\n"
            "• A meta.ai/media-share URL → I'll download the video"
        )
        return

    url    = match.group()
    status = await update.message.reply_text("⬇️ Downloading video...")

    with tempfile.TemporaryDirectory() as tmp:
        output_template = str(Path(tmp) / "video.%(ext)s")

        try:
            result = subprocess.run(
                [
                    "yt-dlp", "--no-playlist",
                    "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                    "--merge-output-format", "mp4",
                    "-o", output_template, url,
                ],
                capture_output=True, text=True, timeout=120,
            )

            if result.returncode != 0:
                await status.edit_text(f"❌ Download failed:\n{result.stderr[-500:]}")
                return

            videos = list(Path(tmp).glob("video.*"))
            if not videos:
                await status.edit_text("❌ No video file found after download.")
                return

            video_path   = videos[0]
            file_size_mb = video_path.stat().st_size / (1024 * 1024)
            await status.delete()

            if file_size_mb > 50:
                await update.message.reply_text(
                    f"⚠️ Video is {file_size_mb:.1f}MB — too large for Telegram (50MB limit)."
                )
                return

            with open(video_path, "rb") as f:
                await update.message.reply_document(
                    document=f, filename=video_path.name,
                    caption=f"✅ Downloaded! ({file_size_mb:.1f}MB)",
                )

        except subprocess.TimeoutExpired:
            await status.edit_text("❌ Download timed out (>2 min). Try again.")
        except Exception as e:
            await status.edit_text(f"❌ Error: {e}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_image))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    log.info("Bot is running...")
    app.run_polling()


if __name__ == "__main__":
    main()
