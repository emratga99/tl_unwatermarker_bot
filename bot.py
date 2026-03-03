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
from pathlib import Path

import cv2
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from gemini_watermark_remover import WatermarkRemover

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

BOT_TOKEN        = os.environ["BOT_TOKEN"]
ALLOWED_CHAT_ID  = int(os.environ["CHAT_ID"])

remover = WatermarkRemover()

META_AI_URL_PATTERN = re.compile(r"https?://(?:www\.)?meta\.ai/media-share/\S+")


def is_authorized(update: Update) -> bool:
    return update.effective_chat.id == ALLOWED_CHAT_ID


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return

    if update.message.photo:
        file = await update.message.photo[-1].get_file()
        ext = ".jpg"
    elif update.message.document and update.message.document.mime_type.startswith("image/"):
        file = await update.message.document.get_file()
        ext = Path(update.message.document.file_name).suffix or ".jpg"
    else:
        return

    status = await update.message.reply_text("⏳ Removing watermark...")

    with tempfile.TemporaryDirectory() as tmp:
        input_path  = Path(tmp) / f"input{ext}"
        output_path = Path(tmp) / f"clean{ext}"

        await file.download_to_drive(input_path)

        try:
            image   = cv2.imread(str(input_path))
            cleaned = remover.remove_watermark(image)
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


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return

    text = update.message.text.strip()
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
                    "yt-dlp",
                    "--no-playlist",
                    "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                    "--merge-output-format", "mp4",
                    "-o", output_template,
                    url,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                log.error(f"yt-dlp error: {result.stderr}")
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
                    f"⚠️ Video is {file_size_mb:.1f}MB — too large for Telegram (50MB limit).\n"
                    "Try a shorter clip."
                )
                return

            with open(video_path, "rb") as f:
                await update.message.reply_document(
                    document=f,
                    filename=video_path.name,
                    caption=f"✅ Downloaded! ({file_size_mb:.1f}MB)",
                )

        except subprocess.TimeoutExpired:
            await status.edit_text("❌ Download timed out (>2 min). Try again.")
        except Exception as e:
            log.error(f"Video download failed: {e}")
            await status.edit_text(f"❌ Error: {e}")


def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_image))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    log.info("Bot is running...")
    app.run_polling()


if __name__ == "__main__":
    main()
