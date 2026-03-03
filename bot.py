"""
Gemini Watermark Remover Bot
Send any Gemini-generated image → get back a clean, watermark-free version.
"""

import logging
import os
import tempfile
from pathlib import Path

from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from gemini_watermark_remover import WatermarkRemover

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

BOT_TOKEN = os.environ["BOT_TOKEN"]
ALLOWED_CHAT_ID = int(os.environ["CHAT_ID"])  # Only respond to you

remover = WatermarkRemover()


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id

    # Ignore anyone who isn't you
    if chat_id != ALLOWED_CHAT_ID:
        await update.message.reply_text("⛔ Unauthorized.")
        return

    # Get the highest-resolution photo or document
    if update.message.photo:
        file = await update.message.photo[-1].get_file()
        ext = ".jpg"
    elif update.message.document and update.message.document.mime_type.startswith("image/"):
        file = await update.message.document.get_file()
        ext = Path(update.message.document.file_name).suffix or ".jpg"
    else:
        await update.message.reply_text("📸 Send me a Gemini image and I'll remove the watermark!")
        return

    status = await update.message.reply_text("⏳ Processing...")

    with tempfile.TemporaryDirectory() as tmp:
        input_path = Path(tmp) / f"input{ext}"
        output_path = Path(tmp) / f"clean{ext}"

        await file.download_to_drive(input_path)

        try:
            import cv2
            image = cv2.imread(str(input_path))
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
            log.error(f"Processing failed: {e}")
            await status.edit_text(f"❌ Failed to process image: {e}")


async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id != ALLOWED_CHAT_ID:
        return
    await update.message.reply_text(
        "👋 Ready! Send me any Gemini-generated image and I'll remove the watermark instantly."
    )


def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_image))
    app.add_handler(MessageHandler(filters.TEXT & filters.Regex(r"^/start"), handle_start))
    log.info("Bot is running...")
    app.run_polling()


if __name__ == "__main__":
    main()
