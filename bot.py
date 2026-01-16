import os
import asyncio
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from dotenv import load_dotenv
import chroma_remove

import cv2
import numpy as np
import io

# Load environment variables
load_dotenv()

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


async def process_and_reply(update: Update, attachment):
    """
    Downloads file to memory, processes it, and sends the result back.
    Shared logic for both Photo and Document attributes.
    """
    # Send processing status
    status_msg = await update.message.reply_text("⏳ Processing...")
    
    try:
        # 10MB limit check
        if attachment.file_size and attachment.file_size > 10 * 1024 * 1024:
            await status_msg.edit_text("❌ File is too big. Please send an image smaller than 10MB.")
            return

        # Get the actual file object (this effectively 'prepares' the download link)
        # We do this here so it doesn't block the main handler
        try:
            telegram_file = await attachment.get_file()
        except Exception as e:
            logging.error(f"Failed to get file info: {e}")
            await status_msg.edit_text("❌ Could not retrieve file information from Telegram.")
            return

        # Download file to memory
        with io.BytesIO() as f_in:
            await telegram_file.download_to_memory(out=f_in)
            f_in.seek(0)
            file_bytes = f_in.read()

        # Run CPU-bound processing in a separate thread
        loop = asyncio.get_running_loop()
        try:
            encoded_img = await loop.run_in_executor(None, process_image_sync, file_bytes)
        except ValueError as e:
            await status_msg.edit_text(str(e))
            return

        # Delete status message before sending result
        await status_msg.delete()

        # Send result back from memory
        with io.BytesIO(encoded_img.tobytes()) as f_out:
            f_out.name = "processed.png"
            await update.message.reply_document(document=f_out, filename="processed.png")
        
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        try:
            await status_msg.edit_text("❌ An internal error occurred while processing the image.")
        except Exception:
             logging.error("Could not edit status message to report error.")

def process_image_sync(file_bytes):
    """
    Synchronous function to handle CPU-bound image processing.
    Decodes, processes, and encodes the image.
    """
    # Convert bytes to numpy array
    nparr = np.frombuffer(file_bytes, np.uint8)
    
    # Decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
         raise ValueError("❌ The file you sent doesn't look like a valid image.")

    # Process image
    try:
        result_img = chroma_remove.process_image(img)
    except ValueError as e:
        raise ValueError(f"❌ Processing failed: {e}")

    # Encode result to PNG
    success, encoded_img = cv2.imencode('.png', result_img)
    if not success:
        raise ValueError("Could not encode result image")
        
    return encoded_img

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.photo:
        return

    user = update.effective_user
    logging.info(f"Received photo from {user.first_name} (id: {user.id})")

    # Get the largest photo.
    # We pass the PhotoSize object itself to the background task.
    # AND we use create_task to not block the receiver loop
    photo_size = update.message.photo[-1]
    asyncio.create_task(process_and_reply(update, photo_size))

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.document:
        return

    user = update.effective_user
    doc = update.message.document
    
    # MIME type check (basic filter)
    if not doc.mime_type or not doc.mime_type.startswith('image/'):
        return

    logging.info(f"Received document from {user.first_name} (id: {user.id}), size: {doc.file_size}")

    # Pass the Document object itself to the background task
    asyncio.create_task(process_and_reply(update, doc))

if __name__ == '__main__':
    TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    if not TOKEN:
        print("Error: TELEGRAM_BOT_TOKEN not found in environment variables.")
        exit(1)

    # Increase connection pool timeouts to handle multiple concurrent uploads better
    application = (
        ApplicationBuilder()
        .token(TOKEN)
        .connect_timeout(30.0)
        .read_timeout(30.0)
        .write_timeout(30.0)
        .build()
    )
    
    photo_handler = MessageHandler(filters.PHOTO, handle_photo)
    document_handler = MessageHandler(filters.Document.ALL, handle_document)
    
    application.add_handler(photo_handler)
    application.add_handler(document_handler)
    
    print("Bot is running...")
    application.run_polling()
