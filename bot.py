import os
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

async def process_and_reply(update: Update, telegram_file):
    """
    Downloads file to memory, processes it, and sends the result back.
    Shared logic for both Photo and Document attributes.
    """
    try:
        # 10MB limit check (file_size is in bytes)
        # telegram_file object usually has file_size attribute, but for PhotoSize it definitely does.
        # For a File object obtained via get_file(), file_size is also available.
        if telegram_file.file_size and telegram_file.file_size > 10 * 1024 * 1024:
            await update.message.reply_text("File is too big. Please send an image smaller than 10MB.")
            return

        # Download file to memory
        with io.BytesIO() as f_in:
            await telegram_file.download_to_memory(out=f_in)
            f_in.seek(0)
            file_bytes = np.asarray(bytearray(f_in.read()), dtype=np.uint8)
            
            # Decode image - this acts as a robust prefix/magic byte check
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None:
                 await update.message.reply_text("The file you sent doesn't look like a valid image.")
                 return

            # Process image (in-memory)
            try:
                result_img = chroma_remove.process_image(img)
            except ValueError as e:
                await update.message.reply_text(f"Processing failed: {e}")
                return

            # Encode result to PNG
            success, encoded_img = cv2.imencode('.png', result_img)
            if not success:
                raise ValueError("Could not encode result image")
            
            # Send result back from memory
            with io.BytesIO(encoded_img.tobytes()) as f_out:
                f_out.name = "processed.png"
                await update.message.reply_document(document=f_out, filename="processed.png")
        
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        await update.message.reply_text("An internal error occurred while processing the image.")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.photo:
        return

    user = update.effective_user
    logging.info(f"Received photo from {user.first_name} (id: {user.id})")

    # Get the file ID of the largest photo
    photo_file = await update.message.photo[-1].get_file()
    await process_and_reply(update, photo_file)

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.document:
        return

    user = update.effective_user
    doc = update.message.document
    
    # MIME type check (basic filter)
    if not doc.mime_type or not doc.mime_type.startswith('image/'):
        # Optionally silently ignore non-images or tell the user
        # await update.message.reply_text("Please send an image file.")
        return

    logging.info(f"Received document from {user.first_name} (id: {user.id}), size: {doc.file_size}")

    # For documents, we must get the file object to check size accurately via API if needed, 
    # but the Document object itself usually has file_size.
    if doc.file_size and doc.file_size > 10 * 1024 * 1024:
         await update.message.reply_text("File is too big. Please send an image smaller than 10MB.")
         return

    # Get the actual file object for downloading
    doc_file = await doc.get_file()
    await process_and_reply(update, doc_file)

if __name__ == '__main__':
    TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    if not TOKEN:
        print("Error: TELEGRAM_BOT_TOKEN not found in environment variables.")
        exit(1)

    application = ApplicationBuilder().token(TOKEN).build()
    
    photo_handler = MessageHandler(filters.PHOTO, handle_photo)
    document_handler = MessageHandler(filters.Document.ALL, handle_document)
    
    application.add_handler(photo_handler)
    application.add_handler(document_handler)
    
    print("Bot is running...")
    application.run_polling()
