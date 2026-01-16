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

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.photo:
        return

    user = update.effective_user
    logging.info(f"Received photo from {user.first_name} (id: {user.id})")

    # Get the file ID of the largest photo
    photo_file = await update.message.photo[-1].get_file()
    
    try:
        # Download file to memory
        with io.BytesIO() as f_in:
            await photo_file.download_to_memory(out=f_in)
            f_in.seek(0)
            file_bytes = np.asarray(bytearray(f_in.read()), dtype=np.uint8)
            
            # Decode image
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None:
                 raise ValueError("Could not decode image")

            # Process image (in-memory)
            result_img = chroma_remove.process_image(img)
            
            # Encode result to PNG
            success, encoded_img = cv2.imencode('.png', result_img)
            if not success:
                raise ValueError("Could not encode result image")
            
            # Send result back from memory
            with io.BytesIO(encoded_img.tobytes()) as f_out:
                f_out.name = "processed.png"
                await update.message.reply_document(document=f_out, filename="processed.png")
        
    except ValueError as val_err:
        await update.message.reply_text(f"Processing failed: {val_err}")
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        await update.message.reply_text("An internal error occurred while processing the image.")

if __name__ == '__main__':
    TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    if not TOKEN:
        print("Error: TELEGRAM_BOT_TOKEN not found in environment variables.")
        exit(1)

    application = ApplicationBuilder().token(TOKEN).build()
    
    photo_handler = MessageHandler(filters.PHOTO, handle_photo)
    application.add_handler(photo_handler)
    
    print("Bot is running...")
    application.run_polling()
