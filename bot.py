import os
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from dotenv import load_dotenv
import chroma_remove

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
    
    # Download file
    input_filename = f"temp_{user.id}_in.jpg"
    output_filename = f"temp_{user.id}_out.png"
    
    try:
        await photo_file.download_to_drive(input_filename)
        
        # Process image
        chroma_remove.remove_chroma(input_filename, output_filename)
        
        # Send result back
        await update.message.reply_document(document=open(output_filename, 'rb'), filename="processed.png")
        
    except ValueError as val_err:
        await update.message.reply_text(f"Processing failed: {val_err}")
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        await update.message.reply_text("An internal error occurred while processing the image.")
    finally:
        # Cleanup
        if os.path.exists(input_filename):
            os.remove(input_filename)
        if os.path.exists(output_filename):
            os.remove(output_filename)

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
