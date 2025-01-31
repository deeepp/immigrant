import os
import pytesseract
import requests
from openai import OpenAI
from PIL import Image
from io import BytesIO
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse

# Twilio setup
account_sid = ''  # Replace with your actual Account SID
auth_token = ''    # Replace with your actual Auth Token
twilio_number = 'whatsapp:+14155238886'  # Twilio sandbox number or your WhatsApp business number

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=""
)


app = Flask(__name__)

# Dictionary to store recent summaries for each user (to allow translation)
user_summaries = {}

@app.route("/webhook", methods=["POST"])
def webhook():
    incoming_msg = request.form.get("Body", "").strip().lower()  # Extract text message
    sender = request.form.get("From")  # Extract sender's number
    media_url = request.form.get("MediaUrl0")  # Get the first media URL if available

    print(f"Received message: {incoming_msg} from {sender}")
    response = MessagingResponse()

    if media_url:  # If an image is attached
        print(f"Image received: {media_url}")
        try:
            extracted_text = extract_text_from_image(media_url)
            summarized_text = summarize_text_with_gpt(extracted_text)

            # Store the summary in user session for later translation
            user_summaries[sender] = summarized_text

            response.message(f"Summary: {summarized_text}\n\nWould you like this translated into Spanish? Reply with 'Yes' or 'Translate to Spanish'.")
        except Exception as e:
            response.message("Failed to process image. Try again.")
            print(f"Error: {e}")

    elif incoming_msg in ["yes", "translate to spanish", "spanish", "sí", "translate"]:
        # Fetch the last summary for this user
        if sender in user_summaries:
            translated_text = translate_text_with_gpt(user_summaries[sender], "Spanish")
            response.message(f"Translated to Spanish: {translated_text}")
        else:
            response.message("No summary found to translate. Please send an image first.")

    else:
        response.message("Please send an image to extract text.")

    return str(response)

def extract_text_from_image(image_url):
    """ Download the image using Twilio authentication, run OCR, and return extracted text. """
    
    # Authenticate request to download Twilio media
    response = requests.get(image_url, auth=(account_sid, auth_token))
    
    if response.status_code != 200:
        raise Exception("Failed to download image from Twilio.")

    # Read image into memory
    image = Image.open(BytesIO(response.content))

    # Run OCR
    extracted_text = pytesseract.image_to_string(image)

    return extracted_text.strip()

def summarize_text_with_gpt(text):
    """ Summarize the extracted text using OpenAI GPT. """
    prompt = f"Summarize this text in simple language:\n\n{text}"

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a helpful AI assistant that summarizes text."},
                  {"role": "user", "content": prompt}],
         temperature=0.7,
    )

    return response.choices[0].message.content.strip()

def translate_text_with_gpt(text, target_language):
    """ Translate the given text into the specified language using OpenAI GPT. """
    prompt = f"Translate the following text into {target_language}:\n\n{text}"

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a helpful AI assistant that translates text."},
                  {"role": "user", "content": prompt}],
         temperature=0.7,
    )

    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    app.run(debug=True, port=5001)