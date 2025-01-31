import os
import pytesseract
import requests
from openai import OpenAI
from PIL import Image
from io import BytesIO
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse

# Twilio API Credentials (Replace with your own)
TWILIO_ACCOUNT_SID = "##"
TWILIO_AUTH_TOKEN = "##"
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="###",
)

app = Flask(__name__)

@app.route("/", methods=["POST"])
def webhook():
    incoming_msg = request.form.get("Body")  # Extract text message
    sender = request.form.get("From")  # Extract sender's number
    media_url = request.form.get("MediaUrl0")  # Get the first media URL if available

    print(f"Received message: {incoming_msg} from {sender}")
    response = MessagingResponse()

    if media_url:  # If an image is attached
        print(f"Image received: {media_url}")
        try:
            extracted_text = extract_text_from_image(media_url)
            summarized_text = summarize_text_with_gpt(extracted_text)
            response.message(f"Extracted text: {extracted_text}")
        except Exception as e:
            response.message("Failed to process image. Try again.")
            print(f"Error: {e}")
    else:
        response.message("Please send an image to extract text.")

    return str(response)

def extract_text_from_image(image_url):
    """ Download the image using Twilio authentication, run OCR, and return extracted text. """
    
    # Authenticate request to download Twilio media
    response = requests.get(image_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
    
    if response.status_code != 200:
        raise Exception("Failed to download image from Twilio.")

    # Read image into memory
    image = Image.open(BytesIO(response.content))

    # Run OCR
    extracted_text = pytesseract.image_to_string(image)

    return extracted_text.strip()
def summarize_text_with_gpt(text):
    """ Summarize the extracted text using OpenAI GPT. """
    # openai.api_key = OPENAI_API_KEY

    prompt = f"Summarize this text in simple language:\n\n{text}"

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a helpful AI assistant that summarizes text."},
                  {"role": "user", "content": prompt}],
         temperature=0.7,
    )

    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    app.run(debug=True, port=5002)
