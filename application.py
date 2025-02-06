import os
import pytesseract
import requests
import time
from openai import OpenAI
from PIL import Image
from io import BytesIO
from flask import Flask, request, Response
from twilio.twiml.messaging_response import MessagingResponse
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from threading import Timer

# Initialize Flask app
app = Flask(__name__)

# Twilio & OpenAI API Credentials
TWILIO_ACCOUNT_SID = 'AC749a40e3778fb3b90acf616960eed8ae'  # Replace with your actual Account SID
TWILIO_AUTH_TOKEN = '4c10cc9439adcd9119b38e0f868efea7'    # Replace with your actual Auth Token
twilio_number = 'whatsapp:+14155238886'  # Twilio sandbox number or your WhatsApp business number

client = OpenAI(api_key="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")

# Initialize Presidio for text anonymization
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# Store user sessions & batch images
user_sessions = {}
image_batches = {}

# Image batch timeout (seconds)
IMAGE_BATCH_TIMEOUT = 5

def download_image(image_url):
    """Downloads an image from Twilio and processes it."""
    try:
        print(f"📷 Downloading image from: {image_url}")
        response = requests.get(image_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))  # ✅ FIXED AUTH METHOD

        if response.status_code != 200:
            print(f"❌ Failed to download image. Status code: {response.status_code}")
            return None

        # Open image
        image = Image.open(BytesIO(response.content))

        # Convert to grayscale for better OCR accuracy
        image = image.convert("L")

        return image

    except Exception as e:
        print(f"❌ Image Processing Error: {e}")
        return None

def extract_text_from_images(media_urls):
    """Processes multiple images, extracts, and combines text as one document."""
    extracted_texts = []

    for url in media_urls:
        image = download_image(url)
        if image is None:
            continue

        # Extract text using pytesseract
        text = pytesseract.image_to_string(image, lang="eng", config="--psm 6").strip()
        print(f"📝 Extracted Text: {text[:300]}")  # Print first 300 characters for debugging
        
        if text:
            extracted_texts.append(text)

    # Combine all extracted text into one document
    return "\n".join(extracted_texts) if extracted_texts else None

def anonymize_text(text):
    """Anonymizes sensitive data using Presidio."""
    try:
        results = analyzer.analyze(text=text, language="en")
        anonymized_text = anonymizer.anonymize(text=text, analyzer_results=results)
        return anonymized_text.text
    except Exception as e:
        print(f"❌ Anonymization Error: {e}")
        return text  # If anonymization fails, return original text

def summarize_text_with_gpt(text):
    """Summarizes extracted text using GPT-4."""
    try:
        prompt = f"Summarize this text in simple language:\n\n{text}"
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a helpful AI assistant that summarizes text."},
                      {"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Summarization Error: {e}")
        return "Summary not available."

@app.route("/webhook", methods=["POST"])
def webhook():
    """Handles incoming Twilio messages."""
    print("\n📩 Received Twilio Request:")
    print(request.form)

    from_number = request.form.get("From")
    num_media = int(request.form.get("NumMedia", 0))
    body_text = request.form.get("Body", "").strip().lower()
    response = MessagingResponse()

    # 🖐️ Handle Greeting Messages
    if body_text in ["hi", "hello", "hey"]:
        response.message("👋 Hi! Please send one or more images, and I'll extract and summarize the content for you.")
        print("✅ Greeting message sent.")
        return Response(str(response), mimetype="application/xml")  # ✅ FIXED XML RESPONSE FORMAT

    # 🖼️ If images are sent, batch them before processing
    if num_media > 0:
        media_urls = [request.form.get(f"MediaUrl{i}") for i in range(num_media)]
        print(f"🖼️ Number of Media Files: {num_media}")

        # Store images in batch
        if from_number not in image_batches:
            image_batches[from_number] = {"images": [], "timestamp": time.time()}

        image_batches[from_number]["images"].extend(media_urls)

        # **NEW: Process all images in background thread**
        Timer(IMAGE_BATCH_TIMEOUT, process_image_batch, args=[from_number]).start()

        return Response("", status=200)  # ✅ Prevents Twilio from dropping request due to timeout

    # 📝 If user responds with a query and session exists, process the query
    elif body_text and from_number in user_sessions:
        extracted_text = user_sessions.pop(from_number)  # Retrieve and remove stored text
        print(f"💬 User Query: {body_text}")

        reply = summarize_text_with_gpt(f"Provide a concise and direct response (atmost 3 lines) in exactly two lines: {body_text}\n\n{extracted_text}")
        response.message(reply)
        print("✅ Response sent to user.")
        return Response(str(response), mimetype="application/xml")  # ✅ FIXED RESPONSE FORMAT

    # If message is unrelated (e.g., no session exists), prompt user
    else:
        response.message("Please send an image first so I can extract text and assist you!")
        print("⚠️ No prior session found. Asking user to send an image.")
        return Response(str(response), mimetype="application/xml")

def process_image_batch(from_number):
    """Processes all images in a batch and sends the summary."""
    if from_number not in image_batches:
        return

    media_urls = image_batches.pop(from_number)["images"]
    extracted_text = extract_text_from_images(media_urls)

    if not extracted_text:
        send_twilio_message(from_number, "❌ Could not extract text from the images. Please try again.")
        return

    # Anonymize and summarize the text
    anonymized_text = anonymize_text(extracted_text)
    summary = summarize_text_with_gpt(anonymized_text)

    # Store extracted text in user session
    user_sessions[from_number] = anonymized_text

    # Send the summary response back
    send_twilio_message(from_number, f"🔎 Summary (All Images Combined):\n{summary}\n\n❓ What do you want to know from this?")

def send_twilio_message(to, message):
    """Sends a WhatsApp message via Twilio API."""
    requests.post(
        f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json",
        auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
        data={"From": twilio_number, "To": to, "Body": message}
    )

if __name__ == "__main__":
    app.run(debug=True, port=5001)
