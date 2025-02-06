import os
import pytesseract
import requests
import time
from PIL import Image
from io import BytesIO
from flask import Flask, request, Response
from twilio.twiml.messaging_response import MessagingResponse
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from threading import Timer
import json

# Initialize Flask app
app = Flask(__name__)

# Twilio API Credentials
TWILIO_ACCOUNT_SID = 'AC979b7c1910156016bcd92fb11773217e'  # Replace with actual Account SID
TWILIO_AUTH_TOKEN = 'b792f7ce9131d3b5fc1d2789f708bbad'  # Replace with actual Auth Token
twilio_number = 'whatsapp:+14155238886'  # Twilio WhatsApp number

# Initialize Presidio for text anonymization
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# Store user sessions & batch images
user_sessions = {}
image_batches = {}

# Image batch timeout (seconds)
IMAGE_BATCH_TIMEOUT = 3  # Reduced timeout for better performance

def download_image(image_url):
    """Downloads an image from Twilio and processes it."""
    try:
        print(f"📷 Downloading image from: {image_url}")
        response = requests.get(image_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))

        if response.status_code != 200:
            print(f"❌ Failed to download image. Status code: {response.status_code}")
            return None

        # Open and convert image to grayscale
        image = Image.open(BytesIO(response.content)).convert("L")
        valid_formats = ["JPEG", "PNG", "BMP", "GIF", "TIFF", "WEBP"]
        if image.format not in valid_formats:
            print(f"❌ Unsupported image format: {image.format}")
            return None
        return image.convert("L")

    except Exception as e:
        print(f"❌ Image Processing Error: {e}")
        return None

    except Exception as e:
        print(f"❌ Image Processing Error: {e}")
        return None

def extract_text_from_images(media_urls):
    """Processes multiple images, extracts, and combines text as one document."""
    extracted_texts = []

    for url in media_urls:
        image = download_image(url)
        if image:
            text = pytesseract.image_to_string(image, lang="eng", config="--psm 6").strip()
            print(f"📝 Extracted Text: {text[:300]}")  # Debugging: Print first 300 characters
            if text:
                extracted_texts.append(text)

    return "\n".join(extracted_texts) if extracted_texts else None

def anonymize_text(text):
    """Anonymizes sensitive data using Presidio."""
    try:
        results = analyzer.analyze(text=text, language="en")
        return anonymizer.anonymize(text=text, analyzer_results=results).text
    except Exception as e:
        print(f"❌ Anonymization Error: {e}")
        return text






def query_llama3(prompt):
    """Queries the local Llama 3 model using Ollama and correctly handles streaming responses."""
    try:
        response = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={"model": "llama3", "prompt": prompt},
            stream=True,  # ✅ Enable streaming
            timeout=(10, 30)  # ✅ Proper timeout settings
        )

        collected_response = ""  # ✅ Store the final response text

        print("\n🔹 DEBUG: Starting to process Ollama stream...\n")

        # ✅ Read the response line-by-line to handle streaming properly
        for line in response.iter_lines():
            if line:
                try:
                    # ✅ Decode and parse the JSON response properly
                    json_line = json.loads(line.decode("utf-8").strip())

                    # ✅ If response contains text, append it
                    if "response" in json_line:
                        collected_response += json_line["response"]
                        print(f"📝 Chunk Received: {json_line['response']}")  # ✅ Debugging chunk-by-chunk output

                    # ✅ Stop processing if Ollama signals completion
                    if json_line.get("done", False):
                        break

                except json.JSONDecodeError as e:
                    print(f"❌ JSON Decode Error: {e}, Line: {line}")

        final_output = collected_response.strip() if collected_response else "Error: No response received."
        print("\n✅ Final Llama Response:", final_output)  # ✅ Debugging final output

        return final_output

    except requests.exceptions.Timeout:
        print("❌ Ollama Query Error: Request timed out.")
        return "Error: Timeout occurred."

    except Exception as e:
        print(f"❌ Ollama Query Error: {e}")
        return f"Error: {str(e)}"



def summarize_text(text):
    """Summarizes extracted text using Ollama Llama 3, ensuring actual content is summarized."""
    try:
        # 🔥 FORCE SUMMARIZATION PROMPT 🔥
        prompt = f"""
        Summarize the following text accurately. **Extract key details, names, dates, numbers, and important points.**
        **If there is any content, provide a summary. Do not respond with 'Not mentioned' or refuse to summarize.**

        **Text to Summarize:**
        {text}

        **Summary:**
        """

        response = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={"model": "llama3", "prompt": prompt},
            stream=True,  # ✅ Handle streaming response
            timeout=(10, 30)  # ✅ Prevent timeout issues
        )

        collected_response = ""  # ✅ Store full response

        print("\n🔹 DEBUG: Processing Ollama stream...\n")

        # ✅ Read response line-by-line to capture all output
        for line in response.iter_lines():
            if line:
                try:
                    json_line = json.loads(line.decode("utf-8").strip())

                    # ✅ Append received text if key "response" exists
                    if "response" in json_line:
                        collected_response += json_line["response"]
                        print(f"📝 Chunk Received: {json_line['response']}")  # ✅ Debugging chunk output

                    # ✅ Stop processing when Ollama signals completion
                    if json_line.get("done", False):
                        break

                except json.JSONDecodeError as e:
                    print(f"❌ JSON Decode Error: {e}, Skipping Line: {line}")

        final_summary = collected_response.strip() if collected_response else "Summary not available."
        print("\n✅ Final Summary:", final_summary)  # ✅ Debugging output

        return final_summary

    except requests.exceptions.Timeout:
        print("❌ Ollama Summarization Error: Request timed out.")
        return "Summary not available."

    except Exception as e:
        print(f"❌ Ollama Summarization Error: {e}")
        return "Summary not available."




def answer_user_question(question, document_text):
    """Provides **direct, accurate answers** to user questions."""
    prompt = f"""
    Based on the following document:

    {document_text}

    **Answer the user's question as accurately and concisely as possible.**
    - If the document **explicitly states the answer**, provide a **direct 'Yes' or 'No'**.
    - If additional context is needed, provide **a single-sentence explanation**.
    - If the answer is **not mentioned**, respond with: "Not mentioned in the document."

    **User Question:** {question}
    """
    return query_llama3(prompt)

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
        return Response(str(response), mimetype="application/xml")

    # 🖼️ If images are sent, store them in a batch before processing
    if num_media > 0:
        media_urls = [request.form.get(f"MediaUrl{i}") for i in range(num_media)]
        print(f"🖼️ Number of Media Files: {num_media}")

        # Store images in batch
        if from_number not in image_batches:
            image_batches[from_number] = {"images": [], "timestamp": time.time()}

        image_batches[from_number]["images"].extend(media_urls)

        # ✅ **Process all images together after timeout**
        Timer(IMAGE_BATCH_TIMEOUT, process_image_batch, args=[from_number]).start()

        return Response("", status=200)

    # 📝 If user asks a follow-up question
    elif body_text and from_number in user_sessions:
        if body_text in ["exit", "no", "stop", "quit"]:
            del user_sessions[from_number]
            response.message("Hope that was useful! See you soon 👋")
            return Response(str(response), mimetype="application/xml")

        extracted_text = user_sessions[from_number]  # Keep session active
        print(f"💬 User Query: {body_text}")

        reply = answer_user_question(body_text, extracted_text)
        response.message(reply + "\n\n❓ Ask another question or type 'exit' to stop.")
        print("✅ Response sent to user.")
        return Response(str(response), mimetype="application/xml")

    # If no valid request, prompt user
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

    # ✅ **Ensure all images are processed as one document**
    combined_text = anonymize_text(extracted_text)
    summary = summarize_text(combined_text)

    # Store extracted text in user session
    user_sessions[from_number] = combined_text

    # ✅ **Send only one message with all images summarized together**
    send_twilio_message(from_number, f"🔎 Summary (All Images Combined):\n{summary}\n\n❓ What do you want to know from this?")

def send_twilio_message(to, message):
    """Sends a WhatsApp message via Twilio API, splitting it if it exceeds the character limit."""
    max_length = 1600  # Twilio's character limit per message
    messages = [message[i:i + max_length] for i in range(0, len(message), max_length)]

    for msg in messages:
        requests.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json",
            auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
            data={"From": twilio_number, "To": to, "Body": msg}
        )

@app.route("/test", methods=["POST"])
def test_summarization():
    """Test the summarization functionality without Twilio."""
    data = request.get_json()
    
    if not data or "text" not in data:
        return {"error": "No text provided."}, 400  # Return error if text is missing

    text = data["text"]
    summary = summarize_text(text)

    return {"summary": summary}, 200  # Return summary as JSON response


if __name__ == "__main__":
    app.run(debug=True, port=5001)

