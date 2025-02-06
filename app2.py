import os
import pytesseract
import requests
from openai import OpenAI
from PIL import Image
from io import BytesIO
from flask import Flask, send_from_directory, request, Response
from twilio.twiml.messaging_response import MessagingResponse
from presidio_analyzer import AnalyzerEngine
from flask import send_from_directory
from presidio_anonymizer import AnonymizerEngine
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydub import AudioSegment
from openai import OpenAI

client = OpenAI()
from gtts import gTTS
import pickle
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import requests
from openai import OpenAI

client = OpenAI()
from pydub import AudioSegment


analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

account_sid = os.getenv("ACCOUNT_SID")
auth_token = os.getenv("AUTH_TOKEN")
twilio_number = os.getenv("TWILIO_NUMBER")
openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

app = Flask(__name__)

# Dictionary to store user state and summaries
user_state = {}
user_summaries = {}
ngrokurl = "https://4329-216-165-95-191.ngrok-free.app"
os.makedirs('audio_files', exist_ok=True)

@app.route('/audio_files/<filename>')
def serve_audio(filename):
    file_path = f"audio_files/{filename}"

    # Check if the file exists
    if not os.path.exists(file_path):
        return f"File {filename} not found", 404

    # Log the Range header for debugging
    range_header = request.headers.get("Range", None)
    print(f"Range header: {range_header}")

    if not range_header:
        return send_from_directory("audio_files", filename, mimetype="audio/mpeg")

    # Extract range values
    start, end = range_header.replace("bytes=", "").split("-")
    start = int(start) if start else 0
    end = int(end) if end else None

    with open(file_path, "rb") as f:
        f.seek(start)
        data = f.read() if end is None else f.read(end - start + 1)

    # Log the data length
    print(f"Data length: {len(data)}")

    # Create the response with partial content (206)
    response = Response(data, status=206, mimetype="audio/mpeg")
    response.headers["Content-Range"] = f"bytes {start}-{end if end else len(data)-1}/{len(data)}"
    response.headers["Accept-Ranges"] = "bytes"

    return response

@app.route("/webhook", methods=["POST"])
def webhook():
    incoming_msg = request.form.get("Body", "").strip().lower()
    sender = request.form.get("From")
    media_url = request.form.get("MediaUrl0")  # First media file (audio/image)
    media_type = request.form.get("MediaContentType0", "").lower()  # Media type (audio/image)

    response = MessagingResponse()

    # Ensure user has a session state
    if sender not in user_state:
        user_state[sender] = "main_menu"
    print(f"user state : {user_state[sender]}")

    # Global exit command (Reset session)
    if incoming_msg in ["exit", "menu", "restart", "no", "nah", "skip"]:
        print("resetting bhai!")
        response.message("Returning to the main menu.")
        user_state[sender] = "main_menu"
        response.message("Welcome! Please select an option:\n1️⃣ Resources\n2️⃣ Document Processing\n3️⃣ Housing Groups\n🎙️ You can also send a voice note to chat!")
        return str(response)

    # Process audio if received
    if media_url and media_type.startswith("audio"):
        print(f"Received audio URL: {media_url}")

        try:
            # Transcribe audio
            transcription = audio_to_text(media_url)
            print(f"Transcription: {transcription}")

            # Send transcription response
            response.message(f"Transcription: {transcription}")

            return str(response)
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            response.message("❌ Failed to transcribe audio. Please try again.")
            return str(response)

    # Main Menu Handling
    if user_state[sender] == "main_menu":
        main_menu_msg = "Welcome! Please select an option:\n1️⃣ Resources\n2️⃣ Document Processing\n3️⃣ Housing Groups\n🎙️ Send a voice note for help!"
        response.message(main_menu_msg)
        audio_file = text_to_audio(main_menu_msg, "main_menu.mp3")
        media_url = f"{ngrokurl}/audio_files/main_menu.mp3"
        response.message().media(media_url)
        print(media_url)
        user_state[sender] = "waiting_for_selection"
        return str(response)

    # Additional logic for selections
    if user_state[sender] == "waiting_for_selection":
        if incoming_msg == "1":
            resources_text = "Here are some useful resources:\n- Visa Info: [link]\n- Job Search Tips: [link]\n- Legal Assistance: [link]"
            response.message(resources_text)
            audio_file = text_to_audio(resources_text, "resources.mp3")
            media_url = f"{ngrokurl}/audio_files/resources.mp3"
            response.message("Audio:", media_url=media_url)
            user_state[sender] = "main_menu"
        elif incoming_msg == "2":
            response.message("Please send an image of the document you want to process.")
            user_state[sender] = "waiting_for_image"
        elif incoming_msg == "3":
            housing_groups_text = "Here are some housing WhatsApp groups:\n- Group 1: [link]\n- Group 2: [link]"
            response.message(housing_groups_text)
            audio_file = text_to_audio(housing_groups_text, "housing_groups.mp3")
            media_url = f"{ngrokurl}/audio_files/housing_groups.mp3"
            response.message("Audio:", media_url=media_url)
            user_state[sender] = "main_menu"
        else:
            response.message("Invalid selection. Please choose 1, 2, or 3.")
        return str(response)

    return str(response)




def extract_text_from_image(image_url):
    """Downloads an image from Twilio and extracts text using pytesseract."""
    try:
        print(f"📷 Downloading image from: {image_url}")

        # ✅ Authenticate with Twilio to access the image
        response = requests.get(image_url, auth=(account_sid,auth_token))

        if response.status_code != 200:
            raise Exception(f"❌ Failed to download image. Status code: {response.status_code}")

        # ✅ Save Image Temporarily for Debugging
        image = Image.open(BytesIO(response.content))
        image_path = "received_image.jpg"
        image.save(image_path)
        print(f"✅ Image saved as {image_path}")

        # ✅ Convert to Grayscale for Better OCR Accuracy
        image = image.convert("L")

        # ✅ Try OCR Extraction
        text = pytesseract.image_to_string(image)
        print(f"📝 Extracted Text: {text[:300]}")  # Print first 300 characters for debugging

        if not text.strip():
            raise ValueError("❌ No text detected in the image.")

        return text.strip()

    except Exception as e:
        print(f"❌ Image Processing Error: {e}")
        return ""

def anonymise(text):
    results = analyzer.analyze(text=text, language="en")
    anonymized_text = anonymizer.anonymize(text=text, analyzer_results=results)
    return anonymized_text

def summarize_text_with_gpt(text):
    prompt = f"Summarize this text in simple language:\n\n{text}"
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a helpful AI assistant that summarizes text."},
                  {"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

def translate_text_with_gpt(text, target_language):
    prompt = f"Translate the following text into {target_language}:\n\n{text}"
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a helpful AI assistant that translates text."},
                  {"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


def extract_dates_and_headings_with_gpt(text):
    prompt = f"""
    Extract all dates and their associated headings/tasks (e.g., deadlines, due dates) from the following text. 
    Return the results in the format:
    - Date: YYYY-MM-DD, Heading/Task: [description]
    - Date: YYYY-MM-DD, Heading/Task: [description]

    Text:
    {text}
    """
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a helpful AI assistant that extracts dates and their associated tasks from text."},
                  {"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


def text_to_audio(text, filename="audio.mp3"):
    # Use gTTS to convert text to speech and save it in the 'audio_files' directory
    audio_path = os.path.join('audio_files', filename)
    tts = gTTS(text, lang='en')
    tts.save(audio_path)
    return audio_path

import openai
import os
import requests
from pydub import AudioSegment

openai.api_key = os.getenv("OPENAI_API_KEY")  # Set your OpenAI API key here

def audio_to_text(audio_url):
    """Download an audio file from Twilio and transcribe it using OpenAI Whisper API."""
    try:
        # Twilio requires authentication to fetch media
        twilio_sid = os.getenv("ACCOUNT_SID")  # Store these in environment variables
        twilio_auth_token = os.getenv("AUTH_TOKEN")

        # Download the audio file with authentication
        print(f"Fetching audio from: {audio_url}")  # Log URL for debugging
        response = requests.get(audio_url, auth=(twilio_sid, twilio_auth_token))

        if response.status_code != 200:
            print(f"Failed to download audio. Status code: {response.status_code}, Response: {response.text}")
            return "Failed to download audio."

        # Save the audio file temporarily
        audio_path = "temp_audio.ogg"
        with open(audio_path, "wb") as f:
            f.write(response.content)

        # Convert to WAV format (Whisper API prefers WAV)
        audio = AudioSegment.from_file(audio_path)
        wav_path = "temp_audio.wav"
        audio.export(wav_path, format="wav")

        # Transcribe using OpenAI Whisper
        with open(wav_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",  # Specify the model for transcription
                file=f,
                response_format="text"  # Get the response as plain text
            )
            print(transcript)

        # Clean up files
        os.remove(audio_path)
        os.remove(wav_path)

        return transcript  # The transcript is returned as plain text

    except Exception as e:
        print(f"Error in transcription: {e}")
        return f"Error in transcription: {e}"


if __name__ == "__main__":
    app.run(debug=True, port=5001)