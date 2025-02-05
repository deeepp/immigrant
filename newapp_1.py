import os
import pytesseract
import requests
from openai import OpenAI
from PIL import Image
from io import BytesIO
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# Twilio setup
account_sid = ''  # Replace with your actual Account SID
auth_token = ''    # Replace with your actual Auth Token
twilio_number = 'whatsapp:+14155238886'  # Twilio sandbox number or your WhatsApp business number

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="",
)

app = Flask(__name__)

# Dictionary to store user state
user_state = {}
user_summaries = {}

@app.route("/webhook", methods=["POST"])
def webhook():
    incoming_msg = request.form.get("Body", "").strip().lower()
    sender = request.form.get("From")  
    media_url = request.form.get("MediaUrl0")  # First media file (image/audio)
    media_type = request.form.get("MediaContentType0","").lower() # Media type (audio/image)

    response = MessagingResponse()

    # Ensure user has a session state
    if sender not in user_state:
        user_state[sender] = "main_menu"
    print(f"user state : {user_state[sender]}")

    # ✅ Global exit command (Reset session)
    if incoming_msg in ["exit", "menu", "restart", "no", "nah", "skip"]:
        print("resetting bhai!")
        response.message("Returning to the main menu.")
        user_state[sender] = "main_menu"
        response.message("Welcome! Please select an option:\n1️⃣ Resources\n2️⃣ Document Processing\n3️⃣ Housing Groups\n🎙️ You can also send a voice note to chat!")
        print("sent main menu message sur")
        return str(response)
    if incoming_msg in ['hi',"Hello","hello","hey"]:
        print("user greeted ")
        response.message("Hello! How can I assist you today?:😊\n \n1️⃣ Resources\n2️⃣ Document Processing\n3️⃣ Housing Groups\n🎙️ You can also send a voice note to chat!")

        user_state[sender]='waiting_for_selection'
        return str(response)
    # ✅ Main Menu 
    if user_state[sender] == "main_menu":
        response.message("Welcome! Please select an option:\n1️⃣ Resources\n2️⃣ Document Processing\n3️⃣ Housing Groups\n🎙️ Send a voice note for help!")
        user_state[sender] = "waiting_for_selection"
        return str(response)
    

    # ✅ User Selection Handling
    if user_state[sender] == "waiting_for_selection":
        if incoming_msg == "1":
            response.message("Here are some useful resources:\n- Visa Info: [link]\n- Job Search Tips: [link]\n- Legal Assistance: [link]")
            user_state[sender] = "main_menu"
        elif incoming_msg == "2":
            response.message("Please send an image of the document you want to process.")
            user_state[sender] = "waiting_for_image"
        elif incoming_msg == "3":
            response.message("Here are some housing WhatsApp groups:\n- Group 1: [link]\n- Group 2: [link]")
            user_state[sender] = "main_menu"
        else:
            response.message("Invalid selection. Please choose 1, 2, or 3.")
        return str(response)

    # ✅ Image Processing
    if user_state[sender] == "waiting_for_image":
        if not media_url or not media_type.startswith("image"):
            response.message("No image detected. Please send a document image or type 'menu' to go back.")
            return str(response)

        try:
            extracted_text = extract_text_from_image(media_url)
            if not extracted_text.strip():
                raise ValueError("No text extracted from the image.")

            # **Anonymization**
            results = analyzer.analyze(text=extracted_text, language="en")
            anonymised_text = anonymizer.anonymize(text=extracted_text, analyzer_results=results)

            # **Summarization**
            summarized_text = summarize_text_with_gpt(anonymised_text)

            # **Extract Dates and Headings**
            date_heading_results = extract_dates_and_headings_with_gpt(anonymised_text)
            date_summary = f"\n\nDeadlines and Tasks:\n{date_heading_results}" if date_heading_results else "\n\nNo deadlines or tasks found."

            # **Send Summary**
            response.message(f"Summary: {summarized_text}{date_summary}\n\nWould you like this translated into Spanish? Reply with 'Yes' or 'Translate'.")

            # **Store Summary & Update State**
            user_summaries[sender] = summarized_text
            user_state[sender] = "waiting_for_translation"

        except Exception as e:
            print(f"Error processing image: {e}")
            response.message("❌ Failed to process the image. Ensure it's a clear document and try again.")
            user_state[sender] = "main_menu"
        return str(response)

    # ✅ Audio Processing (Voice Messages)
    if media_url and media_type.startswith("audio"):
        try:
            extracted_text = process_audio(media_url)
            response.message(f"🗣️ Transcribed: {extracted_text}\n\nReply with '1' for Resources, '2' for Documents, or '3' for Housing Groups.")
            user_state[sender] = "waiting_for_selection"
        except Exception as e:
            response.message("❌ Failed to process audio. Try again.")
            print(f"Audio Processing Error: {e}")
            user_state[sender] = "main_menu"

    # ✅ Translation Handling
    if user_state[sender] == "waiting_for_translation":
        if incoming_msg in ["yes", "translate", "spanish", "sí"]:
            if sender in user_summaries:
                translated_text = translate_text_with_gpt(user_summaries[sender], "Spanish")
                response.message(f"Translated to Spanish: {translated_text}")
            else:
                response.message("No summary found to translate. Please send an image first.")
        else:
            response.message("Okay! Returning to the main menu.")

        user_state[sender] = "main_menu"
    else:
        response.message("I'm sorry , I didnt understand ! please type menu to return to the main menu.")
    

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
def process_audio(audio_url):
    """Downloads and transcribes audio using OpenAI Whisper."""
    response = requests.get(audio_url, auth=(account_sid, auth_token))
    if response.status_code != 200:
        raise Exception("Failed to download audio from Twilio.")

    # Save the file temporarily
    audio_path = "temp_audio.ogg"
    with open(audio_path, "wb") as f:
        f.write(response.content)

    # Convert to WAV format for Whisper
    audio = AudioSegment.from_file(audio_path)
    wav_path = "temp_audio.wav"
    audio.export(wav_path, format="wav")

    # Transcribe using OpenAI Whisper
    with open(wav_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)

    # Cleanup temporary files
    os.remove(audio_path)
    os.remove(wav_path)

    return transcript["text"]
    
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


if __name__ == "__main__":
    app.run(debug=True, port=5006)


