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
import openai
from gtts import gTTS
import pickle
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS


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
ngrokurl = "https://cb47-2607-fb91-320b-10bd-d488-d5d8-e2f7-5928.ngrok-free.app"
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
    media_url = request.form.get("MediaUrl0")  # First media file (image/audio)
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

    # User Selection Handling
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

    # Additional handling for image or other types of content
    if user_state[sender] == "waiting_for_image":
        if not media_url or not media_type.startswith("image"):
            response.message("No image detected. Please send a document image or type 'menu' to go back.")
            return str(response)

        try:
            extracted_text = extract_text_from_image(media_url)
            if not extracted_text.strip():
                raise ValueError("No text extracted from the image.")
            
            # Summarizing and converting to audio
            summarized_text = summarize_text_with_gpt(extracted_text)
            response.message(f"Summary: {summarized_text}")
            audio_file_summary = text_to_audio(summarized_text, "summary.mp3")
            media_url = f"{ngrokurl}/audio_files/summary.mp3"
            response.message("Audio:", media_url=media_url)

            # Storing user summary and updating state
            user_summaries[sender] = summarized_text
            user_state[sender] = "waiting_for_translation"

        except Exception as e:
            print(f"Error processing image: {e}")
            response.message("❌ Failed to process the image. Ensure it's a clear document and try again.")
            user_state[sender] = "main_menu"
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
    with open(wav_path, "rb") as f:
        transcript = openai.Audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="text"
        )

    # Cleanup temporary files
    os.remove(audio_path)
    os.remove(wav_path)

    return transcript.strip()

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

def split_document(text, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap, 
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text(text)

def store_embeddings(chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")  
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)  
    return vectorstore

def retrieve_relevant_chunks(query, vectorstore, top_k=3):
    retrieved_docs = vectorstore.similarity_search(query, k=top_k)
    return [doc.page_content for doc in retrieved_docs]

def generate_response(query, relevant_chunks):
    context = "\n".join(relevant_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a helpful AI assistant."},
                  {"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

@app.route("/query", methods=["POST"])
def text_query():
    incoming_msg = request.form.get("Body", "").strip()
    vectorstore = load_faiss_vectorstore()  # Load stored embeddings
    relevant_chunks = retrieve_relevant_chunks(incoming_msg, vectorstore)
    response_text = generate_response(incoming_msg, relevant_chunks)
    
    response = MessagingResponse()
    response.message(response_text)
    return str(response)

@app.route("/audio_query", methods=["POST"])
def audio_query():
    media_url = request.form.get("MediaUrl0")
    media_type = request.form.get("MediaContentType0", "").lower()
    response = MessagingResponse()

    if media_url and media_type.startswith("audio"):
        try:
            transcribed_text = process_audio(media_url)
            vectorstore = load_faiss_vectorstore()  # Load stored embeddings
            relevant_chunks = retrieve_relevant_chunks(transcribed_text, vectorstore)
            rag_response = generate_response(transcribed_text, relevant_chunks)
            
            response.message(f"🗣️ Transcribed: {transcribed_text}\n\nResponse: {rag_response}")
        except Exception as e:
            response.message("❌ Failed to process audio. Try again.")
            print(f"Audio Processing Error: {e}")
    else:
        response.message("No audio detected. Please send an audio message.")
    
    return str(response)

def save_faiss_vectorstore(vectorstore, filepath="vectorstore.pkl"):
    with open(filepath, "wb") as f:
        pickle.dump(vectorstore, f)

def load_faiss_vectorstore(filepath="vectorstore.pkl"):
    with open(filepath, "rb") as f:
        return pickle.load(f)
    

def text_to_audio(text, filename="audio.mp3"):
    # Use gTTS to convert text to speech and save it in the 'audio_files' directory
    audio_path = os.path.join('audio_files', filename)
    tts = gTTS(text, lang='en')
    tts.save(audio_path)
    return audio_path



if __name__ == "__main__":
    app.run(debug=True, port=5001)