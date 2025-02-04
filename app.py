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
account_sid = '_'  # Replace with your actual Account SID
auth_token = '_'    # Replace with your actual Auth Token
twilio_number = 'whatsapp:+14155238886'  # Twilio sandbox number or your WhatsApp business number

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="_",
)

app = Flask(__name__)

# Dictionary to store user state
user_state = {}
user_summaries = {}

@app.route("/webhook", methods=["POST"])
def webhook():
    incoming_msg = request.form.get("Body", "").strip().lower()
    sender = request.form.get("From")  # Extract sender's number
    media_url = request.form.get("MediaUrl0")  # Get the first media URL if available

    response = MessagingResponse()

    if sender not in user_state:
        user_state[sender] = "main_menu"

    if user_state[sender] == "main_menu":
        response.message("Welcome! Please select an option:\n1️⃣ Resources\n2️⃣ Document Processing\n3️⃣ Housing Groups")
        user_state[sender] = "waiting_for_selection"
    
    elif user_state[sender] == "waiting_for_selection":
        if incoming_msg == "1":
            response.message("Here are some useful resources:\n- Visa Information: [link]\n- Job Search Tips: [link]\n- Legal Assistance: [link]")
            user_state[sender] = "main_menu"
        elif incoming_msg == "2":
            response.message("Please send an image of the document you want to process.")
            user_state[sender] = "waiting_for_image"
        elif incoming_msg == "3":
            response.message("Here are some housing WhatsApp groups:\n- Group 1: [https://chat.whatsapp.com/C0vehIlSJIc5mGzWWvhIOY]\n- Group 2: [https://chat.whatsapp.com/CvxidrvmX5bITh8kXOl21m]\n- Group 3: [link]")
            user_state[sender] = "main_menu"
        else:
            response.message("Invalid selection. Please choose 1, 2, or 3.")
    
    elif user_state[sender] == "waiting_for_image" and media_url:
        try:
            # Step 1: Extract text from the image
            extracted_text = extract_text_from_image(media_url)
            
            # Step 2: Anonymize the extracted text
            results = analyzer.analyze(text=extracted_text, language="en")
            anonymised_text = anonymizer.anonymize(text=extracted_text, analyzer_results=results)
            
            # Step 3: Summarize the anonymized text
            summarized_text = summarize_text_with_gpt(anonymised_text)
            
            # Step 4: Extract dates and headings from the anonymized text
            date_heading_results = extract_dates_and_headings_with_gpt(anonymised_text)
            
            # Step 5: Prepare the response message
            if date_heading_results:
                date_summary = "\n\nDeadlines and Tasks:\n" + date_heading_results
            else:
                date_summary = "\n\nNo deadlines or tasks found."
            
            # Step 6: Send the summary and dates/headings to the user
            response.message(f"Summary: {summarized_text}{date_summary}\n\nWould you like this translated into Spanish? Reply with 'Yes' or 'Translate'.")
            
            # Store the summary for potential translation
            user_summaries[sender] = summarized_text
            user_state[sender] = "waiting_for_translation"
        except Exception as e:
            response.message("Failed to process the image. Try again.")
            print(f"Error: {e}")
    
    elif user_state[sender] == "waiting_for_translation" and incoming_msg in ["yes", "translate", "spanish", "sí"]:
        if sender in user_summaries:
            translated_text = translate_text_with_gpt(user_summaries[sender], "Spanish")
            response.message(f"Translated to Spanish: {translated_text}")
        else:
            response.message("No summary found to translate. Please send an image first.")
        user_state[sender] = "main_menu"
    
    return str(response)



def extract_text_from_image(image_url):
    response = requests.get(image_url, auth=(account_sid, auth_token))
    if response.status_code != 200:
        raise Exception("Failed to download image from Twilio.")
    image = Image.open(BytesIO(response.content))
    return pytesseract.image_to_string(image).strip()

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


if __name__ == "__main__":
    app.run(debug=True, port=5001)


