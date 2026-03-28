from sarvamai import SarvamAI

import os
from dotenv import load_dotenv

load_dotenv()

client = SarvamAI(
    api_subscription_key=os.getenv("SARVAM_API_KEY")
)

LANGUAGE_CODES = {
    "hi-IN": "Hindi",
    "bn-IN": "Bengali",
    "kn-IN": "Kannada",
    "ml-IN": "Malayalam",
    "mr-IN": "Marathi",
    "od-IN": "Odia",
    "pa-IN": "Punjabi",
    "ta-IN": "Tamil",
    "te-IN": "Telugu",
    "gu-IN": "Gujarati",
    "as-IN": "Assamese",
    "en-IN": "English",
    "ur-IN": "Urdu",
    "ne-IN": "Nepali",
    "sa-IN": "Sanskrit",
}

def identify_language(question):

    response = client.text.identify_language(
        input=question
    )

    language_code = response.language_code

    language = LANGUAGE_CODES.get(language_code,language_code)

    return language_code, language

def language_translation(question,language_code):

    response = client.text.translate(
        input=question,
        source_language_code=language_code,
        target_language_code="en-IN"
    )

    return response.translated_text

def sarvam_call(question):

    language_code, language = identify_language(question)

    if language_code == "en-IN":
        return question, "English"
    else:
        question = language_translation(question,language_code)

        return question, language

