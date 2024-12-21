import os
import cv2
import numpy as np
from PIL import ImageGrab
import pytesseract
from elevenlabs import stream
from elevenlabs.client import ElevenLabs
from image_processing import known_histograms, known_names, preprocess_image
from voice_settings import change_voice, get_voice

with open("APIkeys.txt", "r") as f:
    api_key = f.read().strip()
if not api_key:
    raise ValueError("API key for ElevenLabs is not set.")
client = ElevenLabs(api_key=api_key)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ['PATH'] += r';C:\mpv'

def read_screen_text(x1, y1, x2, y2):
    """Zachytí danou oblast obrazovky, předzpracuje ji a vrátí rozpoznaný text."""
    im = ImageGrab.grab(bbox=(x1, y1, x2, y2))
    preprocessed_im = preprocess_image(im)
    text = pytesseract.image_to_string(preprocessed_im, lang='eng').replace('\n', ' ')
    print("Captured Text:", text)
    return text

def read_out_loud(text, character):
    """Přečte nahlas předaný text zadanou postavou (hlasem)."""
    voice_id = get_voice(character)
    print(f"Using voice_id: {voice_id} for character: {character}")  # Debug výpis
    audio_stream = client.text_to_speech.convert_as_stream(
        text=text,
        voice_id=voice_id,
        model_id="eleven_multilingual_v2"
    )
    stream(audio_stream)

def evaluate_voice_for_personality(description):
    """Vyhodnotí (pomocí LLM) vhodný hlas na základě popisu osobnosti."""
    # Pro ukázku vracíme platný voice_id
    # V reálné aplikaci by zde bylo volání LLM nebo jiná logika pro výběr hlasu
    return "CwhRBWXzGAHq8TQ4Fs17"  # Platný voice_id

def learning_mode(im):
    """Prompts user for name/personality a uloží novou postavu včetně hlasu."""
    cv2.imshow("Unknown Face", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    name = input("Enter the name for the unknown face: ")
    personality_description = input("Enter a short personality description: ")
    
    known_images_dir = "known_images"
    img_path = os.path.join(known_images_dir, f"{name}.jpg")
    
    counter = 1
    while os.path.exists(img_path):
        img_path = os.path.join(known_images_dir, f"{name}_{counter}.jpg")
        counter += 1
    
    cv2.imwrite(img_path, im)
    
    hist = cv2.calcHist([im], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    known_histograms.append(hist)
    known_names.append(name)

    voice_id = evaluate_voice_for_personality(personality_description)
    change_voice(name, voice_id)