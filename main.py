import cv2
import numpy as np
from PIL import ImageGrab
import pytesseract
import pyttsx3
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Načtení známých obrázků a jejich histogramů
known_histograms = []
known_names = []

def load_known_images():
    known_images_dir = "known_images"
    for filename in os.listdir(known_images_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(known_images_dir, filename)
            img = cv2.imread(img_path)
            hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            known_histograms.append(hist)
            known_names.append(os.path.splitext(filename)[0])

def preprocess_image(im):
    # Převod na stupně šedi
    gray = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2GRAY)
    # Adaptivní prahování
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return binary

def read_screen_text(x1, y1, x2, y2):
    # Snímek obrazovky z definované oblasti
    im = ImageGrab.grab(bbox=(x1, y1, x2, y2))
    # Předzpracování obrazu
    preprocessed_im = preprocess_image(im)
    # Čtení textu z obrazovky s whitelistem znaků
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .'
    text = pytesseract.image_to_string(preprocessed_im, config=custom_config)
    text = text.replace('\n', ' ')
    print("Captured Text:", text)
    return text

def compare_images(im):
    # Vypočítání histogramu pro detekovaný obrázek
    hist = cv2.calcHist([im], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # Porovnání histogramu s histogramy známých obrázků
    best_match = None
    min_distance = float('inf')
    for known_hist, name in zip(known_histograms, known_names):
        distance = cv2.compareHist(hist, known_hist, cv2.HISTCMP_BHATTACHARYYA)
        if distance < min_distance:
            min_distance = distance
            best_match = name

    return best_match

def read_out_loud(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

if __name__ == "__main__":
    # Načtení známých obrázků
    load_known_images()

    # Definujte souřadnice oblasti, kterou chcete zachytit
    x1, y1, x2, y2 = 340, 820, 1600, 1035

    # Čtení textu z obrazovky
    text = read_screen_text(x1, y1, x2, y2)

    # Snímek obrazovky z definované oblasti pro porovnání obrázků
    im = np.array(ImageGrab.grab(bbox=(x1, y1, x2, y2)))

    # Porovnání obrázků
    matched_name = compare_images(im)

    # Přiřazení jména ke čtenému textu
    if matched_name:
        print(f"Detected {matched_name} with text: {text}")
        read_out_loud(f"{matched_name} says: {text}")
    else:
        print("No match found.")