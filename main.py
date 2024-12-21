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
    # Invertování obrazu
    inverted = cv2.bitwise_not(gray)
    # Adaptivní prahování
    binary = cv2.adaptiveThreshold(inverted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return binary

def read_screen_text(x1, y1, x2, y2):
    # Snímek obrazovky z definované oblasti
    im = ImageGrab.grab(bbox=(x1, y1, x2, y2))
    # Předzpracování obrazu
    preprocessed_im = preprocess_image(im)
    
    """
    # Zobrazení předzpracovaného obrazu
    cv2.imshow("Preprocessed Image", preprocessed_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    
    # Čtení textu z obrazovky s whitelistem znaků
    text = pytesseract.image_to_string(preprocessed_im, lang='eng')
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
        if distance < min_distance and distance < 0.3:  # Přidání prahové hodnoty pro lepší rozpoznání
            min_distance = distance
            best_match = name

    return best_match

def read_out_loud(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def learning_mode(im):
    # Zobrazení obrázku a dotaz na jméno
    cv2.imshow("Unknown Face", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    name = input("Enter the name for the unknown face: ")
    
    # Uložení obrázku s novým jménem
    known_images_dir = "known_images"
    img_path = os.path.join(known_images_dir, f"{name}.jpg")
    
    # Pokud soubor již existuje, přidejte číslo k názvu souboru
    counter = 1
    while os.path.exists(img_path):
        img_path = os.path.join(known_images_dir, f"{name}_{counter}.jpg")
        counter += 1
    
    cv2.imwrite(img_path, im)
    
    # Aktualizace známých obrázků a histogramů
    hist = cv2.calcHist([im], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    known_histograms.append(hist)
    known_names.append(name)

if __name__ == "__main__":
    # Načtení známých obrázků
    load_known_images()

    # Definujte souřadnice oblasti, kterou chcete zachytit pro text
    x1_text, y1_text, x2_text, y2_text = 365, 820, 1600, 1035

    # Čtení textu z obrazovky
    text = read_screen_text(x1_text, y1_text, x2_text, y2_text)

    # Definujte souřadnice oblasti, kterou chcete zachytit pro hlavy
    x1_head, y1_head, x2_head, y2_head = 190, 750, 365, 930

    # Snímek obrazovky z definované oblasti pro porovnání obrázků
    im = np.array(ImageGrab.grab(bbox=(x1_head, y1_head, x2_head, y2_head)))

    # Porovnání obrázků
    matched_name = compare_images(im)

    # Přiřazení jména ke čtenému textu
    if matched_name:
        # Odstranění čísel a podtržítek z názvu
        matched_name_clean = ''.join([i for i in matched_name if not i.isdigit() and i != '_'])
        print(f"Detected {matched_name_clean} with text: {text}")
        read_out_loud(f"{matched_name_clean} says: {text}")
    else:
        print("No match found.")
        learning_mode(im)