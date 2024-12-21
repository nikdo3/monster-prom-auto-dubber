import cv2
import numpy as np
import os

known_histograms = []
known_names = []

def load_known_images():
    """Načte všechny známé obrázky z adresáře a uloží jejich histogramy."""
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
    """Předzpracuje snímek (stupně šedi, invertování, adaptivní prahování)."""
    # Převod na stupně šedi
    gray = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2GRAY)
    # Invertování obrazu
    inverted = cv2.bitwise_not(gray)
    # Adaptivní prahování
    binary = cv2.adaptiveThreshold(inverted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return binary

def compare_images(im):
    """Porovná histogram zadaného obrázku s uloženými a vrátí nejbližší shodu."""
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