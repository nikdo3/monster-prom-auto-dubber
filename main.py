import numpy as np
import cv2
from PIL import ImageGrab
from image_processing import load_known_images, compare_images
from app_logic import read_screen_text, read_out_loud, read_out_loud_legacy, learning_mode

if __name__ == "__main__":
    load_known_images()

    # Souřadnice pro čtení textu
    x1_text, y1_text, x2_text, y2_text = 365, 820, 1600, 1035
    text = read_screen_text(x1_text, y1_text, x2_text, y2_text)

    # Souřadnice pro snímek "hlavy"
    x1_head, y1_head, x2_head, y2_head = 190, 750, 365, 930
    # Přímo pořídíme screenshot, místo čtení z tmp_screenshot.jpg
    head_img = ImageGrab.grab(bbox=(x1_head, y1_head, x2_head, y2_head))
    im = np.array(head_img)

    matched_name = compare_images(im)
    if matched_name:
        matched_name_clean = ''.join([i for i in matched_name if not i.isdigit() and i != '_'])
        print(f"Detected {matched_name_clean} with text: {text}")
        read_out_loud_legacy(f"{matched_name_clean} says: {text}", matched_name_clean)
    else:
        print("No match found.")
        learning_mode(im)