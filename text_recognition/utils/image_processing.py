import os
import cv2
import easyocr
import numpy as np


def load_image(filename: str) -> np.ndarray:
    full_image_path = os.path.join(os.path.dirname(__file__), 'images', filename)
    image = cv2.imread(full_image_path)
    return image


def get_text(image: np.ndarray) -> str:
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image, detail=0)
    return ' '.join(result)
