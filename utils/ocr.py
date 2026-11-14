"""Optional local OCR helpers adapted from the original graduation project.

The plate grammar is a conservative demo filter, not a proof of registration.
The caller controls persistence; OCR results are returned in memory only.
"""

import math
import re

import cv2
import numpy as np

PLATE_CITIES = {
    "01": "Adana",
    "02": "Adıyaman",
    "03": "Afyonkarahisar",
    "04": "Ağrı",
    "05": "Amasya",
    "06": "Ankara",
    "07": "Antalya",
    "08": "Artvin",
    "09": "Aydın",
    "10": "Balıkesir",
    "11": "Bilecik",
    "12": "Bingöl",
    "13": "Bitlis",
    "14": "Bolu",
    "15": "Burdur",
    "16": "Bursa",
    "17": "Çanakkale",
    "18": "Çankırı",
    "19": "Çorum",
    "20": "Denizli",
    "21": "Diyarbakır",
    "22": "Edirne",
    "23": "Elazığ",
    "24": "Erzincan",
    "25": "Erzurum",
    "26": "Eskişehir",
    "27": "Gaziantep",
    "28": "Giresun",
    "29": "Gümüşhane",
    "30": "Hakkari",
    "31": "Hatay",
    "32": "Isparta",
    "33": "Mersin",
    "34": "İstanbul",
    "35": "İzmir",
    "36": "Kars",
    "37": "Kastamonu",
    "38": "Kayseri",
    "39": "Kırklareli",
    "40": "Kırşehir",
    "41": "Kocaeli",
    "42": "Konya",
    "43": "Kütahya",
    "44": "Malatya",
    "45": "Manisa",
    "46": "Kahramanmaraş",
    "47": "Mardin",
    "48": "Muğla",
    "49": "Muş",
    "50": "Nevşehir",
    "51": "Niğde",
    "52": "Ordu",
    "53": "Rize",
    "54": "Sakarya",
    "55": "Samsun",
    "56": "Siirt",
    "57": "Sinop",
    "58": "Sivas",
    "59": "Tekirdağ",
    "60": "Tokat",
    "61": "Trabzon",
    "62": "Tunceli",
    "63": "Şanlıurfa",
    "64": "Uşak",
    "65": "Van",
    "66": "Yozgat",
    "67": "Zonguldak",
    "68": "Aksaray",
    "69": "Bayburt",
    "70": "Karaman",
    "71": "Kırıkkale",
    "72": "Batman",
    "73": "Şırnak",
    "74": "Bartın",
    "75": "Ardahan",
    "76": "Iğdır",
    "77": "Yalova",
    "78": "Karabük",
    "79": "Kilis",
    "80": "Osmaniye",
    "81": "Düzce",
}


def preprocess_plate_image(plate_image):
    """Return a thresholded uint8 plate crop without changing the input."""
    if not isinstance(plate_image, np.ndarray) or not plate_image.size:
        raise ValueError("Plate image must be a nonempty NumPy array.")
    if plate_image.dtype != np.uint8:
        raise ValueError("Plate image must contain uint8 pixels.")
    if plate_image.ndim == 2:
        gray = plate_image
    elif plate_image.ndim == 3 and plate_image.shape[2] == 3:
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Plate image must be grayscale or BGR.")
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresholded = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)


def clean_plate_text(text):
    """Accept a complete two-digit province / letters / digits candidate.

    Whitespace and Turkish letter variants are normalized. Punctuation,
    extra prefixes and trailing fragments are rejected, never truncated.
    """
    if not isinstance(text, str):
        return None
    cleaned = "".join(text.upper().split()).translate(
        str.maketrans({"İ": "I", "Ö": "O", "Ü": "U", "Ğ": "G", "Ş": "S", "Ç": "C"})
    )
    if not re.fullmatch(r"[0-9]{2}[A-Z]{1,3}[0-9]{2,4}", cleaned):
        return None
    return cleaned if cleaned[:2] in PLATE_CITIES else None


def read_license_plate(plate_image):
    """Run Tesseract explicitly; return the best complete OCR line.

    Tesseract must be installed on PATH. No global executable path is set.
    Optional pytesseract is imported only when OCR is requested. Confidence
    is the minimum word confidence of a valid line, on Tesseract's 0..100
    scale; it is not calibrated recognition accuracy.
    """
    processed = preprocess_plate_image(plate_image)
    try:
        import pytesseract
    except ImportError as exc:
        raise RuntimeError("OCR requires pytesseract and the local Tesseract executable.") from exc
    data = pytesseract.image_to_data(
        processed,
        config="--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        output_type=pytesseract.Output.DICT,
    )
    empty = {"plate_text": None, "city": None, "confidence": 0.0}
    texts, confidences = data.get("text", []), data.get("conf", [])
    if len(texts) != len(confidences):
        return empty
    line_fields = ("page_num", "block_num", "par_num", "line_num")
    has_lines = all(len(data.get(name, [])) == len(texts) for name in line_fields)
    # Missing line metadata must not make separate OCR lines look like one.
    if any(name in data for name in line_fields) and not has_lines:
        return empty
    lines = {}
    rejected = set()
    for index, (text, raw_confidence) in enumerate(zip(texts, confidences)):
        if not isinstance(text, str) or not text.strip():
            continue
        key = tuple(data[name][index] for name in line_fields) if has_lines else (0,)
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            rejected.add(key)
            continue
        if not math.isfinite(confidence) or not 0 <= confidence <= 100:
            rejected.add(key)
            continue
        lines.setdefault(key, []).append((text, confidence))
    candidates = []
    for key, words in lines.items():
        if key in rejected:
            continue
        plate = clean_plate_text(" ".join(word for word, _ in words))
        if plate:
            candidates.append((min(conf for _, conf in words), plate))
    if not candidates:
        return empty
    confidence, plate = max(candidates)
    return {"plate_text": plate, "city": PLATE_CITIES[plate[:2]], "confidence": confidence}
