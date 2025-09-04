import base64
import io
import json
import os
import re
import requests
from typing import Optional, Tuple, Dict, Any
import cv2
import numpy as np

from flask import Flask, request, jsonify, render_template
from PIL import Image

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# ---------- PaddleOCR ----------
# pip install paddlepaddle paddleocr==2.7.0.3 flask pillow
from paddleocr import PaddleOCR

# ---------- YOLOv11 for Door Detection ----------
# pip install ultralytics
from ultralytics import YOLO

# Model paths (adjust if your folders differ)
DET_DIR = os.getenv("PPOCR_DET_DIR", "official_models/PP-OCRv5_server_det")
REC_DIR = os.getenv("PPOCR_REC_DIR", "official_models/PP-OCRv5_server_rec")
# Optional models (kept disabled per your snippet)
DOC_ORI_DIR = "official_models/PP-LCNet_x1_0_doc_ori"
TEXTLINE_ORI_DIR = "official_models/PP-LCNet_x1_0_textline_ori"
DOC_UNWARP_DIR = "official_models/UVDoc"

# Door detection model path
DOOR_DETECT_MODEL_PATH = os.getenv("DOOR_DETECT_MODEL_PATH", "official_models/door_detect/door_detect_best.pt")

# External API Configuration
EXTERNAL_API_URL = os.getenv("EXTERNAL_API_URL", "https://api.lardnernorth.in/save_image_captured_data")
LN_API_KEY = os.getenv("LN_API_KEY", "ln-93bT2Zx6LmYvA7qW4pXtE5JDnCzO9fLqQkB1RtYh")
EXTERNAL_API_USERNAME = os.getenv("EXTERNAL_API_USERNAME", "text.image")
EXTERNAL_API_PASSWORD = os.getenv("EXTERNAL_API_PASSWORD", "text.image")

# Initialize PaddleOCR
ocr = PaddleOCR(
    lang="en",
    text_detection_model_dir=DET_DIR,
    text_recognition_model_dir=REC_DIR,
    # doc_orientation_classify_model_dir=DOC_ORI_DIR,
    # doc_unwarping_model_dir=DOC_UNWARP_DIR,
    # textline_orientation_model_dir=TEXTLINE_ORI_DIR,
    use_doc_orientation_classify=False, # Disables document orientation classification model via this parameter
    use_doc_unwarping=False, # Disables text image rectification model via this parameter
    use_textline_orientation=False # Disables text line orientation classification model via this parameter
)

# Initialize Door Detection Model
try:
    door_model = YOLO(DOOR_DETECT_MODEL_PATH)
    print(f"Door detection model loaded successfully from: {DOOR_DETECT_MODEL_PATH}")
except Exception as e:
    print(f"Warning: Could not load door detection model: {e}")
    door_model = None

app = Flask(__name__, template_folder="templates")

# ---------- Frontend Route ----------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# ---------- ISO 6346 check digit ----------

# Letter value mapping per ISO 6346 (skips 11, 22, 33)
LETTER_VALUE = {
    "A": 10, "B": 12, "C": 13, "D": 14, "E": 15, "F": 16, "G": 17, "H": 18, "I": 19,
    "J": 20, "K": 21, "L": 23, "M": 24, "N": 25, "O": 26, "P": 27, "Q": 28, "R": 29,
    "S": 30, "T": 31, "U": 32, "V": 34, "W": 35, "X": 36, "Y": 37, "Z": 38
}

def iso6346_compute_check_digit(owner_equipment_serial: str) -> Optional[int]:
    """
    Compute check digit for a 10-character string: 3-letter owner + 1-letter equipment + 6-digit serial.
    Returns integer 0..9, or None if input malformed.
    """
    s = owner_equipment_serial.strip().upper()
    if not re.fullmatch(r"[A-Z]{4}\d{6}", s):
        return None

    total = 0
    for i, ch in enumerate(s):
        if ch.isalpha():
            v = LETTER_VALUE.get(ch)
            if v is None:
                return None
        else:
            v = int(ch)
        total += v * (2 ** i)
    remainder = total % 11
    return 0 if remainder == 10 else remainder

def iso6346_validate(full_code: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], bool]:
    """
    Given something like 'CRSU1476636' (or split tokens), return:
    (owner_equipment, serial, check_digit, reconstructed_full, is_valid)
    """
    code = re.sub(r"\s+", "", full_code.upper())
    m = re.fullmatch(r"([A-Z]{4})(\d{6})(\d)", code)
    if not m:
        return None, None, None, None, False
    owner_equipment, serial, chk = m.group(1), m.group(2), m.group(3)
    comp = iso6346_compute_check_digit(owner_equipment + serial)
    is_valid = (comp is not None and str(comp) == chk)
    return owner_equipment, serial, chk, owner_equipment + serial + chk, is_valid

# ---------- Parsing helpers ----------

def join_texts(rec_texts) -> str:
    # join with single spaces for regex searches
    return " ".join(t for t in rec_texts if isinstance(t, str)).upper()

def token_list(rec_texts):
    return [t.upper() for t in rec_texts if isinstance(t, str)]

def _clean_int_like(num_str: str) -> Optional[str]:
    """
    For weights like '30.480' or '67,200' -> '30480' / '67200'.
    Returns a string of digits or None.
    """
    if not num_str:
        return None
    digits = re.sub(r"[^\d]", "", num_str)
    return digits if digits else None

def _clean_decimal(num_str: str) -> Optional[str]:
    """
    For CU.M like '33.2' -> '33.2' (keep one decimal dot).
    """
    if not num_str:
        return None
    # Keep first dot, strip commas/spaces
    s = re.sub(r"[ ,]", "", num_str)
    # If multiple dots, keep the first
    parts = s.split(".")
    if len(parts) == 1:
        return parts[0] if parts[0] else None
    return parts[0] + "." + "".join(parts[1:]) if parts[0] or any(parts[1:]) else None

def extract_container_info_improved(rec_texts) -> Dict[str, Any]:
    """
    Improved extraction method that processes rec_texts more systematically.
    Returns a dictionary with all required fields.
    """
    # Convert all texts to uppercase and clean them
    cleaned_texts = []
    for text in rec_texts:
        if isinstance(text, str) and text.strip():
            cleaned_texts.append(text.strip().upper())
    
    # Join all texts for comprehensive searching
    full_text = " ".join(cleaned_texts)
    
    result = {
        "container_number": None,
        "iso_code": None,
        "max_gross_weight": {"kg": None, "lbs": None},
        "tare_weight": {"kg": None, "lbs": None},
        "net_weight": {"kg": None, "lbs": None},
        "cu_cap": {"cubic_meters": None, "cubic_feet": None},
        "verified_status": 0
    }
    
    # 1. Extract Container Number (more flexible patterns)
    container_patterns = [
        r"([A-Z]{4})\s*([0-9]{6})\s*([0-9])",  # ABCD 123456 7
        r"([A-Z]{4})([0-9]{6})([0-9])",        # ABCD1234567
        r"([A-Z]{4})\s*([0-9]{6})",            # ABCD 123456 (missing check digit)
        r"([A-Z]{4})([0-9]{6})"                # ABCD123456 (missing check digit)
    ]
    
    for pattern in container_patterns:
        match = re.search(pattern, full_text)
        if match and match.group(1)[-1] in {'U', 'J', 'Z'}:  # Valid equipment type
            owner_equipment = match.group(1)
            serial = match.group(2)
            check_digit = match.group(3) if len(match.groups()) >= 3 else None
            
            if check_digit:
                # Validate existing check digit
                computed_check = iso6346_compute_check_digit(owner_equipment + serial)
                if computed_check is not None and str(computed_check) == check_digit:
                    result["container_number"] = owner_equipment + serial + check_digit
                    result["verified_status"] = 1
                    break
            else:
                # Compute missing check digit
                computed_check = iso6346_compute_check_digit(owner_equipment + serial)
                if computed_check is not None:
                    result["container_number"] = owner_equipment + serial + str(computed_check)
                    result["verified_status"] = 1
                    break
    
    # 2. Extract ISO Code (more flexible)
    iso_patterns = [
        r"\b([0-9]{2}[A-Z][0-9])\b",  # Standard format like 22G1
        r"([0-9]{2})\s*([A-Z])\s*([0-9])"  # Separated format like 22 G 1
    ]
    
    for pattern in iso_patterns:
        match = re.search(pattern, full_text)
        if match:
            if len(match.groups()) == 1:
                result["iso_code"] = match.group(1)
            else:
                result["iso_code"] = match.group(1) + match.group(2) + match.group(3)
            break
    
    # 3. Extract Weights (improved patterns with more flexibility)
    # Max Gross Weight - handle various formats including split tokens
    max_gross_patterns = [
        r"MAX\.?\s*GROSS\s*([0-9]+[.,]?[0-9]*)\s*KGS?\.?\s*([0-9]+[.,]?[0-9]*)\s*LBS?\.?",
        r"MAX\.?\s*([0-9]+[.,]?[0-9]*)\s*KGS?\.?\s*GROSS\s*([0-9]+[.,]?[0-9]*)\s*LBS?\.?",
        r"GROSS\s*([0-9]+[.,]?[0-9]*)\s*KGS?\.?\s*([0-9]+[.,]?[0-9]*)\s*LBS?\.?",
        r"([0-9]+[.,]?[0-9]*)\s*KGS?\.?\s*GROSS\s*([0-9]+[.,]?[0-9]*)\s*LBS?\.?",
        r"MAX\.?\s*([0-9]+[.,]?[0-9]*)\s*KG.*?([0-9]+[.,]?[0-9]*)\s*LB"
    ]
    
    for pattern in max_gross_patterns:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            result["max_gross_weight"]["kg"] = _clean_int_like(match.group(1))
            result["max_gross_weight"]["lbs"] = _clean_int_like(match.group(2))
            break
    
    # Tare Weight - more flexible patterns
    tare_patterns = [
        r"TARE\s*(?:WGT\.?)?\s*([0-9]+[.,]?[0-9]*)\s*KGS?\.?\s*([0-9]+[.,]?[0-9]*)\s*LBS?\.?",
        r"TARE\s*([0-9]+[.,]?[0-9]*)\s*KGS?\.?\s*WGT\.?\s*([0-9]+[.,]?[0-9]*)\s*LBS?\.?",
        r"([0-9]+[.,]?[0-9]*)\s*KGS?\.?\s*TARE.*?([0-9]+[.,]?[0-9]*)\s*LBS?\.?",
        r"TARE\s*([0-9]+[.,]?[0-9]*)\s*KG.*?([0-9]+[.,]?[0-9]*)\s*LB"
    ]
    
    for pattern in tare_patterns:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            result["tare_weight"]["kg"] = _clean_int_like(match.group(1))
            result["tare_weight"]["lbs"] = _clean_int_like(match.group(2))
            break
    
    # Net/Cargo Weight - enhanced patterns
    net_patterns = [
        r"MAX\.?\s*CARGO\s*([0-9]+[.,]?[0-9]*)\s*KGS?\.?\s*([0-9]+[.,]?[0-9]*)\s*LBS?\.?",
        r"CARGO\s*([0-9]+[.,]?[0-9]*)\s*KGS?\.?\s*([0-9]+[.,]?[0-9]*)\s*LBS?\.?",
        r"NET\s*([0-9]+[.,]?[0-9]*)\s*KGS?\.?\s*([0-9]+[.,]?[0-9]*)\s*LBS?\.?",
        r"([0-9]+[.,]?[0-9]*)\s*KGS?\.?\s*CARGO.*?([0-9]+[.,]?[0-9]*)\s*LBS?\.?",
        r"MAX\.?\s*CARGO\s*([0-9]+[.,]?[0-9]*)\s*KG.*?([0-9]+[.,]?[0-9]*)\s*LB"
    ]
    
    for pattern in net_patterns:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            result["net_weight"]["kg"] = _clean_int_like(match.group(1))
            result["net_weight"]["lbs"] = _clean_int_like(match.group(2))
            break
    
    # 4. Extract Cubic Capacity
    cu_patterns = [
        r"CU\.?\s*CAP\.?\s*([0-9]+[.,]?[0-9]*)\s*(?:CU\.?M\.?|M3).*?([0-9]+[.,]?[0-9]*)\s*(?:CU\.?FT\.?|FT3)",
        r"([0-9]+[.,]?[0-9]*)\s*(?:CU\.?M\.?|M3).*?([0-9]+[.,]?[0-9]*)\s*(?:CU\.?FT\.?|FT3)",
        r"([0-9]+[.,]?[0-9]*)\s*(?:CU\.?FT\.?|FT3).*?([0-9]+[.,]?[0-9]*)\s*(?:CU\.?M\.?|M3)"
    ]
    
    for pattern in cu_patterns:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            if "FT" in pattern and "M" in pattern:
                if "FT.*M" in pattern:  # FT first, then M
                    result["cu_cap"]["cubic_feet"] = _clean_int_like(match.group(1))
                    result["cu_cap"]["cubic_meters"] = _clean_decimal(match.group(2))
                else:  # M first, then FT
                    result["cu_cap"]["cubic_meters"] = _clean_decimal(match.group(1))
                    result["cu_cap"]["cubic_feet"] = _clean_int_like(match.group(2))
            break
    
    # 5. Fallback: Try to extract individual numbers and match common patterns
    if not any([result["max_gross_weight"]["kg"], result["tare_weight"]["kg"], result["cu_cap"]["cubic_meters"]]):
        # Extract all numbers and try to assign them based on typical ranges
        numbers = re.findall(r'[0-9]+[.,]?[0-9]*', full_text)
        numbers = [n.replace(',', '') for n in numbers if len(n.replace('.', '').replace(',', '')) >= 2]
        
        for num in numbers:
            val = float(num.replace(',', '')) if '.' in num else int(num.replace(',', ''))
            
            # Typical weight ranges (in kg)
            if 25000 <= val <= 35000 and not result["max_gross_weight"]["kg"]:  # Max gross weight
                result["max_gross_weight"]["kg"] = str(int(val))
                result["max_gross_weight"]["lbs"] = str(int(val * 2.2))
            elif 2000 <= val <= 5000 and not result["tare_weight"]["kg"]:  # Tare weight
                result["tare_weight"]["kg"] = str(int(val))
                result["tare_weight"]["lbs"] = str(int(val * 2.2))
            elif 20 <= val <= 80 and '.' in num and not result["cu_cap"]["cubic_meters"]:  # Cubic meters
                result["cu_cap"]["cubic_meters"] = num
                result["cu_cap"]["cubic_feet"] = str(int(val * 35.3))
    
    # Enhanced fallback for the specific OCR format we saw
    # Handle cases like: ['30.480KGS.', '67.200LBS.', '2.250KGS.', '4.960LBS.', '28.230KGS.', '62.240LBS.']
    weight_tokens = []
    for text in cleaned_texts:
        if re.search(r'[0-9]+\.[0-9]+KGS?\.?', text, re.IGNORECASE):
            kg_match = re.search(r'([0-9]+\.[0-9]+)', text)
            if kg_match:
                weight_tokens.append(('kg', kg_match.group(1)))
        elif re.search(r'[0-9]+\.[0-9]+LBS?\.?', text, re.IGNORECASE):
            lbs_match = re.search(r'([0-9]+\.[0-9]+)', text)
            if lbs_match:
                weight_tokens.append(('lbs', lbs_match.group(1)))
    
    # Try to assign weights based on typical values
    for unit, value in weight_tokens:
        val = float(value.replace(',', ''))
        if unit == 'kg':
            if 25000 <= val <= 35000 and not result["max_gross_weight"]["kg"]:
                result["max_gross_weight"]["kg"] = _clean_int_like(value)
            elif 2000 <= val <= 5000 and not result["tare_weight"]["kg"]:
                result["tare_weight"]["kg"] = _clean_int_like(value)
            elif 20000 <= val <= 35000 and not result["net_weight"]["kg"]:
                result["net_weight"]["kg"] = _clean_int_like(value)
        elif unit == 'lbs':
            if 55000 <= val <= 75000 and not result["max_gross_weight"]["lbs"]:
                result["max_gross_weight"]["lbs"] = _clean_int_like(value)
            elif 4000 <= val <= 11000 and not result["tare_weight"]["lbs"]:
                result["tare_weight"]["lbs"] = _clean_int_like(value)
            elif 50000 <= val <= 70000 and not result["net_weight"]["lbs"]:
                result["net_weight"]["lbs"] = _clean_int_like(value)
    
    # Calculate net weight ONLY if not detected, and ensure it's not negative
    if (not result["net_weight"]["kg"] and 
        result["max_gross_weight"]["kg"] and 
        result["tare_weight"]["kg"]):
        try:
            gross_kg = int(result["max_gross_weight"]["kg"])
            tare_kg = int(result["tare_weight"]["kg"])
            net_kg = gross_kg - tare_kg
            if net_kg > 0:
                result["net_weight"]["kg"] = str(net_kg)
                result["net_weight"]["lbs"] = str(int(net_kg * 2.2))
            else:
                result["net_weight"]["kg"] = None
                result["net_weight"]["lbs"] = None
        except:
            result["net_weight"]["kg"] = None
            result["net_weight"]["lbs"] = None
    
    return result

def extract_container_core(tokens) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], bool]:
    """
    Try to find 4-letter code (ending with U/J/Z), 6-digit serial, optional 1-digit check.
    Return owner_equipment, serial, check_digit, full_code, is_valid
    """
    owner_equipment = None
    serial = None
    check_digit = None

    # Find a 4-letter token that ends in U/J/Z (equipment category)
    for i, t in enumerate(tokens):
        if re.fullmatch(r"[A-Z]{4}", t) and t[-1] in {"U", "J", "Z"}:
            owner_equipment = t
            # Serial is usually the next token with 6 digits
            if i + 1 < len(tokens) and re.fullmatch(r"\d{6}", re.sub(r"[^\d]", "", tokens[i + 1])):
                serial = re.sub(r"[^\d]", "", tokens[i + 1])[:6]
                # Optional check digit might follow
                if i + 2 < len(tokens) and re.fullmatch(r"\d", re.sub(r"[^\d]", "", tokens[i + 2])):
                    check_digit = re.sub(r"[^\d]", "", tokens[i + 2])[:1]
            break

    if owner_equipment and serial:
        if check_digit:
            _, _, _, full_code, ok = iso6346_validate(owner_equipment + serial + check_digit)
            return owner_equipment, serial, check_digit, full_code, ok
        else:
            comp = iso6346_compute_check_digit(owner_equipment + serial)
            if comp is not None:
                return owner_equipment, serial, str(comp), owner_equipment + serial + str(comp), True
    return owner_equipment, serial, check_digit, None, False

def extract_iso_code(tokens) -> Optional[str]:
    # ISO size/type: e.g., 22G1, 45R1, etc.
    for t in tokens:
        if re.fullmatch(r"\d{2}[A-Z]\d", t):
            return t
    return None

def extract_weights_and_cu(text: str) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Robust regex extraction tolerant to punctuation/spacing and OCR errors:
      MAX. GROSS 30.480KGS. 67.200LBS.
      MAX 30.488KGS GHOSS 072001ES (OCR error case)
      TARE 2.250KGS. 4.960LBS.
      TARE WGT. 2230KGS 4.920LBS
      MAX. CARGO 28.230KGS. 62.240LBS.
      CU. CAP. 33.2 CU.M. 1.170 CU.FT.
    Returns dict with kg/lbs strings (digits only) or None.
    """
    out = {
        "max_gross_weight": {"kg": None, "lbs": None},
        "tare_weight": {"kg": None, "lbs": None},
        "net_weight": {"kg": None, "lbs": None},
        "cu_cap": {"cubic_meters": None, "cubic_feet": None},
    }

    # MAX GROSS - handle both normal and OCR error cases
    mg = re.search(r"MAX\.?\s*(?:GROSS|GHOSS)\s*([0-9][\d.,]*)\s*KGS?\.?\s*([0-9][\d.,]*)\s*LBS?\.?", text, re.IGNORECASE)
    if not mg:
        # Alternative pattern: MAX 30.488KGS GHOSS (sometimes LBS value is corrupted)
        mg_kg = re.search(r"MAX\s+([0-9][\d.,]*)\s*KGS", text, re.IGNORECASE)
        if mg_kg:
            out["max_gross_weight"]["kg"] = _clean_int_like(mg_kg.group(1))
            # Try to find a reasonable LBS value nearby (typically 2.2x the kg value)
            kg_val = _clean_int_like(mg_kg.group(1))
            if kg_val and kg_val.isdigit():
                estimated_lbs = str(int(int(kg_val) * 2.2))
                out["max_gross_weight"]["lbs"] = estimated_lbs
    else:
        out["max_gross_weight"]["kg"] = _clean_int_like(mg.group(1))
        out["max_gross_weight"]["lbs"] = _clean_int_like(mg.group(2))

    # TARE (optionally "WGT.") - handle both formats
    tr = re.search(r"TARE(?:\s*WGT\.?)?\s*([0-9][\d.,]*)\s*KGS?\.?\s*([0-9][\d.,]*)\s*LBS?\.?", text, re.IGNORECASE)
    if tr:
        out["tare_weight"]["kg"] = _clean_int_like(tr.group(1))
        out["tare_weight"]["lbs"] = _clean_int_like(tr.group(2))

    # MAX CARGO / NET
    nw = re.search(r"(?:MAX\.?\s*CARGO|NET)\s*([0-9][\d.,]*)\s*KGS?\.?\s*([0-9][\d.,]*)\s*LBS?\.?", text, re.IGNORECASE)
    if nw:
        out["net_weight"]["kg"] = _clean_int_like(nw.group(1))
        out["net_weight"]["lbs"] = _clean_int_like(nw.group(2))

    # CU CAP - handle variations like "1170CUFT 33.2CUM CAP"
    cu = re.search(r"(?:CU\.?\s*CAP\.?|CAP\.?)\s*([0-9][\d.,]*)\s*(?:CU\.?M\.?|CUM).*?([0-9][\d.,]*)\s*(?:CU\.?F?T?\.?|CUFT)", text, re.IGNORECASE)
    if not cu:
        # Alternative pattern: 1170CUFT 33.2CUM CAP
        cu_alt = re.search(r"([0-9][\d.,]*)\s*(?:CU\.?F?T?\.?|CUFT)\s*([0-9][\d.,]*)\s*(?:CU\.?M\.?|CUM)", text, re.IGNORECASE)
        if cu_alt:
            out["cu_cap"]["cubic_feet"] = _clean_int_like(cu_alt.group(1))
            out["cu_cap"]["cubic_meters"] = _clean_decimal(cu_alt.group(2))
    else:
        out["cu_cap"]["cubic_meters"] = _clean_decimal(cu.group(1))
        out["cu_cap"]["cubic_feet"] = _clean_int_like(cu.group(2))

    return out

def image_to_base64(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode("ascii")

def load_image_from_request() -> Tuple[Optional[bytes], Optional[str]]:
    """
    Returns (raw_bytes, error_message)
    """
    if "image" in request.files:
        f = request.files["image"]
        raw = f.read()
        if not raw:
            return None, "Empty image file."
        return raw, None
    data = request.get_json(silent=True) or {}
    if "image_b64" in data:
        try:
            raw = base64.b64decode(data["image_b64"], validate=True)
            return raw, None
        except Exception:
            return None, "Invalid base64 in 'image_b64'."
    return None, "No image provided. Use form-data field 'image' or JSON key 'image_b64'."

# ---------- Door Detection Functions ----------

def detect_door_end(image_array: np.ndarray, confidence_threshold: float = 0.5) -> Tuple[bool, float, Dict]:
    """
    Detect if the input image contains a container door end.
    
    Args:
        image_array: RGB image array from PIL/numpy
        confidence_threshold: Minimum confidence for door detection
    
    Returns:
        Tuple of (is_door_detected, max_confidence, detection_details)
    """
    if door_model is None:
        return False, 0.0, {"error": "Door detection model not loaded"}
    
    try:
        # Convert RGB to grayscale
        if len(image_array.shape) == 3:
            gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image_array
        
        # Convert back to 3-channel for YOLO (expects 3 channels)
        gray_rgb = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
        
        # Run inference
        results = door_model(gray_rgb, verbose=False)
        
        # Parse results
        detection_details = {
            "detections": [],
            "total_detections": 0,
            "image_shape": gray_rgb.shape
        }
        
        max_confidence = 0.0
        door_detected = False
        
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    detection_info = {
                        "confidence": confidence,
                        "class_id": class_id,
                        "bbox": [x1, y1, x2, y2],
                        "area": (x2 - x1) * (y2 - y1)
                    }
                    
                    detection_details["detections"].append(detection_info)
                    
                    if confidence > max_confidence:
                        max_confidence = confidence
                    
                    if confidence >= confidence_threshold:
                        door_detected = True
        
        detection_details["total_detections"] = len(detection_details["detections"])
        detection_details["max_confidence"] = max_confidence
        detection_details["door_detected"] = door_detected
        
        return door_detected, max_confidence, detection_details
        
    except Exception as e:
        return False, 0.0, {"error": f"Door detection failed: {str(e)}"}

def convert_to_grayscale(image_array: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to grayscale.
    """
    if len(image_array.shape) == 3:
        return cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    return image_array


# ---------- API ----------
@app.route("/update_extracted", methods=["POST"])
def update_extracted():
    """
    Accepts updated container info from frontend, returns updated values for display.
    """
    data = request.get_json(force=True)
    # Extract all expected fields
    movement = data.get("movement", "DEPOT_IN")
    location = data.get("location", "INHAL")
    container_number = data.get("container_number")
    iso_code = data.get("iso_code")
    max_gross_weight = data.get("max_gross_weight", {"kg": None, "lbs": None})
    tare_weight = data.get("tare_weight", {"kg": None, "lbs": None})
    net_weight = data.get("net_weight", {"kg": None, "lbs": None})
    cu_cap = data.get("cu_cap", {"cubic_meters": None, "cubic_feet": None})
    verified_status = data.get("verified_status", 0)
    door_detection = data.get("door_detection", {"door_detected": False, "max_confidence": 0.0, "details": {}})
    original_data = data.get("original_data", {})

    # Optionally, you can validate/correct the values here
    # For now, just echo back the updated values
    payload = {
        "movement": movement,
        "location": location,
        "container_number": container_number,
        "iso_code": iso_code,
        "max_gross_weight": max_gross_weight,
        "tare_weight": tare_weight,
        "net_weight": net_weight,
        "verified_status": verified_status,
        "original_data": original_data,
        "cu_cap": cu_cap
    }
    
    print("=== UPDATE_EXTRACTED API Response ===")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("====================================")
    
    return jsonify(payload), 200

@app.route("/save_correct_data", methods=["POST"])
def save_correct_data():
    """
    Endpoint to save correct data to external API when "Correct" button is clicked.
    """
    try:
        # Get the raw request data for debugging
        raw_data = request.get_data()
        content_type = request.content_type
        
        print("=== RAW REQUEST DEBUG ===")
        print(f"Content-Type: {content_type}")
        print(f"Raw data: {raw_data}")
        print("========================")
        
        data = request.get_json(force=True)
        
        print("=== RECEIVED DATA ===")
        print(f"Data type: {type(data)}")
        print(f"Data: {json.dumps(data, indent=2, ensure_ascii=False)}")
        print("====================")
        
        # Validate that we have data
        if not data:
            return jsonify({
                "success": False,
                "message": "No data received"
            }), 400
        
        # The data will be sent as form-encoded with 'transaction' field containing JSON
        
        # Prepare headers for external API
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json',
            'LN-API-Key': LN_API_KEY
        }
        
        # Prepare basic authentication
        auth = (EXTERNAL_API_USERNAME, EXTERNAL_API_PASSWORD)
        
        # Make the POST request to external API with basic auth
        # Convert JSON payload to form-encoded data
        form_data = {
            'transaction': json.dumps(data)  # Send the transaction data as JSON string in form field
        }
        
        # Log the data being sent
        print("=== SENDING TO EXTERNAL API ===")
        print(f"URL: {EXTERNAL_API_URL}")
        print(f"Headers: {headers}")
        print(f"Auth: {EXTERNAL_API_USERNAME}:{'*' * len(EXTERNAL_API_PASSWORD)}")
        print(f"Form Data: {form_data}")
        print(f"Transaction JSON: {json.dumps(data, indent=2, ensure_ascii=False)}")
        print("==============================")
        
        response = requests.post(
            EXTERNAL_API_URL,
            data=form_data,
            headers=headers,
            auth=auth,
            timeout=30  # 30 second timeout
        )
        
        # Log the response
        print("=== EXTERNAL API RESPONSE ===")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        print("============================")
        
        # Check if the request was successful
        if response.status_code == 200:
            return jsonify({
                "success": True,
                "message": "Data saved successfully to external API",
                "external_response": response.json() if response.content else {}
            }), 200
        else:
            return jsonify({
                "success": False,
                "message": f"External API returned status code {response.status_code}",
                "external_response": response.text
            }), 500
            
    except requests.exceptions.Timeout:
        return jsonify({
            "success": False,
            "message": "Request to external API timed out"
        }), 500
    except requests.exceptions.RequestException as e:
        return jsonify({
            "success": False,
            "message": f"Error connecting to external API: {str(e)}"
        }), 500
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Internal server error: {str(e)}"
        }), 500

@app.route("/detect_door", methods=["POST"])
def detect_door():
    """
    Endpoint to detect if an image contains a container door end.
    Returns detection results without performing OCR.
    """
    raw, err = load_image_from_request()
    if err:
        return jsonify({"error": err}), 400

    # Validate image decodes
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img_array = np.array(img)
    except Exception:
        return jsonify({"error": "Unable to decode image."}), 400

    # Run door detection
    try:
        door_detected, max_confidence, detection_details = detect_door_end(img_array)
        
        response = {
            "door_detected": door_detected,
            "max_confidence": max_confidence,
            "detection_details": detection_details,
            "base64_image": image_to_base64(raw),
            "message": "Door detection completed successfully"
        }
        
        print("=== DETECT_DOOR API Response ===")
        print(json.dumps(response, indent=2, ensure_ascii=False))
        print("===============================")
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({"error": f"Door detection failed: {e}"}), 500

@app.route("/extract", methods=["POST"])
def extract():
    raw, err = load_image_from_request()
    if err:
        return jsonify({"error": err}), 400

    # Get movement and location from form data or JSON
    movement = request.form.get('movement', 'DEPOT_IN')  # Default to DEPOT_IN
    location = request.form.get('location', 'INHAL')     # Default to INHAL
    
    # If JSON request, get from JSON data
    if request.content_type and 'application/json' in request.content_type:
        json_data = request.get_json() or {}
        movement = json_data.get('movement', movement)
        location = json_data.get('location', location)

    # Validate image decodes
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img_array = np.array(img)
    except Exception:
        return jsonify({"error": "Unable to decode image."}), 400

    # Validate image decodes
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img_array = np.array(img)
    except Exception:
        return jsonify({"error": "Unable to decode image."}), 400

    # First, check if this is a door end image
    try:
        door_detected, max_confidence, detection_details = detect_door_end(img_array)
        
        if not door_detected:
            return jsonify({
                "error": "No container door end detected in the image",
                "door_detection": {
                    "door_detected": False,
                    "max_confidence": max_confidence,
                    "details": detection_details
                },
                "base64_image": image_to_base64(raw)
            }), 400
            
    except Exception as e:
        # If door detection fails, log warning but continue with OCR
        print(f"Warning: Door detection failed: {e}")
        door_detected = True  # Assume it's valid to continue
        max_confidence = 0.0
        detection_details = {"error": str(e)}

    # Run OCR on the original color image
    try:
        # PaddleOCR can take path or numpy; easiest is save to bytes->numpy via PIL
        # but PaddleOCR also accepts path-like; we pass ndarray via np.array(img)
        results = ocr.predict(img_array)
    except Exception as e:
        return jsonify({"error": f"OCR failed: {e}"}), 500

    if not results or not isinstance(results, list):
        return jsonify({"error": "No OCR results."}), 500

    res = results[0]
    rec_texts = res.get("rec_texts", []) or []
    
    # Use the improved extraction method
    extracted_info = extract_container_info_improved(rec_texts)

    # Build response in requested format
    payload = {
        "movement": movement,
        "location": location,
        "base64_image": image_to_base64(raw),
        "container_number": extracted_info["container_number"],
        "iso_code": extracted_info["iso_code"],
        "max_gross_weight": extracted_info["max_gross_weight"],
        "tare_weight": extracted_info["tare_weight"],
        "net_weight": extracted_info["net_weight"],
        "verified_status": extracted_info["verified_status"],
        "original_data": {},  # Empty object for initial extraction
        "cu_cap": extracted_info["cu_cap"],
        "door_detection": {
            "door_detected": door_detected,
            "max_confidence": max_confidence,
            "details": detection_details
        }
    }

    print("=== EXTRACT API Response ===")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("============================")

    return jsonify(payload), 200

@app.route("/debug_ocr", methods=["POST"])
def debug_ocr():
    """Debug endpoint to see raw OCR output and extraction process"""
    raw, err = load_image_from_request()
    if err:
        return jsonify({"error": err}), 400

    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img_array = np.array(img)
    except Exception:
        return jsonify({"error": "Unable to decode image."}), 400

    # Run door detection first
    try:
        door_detected, max_confidence, detection_details = detect_door_end(img_array)
    except Exception as e:
        door_detected = False
        max_confidence = 0.0
        detection_details = {"error": f"Door detection failed: {e}"}

    try:
        results = ocr.predict(img_array)
    except Exception as e:
        return jsonify({"error": f"OCR failed: {e}"}), 500

    if not results or not isinstance(results, list):
        return jsonify({"error": "No OCR results."}), 500

    res = results[0]
    rec_texts = res.get("rec_texts", []) or []
    
    # Show what we extracted
    extracted_info = extract_container_info_improved(rec_texts)
    
    debug_info = {
        "door_detection": {
            "door_detected": door_detected,
            "max_confidence": max_confidence,
            "details": detection_details
        },
        "raw_rec_texts": rec_texts,
        "combined_text": " ".join(rec_texts),
        "extracted_info": extracted_info,
        "total_texts_found": len(rec_texts)
    }
    
    print("=== DEBUG_OCR API Response ===")
    print(json.dumps(debug_info, indent=2, ensure_ascii=False))
    print("=============================")
    
    return jsonify(debug_info), 200

# ---------- Local dev entry ----------
if __name__ == "__main__":
    # Test function with sample OCR texts
    def test_extraction():
        # Sample OCR texts that might come from container images
        sample_texts = [
            "CRSU", "1476636", "MAX", "GROSS", "30480", "KGS", "67200", "LBS",
            "TARE", "WGT", "2200", "KGS", "4850", "LBS", 
            "22G1", "CU", "CAP", "33.2", "CUM", "1172", "CUFT"
        ]
        
        print("Testing extraction with sample texts:")
        print("Input texts:", sample_texts)
        result = extract_container_info_improved(sample_texts)
        print("Extracted result:", result)
        print()
        
        # Test with another common format
        sample_texts2 = [
            "ABCU1234567", "45R1", "MAX GROSS 30480 KGS 67200 LBS",
            "TARE 2200 KGS 4850 LBS", "MAX CARGO 28280 KGS 62350 LBS",
            "CU CAP 33.2 CUM 1172 CUFT"
        ]
        
        print("Testing extraction with concatenated format:")
        print("Input texts:", sample_texts2)
        result2 = extract_container_info_improved(sample_texts2)
        print("Extracted result:", result2)
        print()
    
    # Run test if called directly
    if len(os.sys.argv) > 1 and os.sys.argv[1] == "--test":
        test_extraction()
    else:
        # Optional: make port configurable
        port = int(os.getenv("PORT", "8080"))
        app.run(host="0.0.0.0", port=port, debug=True)
