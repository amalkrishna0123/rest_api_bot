# ocr_space.py - Enhanced version with multi-page PDF support
import io
import os
import requests
from PIL import Image, ImageOps, ImageEnhance

# OCR.space credentials
OCR_API_KEY = "K82681714188957"
OCR_ENDPOINT = "https://api.ocr.space/parse/image"

def _pre_process(img: Image.Image, max_dim=2048) -> bytes:
    """
    Light, fast in-memory prep: orient, contrast, shrink.
    Returns JPEG bytes ready for upload.
    """
    img = ImageOps.exif_transpose(img)                 # auto-rotate via EXIF
    img = ImageEnhance.Contrast(img).enhance(1.15)     # bump contrast
    img.thumbnail((max_dim, max_dim), Image.LANCZOS)   # keep under 5 MB limit

    out = io.BytesIO()
    img.save(out, format="JPEG", quality=90)
    return out.getvalue()

def ocr_space_file(file_obj, is_pdf=False, language="eng"):
    """
    Call OCR.space Parse API.

    :param file_obj: Django InMemoryUploadedFile or file-like object
    :param is_pdf: bool
    :param language: OCR.space language string (default "eng")
    :return: (parsed_text: str, exit_code: int, error: str)
    """
    try:
        if is_pdf:
            # Reset file pointer for PDF
            file_obj.seek(0)
            files = {"file": (file_obj.name, file_obj.read(), "application/pdf")}
        else:
            # Reset file pointer for image
            file_obj.seek(0)
            img_bytes = _pre_process(Image.open(file_obj))
            files = {"file": (file_obj.name, img_bytes, "image/jpeg")}

        data = {
            "apikey": OCR_API_KEY,
            "language": language,
            "isTable": "true",
            "OCREngine": "2",
            "scale": "true",
            "isCreateSearchablePdf": "false",
            "isSearchablePdfHideTextLayer": "false",
        }

        resp = requests.post(OCR_ENDPOINT, files=files, data=data, timeout=90)  # Increased timeout for slower API responses
        
        if resp.status_code != 200:
            return "", 2, f"HTTP {resp.status_code}"

        j = resp.json()
        
        if j.get("IsErroredOnProcessing"):
            return "", 4, j.get("ErrorMessage", "Unknown OCR.space error")

        parsed_results = j.get("ParsedResults", [])
        if not parsed_results:
            return "", 5, "No OCR results returned"
            
        parsed_text = parsed_results[0].get("ParsedText", "")
        return parsed_text.strip(), 0, ""
        
    except requests.RequestException as exc:
        return "", 1, f"Network error: {exc}"
    except Exception as exc:
        return "", 6, f"Processing error: {exc}"



def _has_mrz(txt: str) -> bool:
    t = (txt or "").upper()
    # MRZ has lots of '<' and patterns like P<, two-line dense blocks
    return ('<<' in t or 'P<' in t) and len(t.replace('\n','')) > 80



def ocr_space_file_multi_lang(file_obj, is_pdf=False):
    """
    Try OCR with a broad language sweep.
    Stop early if we get substantial text or detect MRZ.
    Fallback to Tesseract as a last resort.
    """
    # If it’s a PDF, your existing helper already collects all pages
    if is_pdf:
        txt, code, err = ocr_space_pdf_all_pages(file_obj)
        if code == 0 and (len(txt.strip()) > 40 or _has_mrz(txt)):
            return txt, code, err
        # No point multi-cycling PDFs again; we’ll tesseract-fallback below.

    # For images, cycle languages. Start with eng, then high-frequency sets.
    lang_candidates = [
        "eng",                     # English, numbers
        "eng+fra+deu+spa+ita+por", # Western EU block
        "eng+rus+ukr",             # Cyrillic
        "eng+tur+pol+nld+swe+nor+dan+cze+slk",  # misc EU
        "eng+ara+fas+urd",         # Arabic family
        "eng+hin+ben+tam+tel+kan+mal+mar+guj+pan+ori",  # India region
        "eng+tha+vie+ind+msa",     # SE Asia
        "eng+chi_sim+jpn+kor",     # East Asia
    ]

    # Try each language pack until we get enough text or MRZ
    best_text, best_code, best_err = "", 5, "No OCR results returned"
    for lang in lang_candidates:
        file_obj.seek(0)
        txt, code, err = ocr_space_file(file_obj, False, lang)
        if code == 0 and len((txt or "").strip()) > 40:
            return txt, code, err
        if code == 0 and _has_mrz(txt):
            return txt, code, err
        if code == 0 and len(txt.strip()) > len(best_text.strip()):
            best_text, best_code, best_err = txt, code, err

    # If PDF, we likely already tried; if image failed above, do a local fallback
    try:
        from PIL import Image
        import pytesseract
        file_obj.seek(0)
        img = Image.open(file_obj)
        # broad multilingual fallback. Adjust packs you have installed.
        txt = pytesseract.image_to_string(img, lang="eng")
        if len(txt.strip()) > len(best_text.strip()):
            best_text, best_code, best_err = txt, 0, ""
    except Exception as e:
        if not best_text:
            return "", 6, f"Tesseract fallback failed: {e}"

    return best_text.strip(), best_code, best_err



def ocr_space_pdf_all_pages(file_obj):
    """
    Process all pages of a PDF file using OCR.space
    Returns combined text from all pages.
    """
    try:
        # Reset file pointer
        file_obj.seek(0)
        pdf_content = file_obj.read()
        
        combined_text = ""
        
        # OCR.space automatically processes all pages in PDF files
        # We'll use a single API call as OCR.space handles multi-page PDFs
        files = {"file": (file_obj.name, pdf_content, "application/pdf")}
        
        data = {
            "apikey": OCR_API_KEY,
            "language": "eng",  # Start with English
            "isTable": "true",
            "OCREngine": "2",
            "scale": "true",
            "isCreateSearchablePdf": "false",
            "isSearchablePdfHideTextLayer": "false",
        }

        # First try with English
        resp = requests.post(OCR_ENDPOINT, files=files, data=data, timeout=90)
        
        if resp.status_code != 200:
            return "", 2, f"HTTP {resp.status_code}"

        j = resp.json()
        
        if j.get("IsErroredOnProcessing"):
            return "", 4, j.get("ErrorMessage", "Unknown OCR.space error")

        parsed_results = j.get("ParsedResults", [])
        if not parsed_results:
            return "", 5, "No OCR results returned"
        
        # Combine text from all pages
        for result in parsed_results:
            page_text = result.get("ParsedText", "")
            if page_text.strip():
                combined_text += page_text + "\n\n"
        
        # Check if we got substantial text with English
        if len(combined_text.strip()) > 10:
            return combined_text.strip(), 0, ""
        
        # If English didn't yield good results, try Arabic
        # Reset file pointer and try Arabic
        file_obj.seek(0)
        files = {"file": (file_obj.name, pdf_content, "application/pdf")}
        data["language"] = "ara"
        
        resp = requests.post(OCR_ENDPOINT, files=files, data=data, timeout=90)
        
        if resp.status_code != 200:
            # Return English result even if it's minimal
            return combined_text.strip(), 0, f"Arabic failed but returning English text"
        
        j = resp.json()
        
        if j.get("IsErroredOnProcessing"):
            return combined_text.strip(), 0, f"Arabic failed but returning English text"
        
        parsed_results = j.get("ParsedResults", [])
        if not parsed_results:
            return combined_text.strip(), 0, f"No Arabic results but returning English text"
        
        # Combine Arabic results
        arabic_text = ""
        for result in parsed_results:
            page_text = result.get("ParsedText", "")
            if page_text.strip():
                arabic_text += page_text + "\n\n"
        
        return arabic_text.strip(), 0, ""
        
    except requests.RequestException as exc:
        return "", 1, f"Network error: {exc}"
    except Exception as exc:
        return "", 6, f"Processing error: {exc}"

def detect_document_side(text):
    """
    Detect if text represents front or back side of Emirates ID.
    Returns 'front', 'back', or 'unknown'
    """
    text_lower = text.lower()
    
    # Back side indicators
    back_indicators = [
        "الكفالة", "employer", "صاحب العمل", "occupation", "المهنة",
        "occupation", "employer", "issuing place", "مكان الإصدار",
        "family sponsor", "كفالة عائلية"
    ]
    
    # Front side indicators  
    front_indicators = [
        "emirates id", "identity card", "بطاقة الهوية", "name", "الاسم",
        "nationality", "الجنسية", "date of birth", "تاريخ الميلاد"
    ]
    
    back_score = sum(1 for indicator in back_indicators if indicator in text_lower)
    front_score = sum(1 for indicator in front_indicators if indicator in text_lower)
    
    if back_score > front_score:
        return "back"
    elif front_score > back_score:
        return "front"
    else:
        return "unknown"


def detect_passport_side(text):
    """
    Detect if text represents front or back side of passport.
    Returns 'front', 'back', or 'unknown'
    """
    text_lower = text.lower()
    
    # Passport front side indicators
    front_indicators = [
        "passport", "republic of india", "republic of", "passport no", "passport number",
        "type", "code", "given name", "surname", "given names", "name",
        "date of birth", "place of birth", "nationality", "sex", "gender",
        "date of issue", "authority", "passport authority"
    ]
    
    # Passport back side indicators (usually has MRZ - Machine Readable Zone)
    back_indicators = [
        "<<<", "p<", "mrz", "machine readable", "document number",
        "personal number", "optional data", "checksum"
    ]
    
    front_score = sum(1 for indicator in front_indicators if indicator in text_lower)
    back_score = sum(1 for indicator in back_indicators if indicator in text_lower)
    
    # If MRZ detected, it's likely the data page (front) or back
    if "<<<" in text_lower or "p<" in text_lower:
        # MRZ can be on front (data page) or back
        # Check if we have name/DOB which indicates front page
        if any(word in text_lower for word in ["given name", "surname", "date of birth", "nationality"]):
            return "front"
        return "back"
    
    if front_score > back_score:
        return "front"
    elif back_score > front_score:
        return "back"
    else:
        return "unknown"


def parse_passport_fields(text):
    """
    Minimal global passport parser.
    Extracts only: passport_number, date_of_birth (dob), and date_of_expiry.
    Works for any country format, MRZ or visible text.
    """
    import re
    from datetime import datetime

    data = {}
    raw = text or ""
    normalized = raw.replace("\r", "\n").upper()

    # ---------------------------------------------------------------------
    # PASSPORT NUMBER DETECTION
    # ---------------------------------------------------------------------
    passport_number = None

    # clean out symbols like ¢, $, etc.
    cleaned_text = re.sub(r"[^A-Z0-9<\n\s]", "", normalized)

    # try visible text patterns
    patterns = [
        r"PASSPORT\s*(?:NO|NUMBER|#)?[:\-]?\s*([A-Z0-9]{6,9})",
        r"DOCUMENT\s*NO[:\-]?\s*([A-Z0-9]{6,9})",
        r"\b([A-Z0-9]{6,9})\b"
    ]

    for pattern in patterns:
        m = re.search(pattern, cleaned_text)
        if m:
            passport_number = m.group(1).replace("<", "").strip()
            break

    # try MRZ fallback (e.g., P<USAJOHN<<DOE<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<)
    if not passport_number:
        mrz_lines = re.findall(r"[P<].{40,}", cleaned_text)
        if mrz_lines:
            mrz2 = mrz_lines[-1]
            doc_match = re.search(r"[A-Z<]{0,3}([A-Z0-9]{6,9})", mrz2)
            if doc_match:
                passport_number = doc_match.group(1).replace("<", "").strip()

    if passport_number:
        data["passport_number"] = passport_number

    # ---------------------------------------------------------------------
    # DATE OF BIRTH (DOB)
    # ---------------------------------------------------------------------
    dob = None
    # try standard text forms (e.g., Date of Birth: 14/03/1990)
    dob_patterns = [
        r"DATE\s*OF\s*BIRTH[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})",
        r"\b(\d{2}\/\d{2}\/\d{4})\b"
    ]
    for pattern in dob_patterns:
        m = re.search(pattern, normalized)
        if m:
            dob = m.group(1).replace("-", "/")
            break

    # MRZ fallback: YYMMDD format (e.g., 900314 in MRZ)
    if not dob:
        mrz_match = re.findall(r"(\d{6})", normalized)
        for seq in mrz_match:
            try:
                year = int(seq[:2])
                month = int(seq[2:4])
                day = int(seq[4:6])
                if 1 <= month <= 12 and 1 <= day <= 31:
                    year = 2000 + year if year < 30 else 1900 + year
                    dob = f"{day:02d}/{month:02d}/{year}"
                    break
            except:
                continue

    if dob:
        data["dob"] = dob

    # ---------------------------------------------------------------------
    # EXPIRY DATE
    # ---------------------------------------------------------------------
    expiry = None
    expiry_patterns = [
        r"DATE\s*OF\s*EXPIRY[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})",
        r"EXPIRES?\s*ON[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})"
    ]
    for pattern in expiry_patterns:
        m = re.search(pattern, normalized)
        if m:
            expiry = m.group(1).replace("-", "/")
            break

    # MRZ fallback (the next 6 digits after DOB in MRZ line)
    if not expiry:
        mrz_lines = re.findall(r"[P<].{40,}", normalized)
        if mrz_lines:
            mrz2 = mrz_lines[-1]
            exp_match = re.findall(r"\d{6}", mrz2)
            if len(exp_match) >= 2:
                seq = exp_match[1]
                try:
                    year = int(seq[:2])
                    month = int(seq[2:4])
                    day = int(seq[4:6])
                    year = 2000 + year if year < 30 else 1900 + year
                    expiry = f"{day:02d}/{month:02d}/{year}"
                except:
                    pass

    if expiry:
        data["date_of_expiry"] = expiry

    return data

