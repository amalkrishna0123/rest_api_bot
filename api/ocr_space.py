# ocr_space.py - Enhanced version with PaddleOCR for PDF processing
import io
import os
import requests
from PIL import Image, ImageOps, ImageEnhance
import numpy as np

# OCR.space credentials (kept as fallback/alternative)
OCR_API_KEY = "K82681714188957"
OCR_ENDPOINT = "https://api.ocr.space/parse/image"

# Initialize PaddleOCR (lazy loading)
_paddle_ocr = None

def _get_paddle_ocr():
    """Initialize and return PaddleOCR instance (singleton pattern)"""
    global _paddle_ocr
    if _paddle_ocr is None:
        try:
            from paddleocr import PaddleOCR
            # Initialize with English and Arabic support
            _paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
        except ImportError:
            raise ImportError(
                "PaddleOCR not installed. Please install with: "
                "pip install paddlepaddle paddleocr"
            )
    return _paddle_ocr

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

        resp = requests.post(OCR_ENDPOINT, files=files, data=data, timeout=90)
        
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

def paddle_ocr_file(file_obj, is_pdf=False):
    """
    Process file using PaddleOCR (FREE, on your own server).
    
    :param file_obj: Django InMemoryUploadedFile or file-like object
    :param is_pdf: bool - if True, processes PDF page by page
    :return: (parsed_text: str, exit_code: int, error: str)
    """
    try:
        ocr = _get_paddle_ocr()
        file_obj.seek(0)
        
        if is_pdf:
            # Process PDF using pymupdf
            try:
                import fitz  # PyMuPDF
                pdf_bytes = file_obj.read()
                
                if not pdf_bytes:
                    return "", 6, "PDF file is empty or could not be read"
                
                if not pdf_bytes.startswith(b'%PDF'):
                    return "", 6, "File does not appear to be a valid PDF"
                
                # Open PDF with PyMuPDF
                pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                combined_text = ""
                
                # Process each page
                for page_num in range(len(pdf_doc)):
                    page = pdf_doc[page_num]
                    # Render page to image (pixmap)
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
                    img_data = pix.tobytes("png")
                    
                    # Convert to PIL Image for PaddleOCR
                    img = Image.open(io.BytesIO(img_data))
                    img_array = np.array(img)
                    
                    # Run PaddleOCR
                    result = ocr.ocr(img_array, cls=True)
                    
                    # Extract text from OCR result
                    page_text = ""
                    if result and result[0]:
                        for line in result[0]:
                            if line and len(line) >= 2:
                                text_info = line[1]
                                if text_info and len(text_info) >= 2:
                                    page_text += text_info[0] + " "
                    
                    if page_text.strip():
                        combined_text += f"--- Page {page_num + 1} ---\n{page_text.strip()}\n\n"
                
                pdf_doc.close()
                return combined_text.strip(), 0, ""
                
            except ImportError:
                return "", 6, "PyMuPDF (pymupdf) not installed. Please install with: pip install pymupdf"
            except Exception as pdf_err:
                return "", 6, f"PDF processing error: {str(pdf_err)}"
        else:
            # Process image
            img = Image.open(file_obj)
            img_array = np.array(img)
            
            # Run PaddleOCR
            result = ocr.ocr(img_array, cls=True)
            
            # Extract text from OCR result
            text = ""
            if result and result[0]:
                for line in result[0]:
                    if line and len(line) >= 2:
                        text_info = line[1]
                        if text_info and len(text_info) >= 2:
                            text += text_info[0] + " "
            
            return text.strip(), 0, ""
            
    except ImportError as e:
        return "", 6, f"PaddleOCR not installed: {str(e)}. Please install with: pip install paddlepaddle paddleocr"
    except Exception as e:
        return "", 6, f"PaddleOCR processing error: {str(e)}"

def ocr_space_file_multi_lang(file_obj, is_pdf=False):
    """
    Try OCR with PaddleOCR first (FREE), then fallback to OCR.space if needed.
    Stop early if we get substantial text or detect MRZ.
    """
    best_text, best_code, best_err = "", 5, "No OCR results returned"
    
    # First try PaddleOCR (FREE, on your own server)
    try:
        text, code, err = paddle_ocr_file(file_obj, is_pdf)
        if code == 0 and len(text.strip()) > 40:
            return text, code, err
        if code == 0 and _has_mrz(text):
            return text, code, err
        if code == 0 and len(text.strip()) > 0:
            best_text, best_code, best_err = text, code, err
    except Exception as e:
        # PaddleOCR failed, continue to OCR.space fallback
        pass
    
    # Fallback to OCR.space if PaddleOCR didn't work well
    if not best_text or len(best_text.strip()) < 40:
        if is_pdf:
            txt, code, err = ocr_space_pdf_all_pages(file_obj)
            if code == 0 and (len(txt.strip()) > 40 or _has_mrz(txt)):
                return txt, code, err
            if code == 0 and len(txt.strip()) > 0:
                if len(txt.strip()) > len(best_text.strip()):
                    best_text, best_code, best_err = txt, code, err
        else:
            # For images, cycle languages
            lang_candidates = [
                "eng",
                "eng+ara+fas+urd",  # Arabic family
            ]
            
            for lang in lang_candidates:
                file_obj.seek(0)
                txt, code, err = ocr_space_file(file_obj, False, lang)
                if code == 0 and len((txt or "").strip()) > 40:
                    return txt, code, err
                if code == 0 and _has_mrz(txt):
                    return txt, code, err
                if code == 0 and len(txt.strip()) > len(best_text.strip()):
                    best_text, best_code, best_err = txt, code, err
    
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
        files = {"file": (file_obj.name, pdf_content, "application/pdf")}
        
        data = {
            "apikey": OCR_API_KEY,
            "language": "eng",
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
        
        # First, extract any successfully parsed pages (even if there's an error)
        if j.get("IsErroredOnProcessing"):
            error_msg = j.get("ErrorMessage", "Unknown OCR.space error")
            
            # If page limit reached, try to process remaining pages with PaddleOCR
            if "maximum page limit" in str(error_msg).lower() or "page limit" in str(error_msg).lower():
                parsed_results = j.get("ParsedResults", [])
                if parsed_results:
                    for result in parsed_results:
                        page_text = result.get("ParsedText", "")
                        if page_text.strip():
                            combined_text += page_text + "\n\n"
                
                # Try to get remaining pages (4-10) with PaddleOCR
                try:
                    import fitz
                    file_obj.seek(0)
                    pdf_bytes = file_obj.read()
                    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                    ocr = _get_paddle_ocr()
                    
                    # Process pages 4-10 (0-indexed: 3-9)
                    for page_num in range(min(3, len(pdf_doc)), min(10, len(pdf_doc))):
                        page = pdf_doc[page_num]
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                        img_data = pix.tobytes("png")
                        img = Image.open(io.BytesIO(img_data))
                        img_array = np.array(img)
                        
                        result = ocr.ocr(img_array, cls=True)
                        page_text = ""
                        if result and result[0]:
                            for line in result[0]:
                                if line and len(line) >= 2:
                                    text_info = line[1]
                                    if text_info and len(text_info) >= 2:
                                        page_text += text_info[0] + " "
                        
                        if page_text.strip():
                            combined_text += f"--- Page {page_num + 1} ---\n{page_text.strip()}\n\n"
                    
                    pdf_doc.close()
                    
                    if len(combined_text.strip()) > 10:
                        return combined_text.strip(), 0, f"Processed first 3 pages via OCR.space, remaining via PaddleOCR"
                except Exception as paddle_err:
                    # If PaddleOCR fails, return what we have
                    if len(combined_text.strip()) > 10:
                        return combined_text.strip(), 0, f"Processed first 3 pages (page limit). {error_msg}"
            
            # If "No images extracted", try with different OCR engine
            if "No images extracted" in str(error_msg) or "no images" in str(error_msg).lower():
                file_obj.seek(0)
                pdf_content = file_obj.read()
                files = {"file": (file_obj.name, pdf_content, "application/pdf")}
                data["OCREngine"] = "1"
                data["detectOrientation"] = "true"
                data["scale"] = "true"
                
                retry_resp = requests.post(OCR_ENDPOINT, files=files, data=data, timeout=90)
                if retry_resp.status_code == 200:
                    retry_j = retry_resp.json()
                    if not retry_j.get("IsErroredOnProcessing"):
                        parsed_results = retry_j.get("ParsedResults", [])
                        if parsed_results:
                            for result in parsed_results:
                                page_text = result.get("ParsedText", "")
                                if page_text.strip():
                                    combined_text += page_text + "\n\n"
                            if len(combined_text.strip()) > 10:
                                return combined_text.strip(), 0, ""
            
            if len(combined_text.strip()) > 10:
                return combined_text.strip(), 0, error_msg
            return "", 4, error_msg

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
        file_obj.seek(0)
        files = {"file": (file_obj.name, pdf_content, "application/pdf")}
        data["language"] = "ara"
        
        resp = requests.post(OCR_ENDPOINT, files=files, data=data, timeout=90)
        
        if resp.status_code != 200:
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

def process_pdf_page_by_page(file_obj):
    """
    Process PDF page-by-page using PaddleOCR and identify which pages contain Emirates ID content.
    Returns a list of page data with text and detected side.
    
    This function handles PDFs with multiple formats:
    - Format 1: Page 1 = front, Page 2 = back
    - Format 2: Pages 2-3 = Emirates ID, Page 1 = visa, Page 4 = other info
    """
    try:
        import fitz  # PyMuPDF
        ocr = _get_paddle_ocr()
        
        # Reset file pointer
        file_obj.seek(0)
        pdf_bytes = file_obj.read()
        
        if not pdf_bytes:
            return []
        
        # Validate PDF header
        if not pdf_bytes.startswith(b'%PDF'):
            return []
        
        # Open PDF with PyMuPDF
        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_data_list = []
        
        # Process each page
        for page_num in range(len(pdf_doc)):
            try:
                page = pdf_doc[page_num]
                # Render page to image
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
                img_data = pix.tobytes("png")
                
                # Convert to PIL Image for PaddleOCR
                img = Image.open(io.BytesIO(img_data))
                img_array = np.array(img)
                
                # Run PaddleOCR
                result = ocr.ocr(img_array, cls=True)
                
                # Extract text from OCR result
                page_text = ""
                if result and result[0]:
                    for line in result[0]:
                        if line and len(line) >= 2:
                            text_info = line[1]
                            if text_info and len(text_info) >= 2:
                                page_text += text_info[0] + " "
                
                # Process page even if text is minimal (might be image-heavy)
                # Detect document side
                side = detect_document_side(page_text) if page_text.strip() else "unknown"
                
                # Check if this page contains Emirates ID content (position-independent)
                is_emirates_id = _is_emirates_id_page(page_text) if page_text.strip() else False
                
                # Always add page data, even if not identified as Emirates ID
                # This allows for fallback detection in views.py
                page_data_list.append({
                    'page_number': page_num + 1,
                    'text': page_text.strip(),
                    'side': side,
                    'is_emirates_id': is_emirates_id
                })
            except Exception as page_err:
                # Log error but continue with other pages
                # Add empty entry to maintain page numbering
                page_data_list.append({
                    'page_number': page_num + 1,
                    'text': '',
                    'side': 'unknown',
                    'is_emirates_id': False
                })
                continue
        
        pdf_doc.close()
        return page_data_list
        
    except ImportError:
        # PyMuPDF or PaddleOCR not installed, fall back to OCR.space for entire PDF
        text, code, err = ocr_space_pdf_all_pages(file_obj)
        if code == 0:
            side = detect_document_side(text)
            return [{
                'page_number': 1,
                'text': text,
                'side': side,
                'is_emirates_id': True
            }]
        return []
    except Exception as e:
        return []

def _is_emirates_id_page(text):
    """
    Check if a page contains Emirates ID content.
    Returns True if the page likely contains Emirates ID information.
    Uses weighted scoring to identify Emirates ID pages regardless of position.
    """
    if not text:
        return False
    
    import re
    text_lower = text.lower()
    
    # Strong Emirates ID indicators (weighted heavily)
    strong_indicators = [
        r'\b\d{3}-\d{4}-\d{7}-\d\b',  # Emirates ID number pattern (strongest - format: 784-1234-1234567-1)
        "emirates id", "identity card", "بطاقة الهوية", "بطاقة الهوية الإماراتية",
        "784-",  # Emirates ID number prefix
        "resident identity card", "بطاقة هوية المقيم",
    ]
    
    # Medium indicators (common Emirates ID fields)
    medium_indicators = [
        "united arab emirates", "الإمارات العربية المتحدة",
        "nationality", "الجنسية",
        "date of birth", "تاريخ الميلاد",
        "occupation", "المهنة",
        "employer", "صاحب العمل",
        "issuing place", "مكان الإصدار",
        "expiry date", "تاريخ انتهاء الصلاحية",
        "issuing date", "تاريخ الإصدار",
    ]
    
    # Weak indicators (might appear on other documents too)
    weak_indicators = [
        "name", "الاسم",
        "address", "العنوان",
        "gender", "الجنس",
    ]
    
    # Check for Emirates ID number pattern (strongest indicator - almost certain)
    if re.search(r'\b\d{3}-\d{4}-\d{7}-\d\b', text):
        return True
    
    # Check for strong indicators
    strong_count = 0
    for indicator in strong_indicators:
        if indicator.startswith('\\') or indicator.startswith('r'):
            # It's a regex pattern
            if re.search(indicator, text, re.IGNORECASE):
                strong_count += 1
        else:
            if indicator in text_lower:
                strong_count += 1
    
    # Check for medium indicators
    medium_count = sum(1 for indicator in medium_indicators if indicator in text_lower)
    
    # Check for weak indicators (only count if we have other indicators)
    weak_count = sum(1 for indicator in weak_indicators if indicator in text_lower)
    
    # Scoring: strong indicators count more
    # Formula: (strong * 3) + (medium * 2) + (weak * 0.5)
    score = (strong_count * 3) + (medium_count * 2) + (weak_count * 0.5)
    
    # Threshold: Position-independent detection
    # - If we have at least one strong indicator, it's likely Emirates ID
    # - Or if we have 3+ medium indicators, it's likely Emirates ID
    # - Or if score >= 4 (combination of indicators)
    # - Or if we have 2+ medium indicators AND 1+ weak indicator (relaxed for edge cases)
    return (strong_count >= 1 or 
            medium_count >= 3 or 
            score >= 4 or 
            (medium_count >= 2 and weak_count >= 1))

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
