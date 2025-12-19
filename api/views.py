import os
from openai import OpenAI
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from google.cloud import speech

import pytesseract
from PIL import Image
import pdfplumber
import openpyxl
import tempfile
from django.views.decorators.http import require_POST
from pdf2image import convert_from_path
from .models import ChatSession, EmiratesIDRecord, PassportRecord
import uuid

import random
import datetime
from django.core.mail import send_mail
from django.conf import settings
from .models import OTP

import json, uuid
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import ChatSession, ChatMessage
from .ocr_space import ocr_space_file

import io, hashlib, tempfile, re, math, json
from datetime import datetime, date
from django.core.cache import cache
from PIL import Image, ImageOps
import numpy as np

# OpenCV + OCR stacks
import cv2
import easyocr
from passporteye import read_mrz
from pdf2image import convert_from_bytes
import requests
# from openai import OpenAI


def deepseek_chat(messages, max_tokens=500, temperature=0.3):
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    response = requests.post(
        DEEPSEEK_API_URL,
        headers=headers,
        json=payload,
        timeout=60,
    )

    response.raise_for_status()
    data = response.json()

    return data["choices"][0]["message"]["content"]



client = OpenAI(
    base_url="https://text.pollinations.ai/openai",
    api_key="pollinations",  
)



SERVICE_ACCOUNT_FILE = r"C:\Users\GL_Amal\OneDrive\Desktop\AMAL KRISHNA\ChatBot - Copy\chatbot_project\backend\chatbot.json"


pytesseract.pytesseract.tesseract_cmd = r"C:\Users\GL_Amal\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

def index(request):
    return render(request, "index.html")


INSURANCE_ONBOARDING_SYSTEM_PROMPT = (
    "You are an assistant that helps collect data for medical insurance onboarding.\n"
    "You MUST output exactly one JSON object and nothing else (no commentary, no text outside JSON).\n"
    "The JSON object must contain:\n"
    "  - reply (string): the message to show to the user\n"
    "  - options (array of strings, optional): predefined buttons if available, [] if free text is expected\n"
    "  - session_updates (object, optional): key/value pairs to update in the session (like step, role, etc.)\n"
    "  - complete (boolean): true only if onboarding is finished\n\n"

    "Behavior rules:\n"
    " - Always ask only ONE question at a time.\n"
    " - Keep replies short and clear.\n"
    " - If you need the user to upload Emirates ID, set session_updates.step to "
    "'awaiting_id_frontside' or 'awaiting_id_backside' and provide a reply that tells the user to upload.\n"
    " - If onboarding is finished, set complete=true and session_updates.step='complete'.\n"
    " - If showing buttons, include them under options. If not, return options as [].\n\n"

    "REQUIRED_FIELDS = [full_name, emirates_id_number, dob, expiry, nationality, occupation, salary].\n"
    " - If any of these fields are missing or null in the context, you must ask the user for it explicitly.\n"
    " - Ask for one missing field at a time (never group them together).\n"
    " - When the user answers, set that field in session_updates.\n"
    " - Once all REQUIRED_FIELDS are filled, continue to sponsor question before completing.\n\n"

    "Allowed step values: start, q1, q2, q2a, q3, salary_q, sponsor_q, awaiting_id_frontside, awaiting_id_backside, awaiting_passport, passport_verified, passport_validation_failed, complete.\n"
    "Session fields you may update: step, looking_for_insurance, role, depender_type, salary, "
    "emirates_id_uploaded, is_completed, full_name, emirates_id_number, dob, expiry, nationality, occupation, sponsor_name.\n\n"
    # "IMPORTANT FLOW: After products are displayed and user selects a product, you must ask for passport upload.\n"
    # "Set session_updates.step='awaiting_passport' and ask: 'Please upload your passport (front and back pages or PDF with both pages).'\n\n"

    "### Few-shot examples ###\n\n"

    "User input:\n"
    "{\"context\":{\"step\":\"start\"},\"last_user_message\":\"\"}\n"
    "Assistant output:\n"
    "{\n"
    "  \"reply\": \"Are you looking for medical insurance?\",\n"
    "  \"options\": [\"Yes\", \"No\"],\n"
    "  \"session_updates\": {\"step\": \"q1\"},\n"
    "  \"complete\": false\n"
    "}\n\n"

    "User input:\n"
    "{\"context\":{\"step\":\"q1\"},\"last_user_message\":\"Yes\"}\n"
    "Assistant output:\n"
    "{\n"
    "  \"reply\": \"Are you an employee or a depender?\",\n"
    "  \"options\": [\"Employee\", \"Depender\"],\n"
    "  \"session_updates\": {\"looking_for_insurance\": \"Yes\", \"step\": \"q2\"},\n"
    "  \"complete\": false\n"
    "}\n\n"

    "User input:\n"
    "{\"context\":{\"step\":\"q2\"},\"last_user_message\":\"Employee\"}\n"
    "Assistant output:\n"
    "{\n"
    "  \"reply\": \"What is your monthly salary?\",\n"
    "  \"options\": [\"below 4000 AED\", \"4000 - 5000 AED\", \"above 5000 AED\"],\n"
    "  \"session_updates\": {\"role\": \"Employee\", \"step\": \"salary_q\"},\n"
    "  \"complete\": false\n"
    "}\n\n"

    "User input:\n"
    "{\"context\":{\"step\":\"q2\"},\"last_user_message\":\"Depender\"}\n"
    "Assistant output:\n"
    "{\n"
    "  \"reply\": \"What type of Depender are you?\",\n"
    "  \"options\": [\"Spouse\", \"Child\"],\n"
    "  \"session_updates\": {\"role\": \"Depender\", \"step\": \"q2a\"},\n"
    "  \"complete\": false\n"
    "}\n\n"

    "User input:\n"
    "{\"context\":{\"step\":\"salary_q\"},\"last_user_message\":\"4000-10000 AED\"}\n"
    "Assistant output:\n"
    "{\n"
    "  \"reply\": \"Please upload your valid Emirates ID.\",\n"
    "  \"options\": [],\n"
    "  \"session_updates\": {\"salary\": \"4000-10000 AED\", \"step\": \"awaiting_id_frontside\"},\n"
    "  \"complete\": false\n"
    "}\n\n"

    "User input:\n"
    "{\"context\":{\"step\":\"q3\",\"emirates_id_data\":{\"full_name\":\"John Doe\",\"emirates_id_number\":\"784-1234-5678901-2\",\"dob\":\"1990-01-01\",\"expiry\":\"2027-03-15\",\"nationality\":\"India\",\"occupation\":null}},\"last_user_message\":\"Yes, continue\"}\n"
    "Assistant output:\n"
    "{\n"
    "  \"reply\": \"I could not find your occupation in the Emirates ID. Could you please tell me your occupation?\",\n"
    "  \"options\": [],\n"
    "  \"session_updates\": {\"step\":\"q3\"},\n"
    "  \"complete\": false\n"
    "}\n\n"

    "User input:\n"
    "{\"context\":{\"step\":\"q3\",\"emirates_id_data\":{\"full_name\":\"John Doe\",\"emirates_id_number\":\"784-1234-5678901-2\",\"dob\":\"1990-01-01\",\"expiry\":\"2027-03-15\",\"nationality\":\"India\",\"occupation\":null}},\"last_user_message\":\"Software Engineer\"}\n"
    "Assistant output:\n"
    "{\n"
    "  \"reply\": \"Thank you. I've recorded your occupation as Software Engineer. Let's continue.\",\n"
    "  \"options\": [],\n"
    "  \"session_updates\": {\"occupation\": \"Software Engineer\", \"step\": \"q4\"},\n"
    "  \"complete\": false\n"
    "}\n\n"

    "User input:\n"
    "{\"context\":{\"step\":\"sponsor_q\",\"last_user_message\":\"No\"}}\n"
    "Assistant output:\n"
    "{\n"
    "  \"reply\": \"Thank you. Your onboarding is now complete.\",\n"
    "  \"options\": [],\n"
    "  \"session_updates\": {\"step\": \"complete\", \"is_completed\": true},\n"
    "  \"complete\": true\n"
    "}\n\n"

    "User input:\n"
    "{\"context\":{\"step\":\"sponsor_q\",\"last_user_message\":\"Yes\"}}\n"
    "Assistant output:\n"
    "{\n"
    "  \"reply\": \"Please provide your sponsor's name.\",\n"
    "  \"options\": [],\n"
    "  \"session_updates\": {\"step\": \"sponsor_q\"},\n"
    "  \"complete\": false\n"
    "}\n\n"

    "Now continue following the same format for every step.\n"
)



@csrf_exempt
def transcribe_audio(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    audio = request.FILES.get("file")
    if not audio:
        return JsonResponse({"error": "no file"}, status=400)

    try:
        client_stt = speech.SpeechClient.from_service_account_file(SERVICE_ACCOUNT_FILE)
        content = audio.read()
        audio_config = speech.RecognitionAudio(content=content)

        # Auto-detect multiple languages, let Google infer encoding + sample rate
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED,
            language_code="en-US",
            alternative_language_codes=["ml-IN", "hi-IN", "ar-AE", "ur-IN", "ur-PK"]
        )

        response = client_stt.recognize(config=config, audio=audio_config)
        text = ""

        for result in response.results:
            text += result.alternatives[0].transcript + " "

        return JsonResponse({"text": text.strip(), "language": "auto"})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


# @csrf_exempt
# def chat_reply(request):
#     import json
#     if request.method != "POST":
#         return JsonResponse({"error": "POST only"}, status=405)
#     payload = json.loads(request.body.decode("utf-8"))
#     user_text = payload.get("user_text", "").strip()

#     if not user_text:
#         return JsonResponse({"error": "user_text required"}, status=400)

#     try:
#         resp = client.chat.completions.create(
#             model="openai",
#             messages=[
#                 {"role": "system", "content": INSURANCE_ONBOARDING_SYSTEM_PROMPT},
#                 {"role": "user", "content": user_text}
#             ],
#             max_tokens=500,

#         )
#         reply = resp.choices[0].message.content.strip()
#         return JsonResponse({"reply": reply})
#     except Exception as e:
#         return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def chat_reply(request):
    import json

    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    payload = json.loads(request.body.decode("utf-8"))

    intent = payload.get("intent", "insurance")
    user_text = payload.get("user_text", "").strip()

    if not user_text:
        return JsonResponse({"error": "user_text required"}, status=400)

    # Select system prompt based on intent
    if intent == "insurance":
        system_prompt = INSURANCE_ONBOARDING_SYSTEM_PROMPT

    elif intent == "details":
        system_prompt = (
            "You are a helpful assistant. "
            "Ask clarifying questions to understand what details the user wants. "
            "Do not mention insurance unless the user asks."
        )

    else:  # free
        system_prompt = (
            "You are a general AI assistant. "
            "Answer naturally and detect user intent from the user's question."
        )

    try:
        resp = client.chat.completions.create(
            model="openai",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ],
            max_tokens=500,
        )

        reply = resp.choices[0].message.content.strip()
        return JsonResponse({"reply": reply})

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)



def extract_from_image(file_path):
    """
    Use pytesseract to OCR an image. Try English+Arabic first if available,
    otherwise fallback to the default language.
    """
    from PIL import Image
    try:
        # try English + Arabic. remove '+ara' if Arabic traineddata not installed.
        return pytesseract.image_to_string(Image.open(file_path), lang='eng+ara')
    except Exception:
        try:
            return pytesseract.image_to_string(Image.open(file_path))
        except Exception as e:
            return f"[pytesseract error: {e}]"



def extract_from_pdf(file_path):
    """
    Try direct text extraction with pdfplumber. If nothing found, render high-DPI images
    and OCR them using pytesseract (with English+Arabic if available).
    """
    text = ""
    try:
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        text += f"[pdfplumber error: {e}]\n"

    # Fallback to OCR if no text found
    if not text.strip():
        try:
            pages = convert_from_path(
                file_path,
                dpi=300,
                poppler_path=r"C:\Users\GL_Amal\Downloads\Release-25.07.0-0\poppler-25.07.0\Library\bin"
            )
            for page_img in pages:
                try:
                    # prefer english+arabic if available
                    text += pytesseract.image_to_string(page_img, lang='eng+ara') + "\n"
                except Exception:
                    text += pytesseract.image_to_string(page_img) + "\n"
        except Exception as e:
            text += f"[OCR error: {e}]"

    return text

def extract_from_excel(file_path):
    text = ""
    wb = openpyxl.load_workbook(file_path)
    for sheet in wb.worksheets:
        for row in sheet.iter_rows(values_only=True):
            text += " ".join([str(cell) for cell in row if cell]) + "\n"
    return text


def parse_fields(text):
    """
    Robust field parser for Emirates ID text.
    Returns a dict with at least possible keys:
      emirates_id, name, address, expiry_date, issuing_date, dob, nationality, sex
    """
    import re
    from datetime import datetime

    data = {}
    raw = text or ""
    normalized = raw.replace('\r', '\n')

    # Define stopwords once, usable in all branches
    stopwords = [
        'resident', 'identity', 'card', 'id number', 'id no', 'number',
        'nationality', 'date of birth', 'date', 'expiry', 'issuing',
        'signature', 'sex', 'passport', 'signature/', 'issue'
    ]

    # 1) Emirates ID number
    m = re.search(r'\b\d{3}-\d{4}-\d{7}-\d\b', raw)
    if m:
        data['emirates_id'] = m.group(0).strip()

    # 2) Labeled name
    m = re.search(r'(?i)(?:Name|NAME|الاسم|اسم)\s*[:\-]?\s*([^\n\r]{2,80})', raw)
    if m:
        name_val = m.group(1).strip()
        name_val = re.sub(r'[\:\-\/\|]+$', '', name_val).strip()
        data['name'] = name_val

    # 3) Fallback name detection if not found
    if 'name' not in data:
        lines = [l.strip() for l in normalized.splitlines() if l.strip()]
        candidate = None
        for idx, line in enumerate(lines):
            low = line.lower()
            if any(sw in low for sw in stopwords):
                continue
            if any(ch.isdigit() for ch in line):
                continue
            if len(line.split()) >= 2 and len(line) > 3:
                if 'united arab emirates' in low or 'arab emirates' in low:
                    continue
                name_candidate = line
                if idx + 1 < len(lines):
                    nxt = lines[idx + 1]
                    if (not any(sw in nxt.lower() for sw in stopwords)
                        and not any(ch.isdigit() for ch in nxt)
                        and len(nxt.split()) <= 3):
                        name_candidate = name_candidate + ' ' + nxt
                candidate = name_candidate
                break
        if candidate:
            data['name'] = candidate

    # 4) Dates
    date_pattern = r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})'
    all_dates = re.findall(date_pattern, raw)

    def find_date_by_labels(labels):
        for lbl in labels:
            m = re.search(r'(?i)' + lbl + r'\s*[:\-]?\s*' + date_pattern, raw)
            if m:
                return m.group(1)
        return None

    dob = find_date_by_labels(['Date of Birth', 'DOB', 'Birth', 'تاريخ الميلاد'])
    issuing = find_date_by_labels([
    'Issuing Date', 'Issue Date', 'Issuing', 'Date of Issue',
    'تاريخ الإصدار'
    ])
    expiry = find_date_by_labels(['Expiry Date', 'Expiry', 'تاريخ الانتهاء'])

    # Convert dates to DD/MMM/YYYY format if found
    def format_date(date_str):
        try:
            # Handle different separators
            if '/' in date_str:
                day, month, year = date_str.split('/')
            elif '-' in date_str:
                day, month, year = date_str.split('-')
            else:
                return date_str
                
            # Convert to proper format
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            if month.isdigit() and 1 <= int(month) <= 12:
                month = month_names[int(month) - 1]
            return f"{day}/{month}/{year}"
        except:
            return date_str

    if dob:
        data['dob'] = format_date(dob)
    if issuing:
        data['issuing_date'] = format_date(issuing)
    if expiry:
        data['expiry_date'] = format_date(expiry)
    if 'expiry_date' not in data and all_dates:
        data['expiry_date'] = format_date(all_dates[-1])
    if not issuing and all_dates:
        # Heuristic: first date is DOB, second is Issuing, last is Expiry
        if len(all_dates) >= 2:
            data["issuing_date"] = format_date(all_dates[1])

    # 5) Nationality
    m = re.search(r'(?i)(?:Nationality|الجنسية)\s*[:\-]?\s*([^\n\r]{2,80})', raw)
    if m:
        data['nationality'] = m.group(1).strip()

    # 6) Sex / Gender
    gender_patterns = [
        r'(?i)(?:Sex|Gender|الجنس)\s*[:\-]?\s*(Male|Female|M|F|ذكر|أنثى)',
        r'(?i)(?:Sex|Gender|الجنس)\s*[:\-]?\s*([^\n]{1,20})',
        r'(?i)(SCANNE|SCANNED|SCAN|SCANNING)',
    ]

    gender_val = None
    for pattern in gender_patterns:
        m = re.search(pattern, raw)
        if m:
            val = m.group(1).strip() if m.group(1) else ""
            if val and not any(word in val.upper() for word in ['SCAN', 'SCANNED', 'SCANNING']):
                gender_val = val
                break
    
    # If we found a gender value, normalize it
    if gender_val:
        val = gender_val.upper()
        if val in ["M", "MALE", "ذكر"]:
            data["gender"] = "Male"
            data["sex"] = "Male"
        elif val in ["F", "FEMALE", "أنثى"]:
            data["gender"] = "Female"
            data["sex"] = "Female"
        else:
            data["gender"] = gender_val
            data["sex"] = gender_val


    # 7) Address
    m = re.search(r'(?i)(?:Address|العنوان)\s*[:\-]?\s*([^\n\r]{2,140})', raw)
    if m:
        data['address'] = m.group(1).strip()
    else:
        if 'nationality' in data:
            lines = [l.strip() for l in normalized.splitlines() if l.strip()]
            for idx, line in enumerate(lines):
                if data['nationality'].lower() in line.lower():
                    if idx + 1 < len(lines):
                        cand = lines[idx + 1]
                        if len(cand) > 4 and not any(ch.isdigit() for ch in cand):
                            data['address'] = cand
                            break
        if 'address' not in data:
            lines = [l.strip() for l in normalized.splitlines() if l.strip()]
            for l in reversed(lines):
                low = l.lower()
                if any(sw in low for sw in stopwords):
                    continue
                if any(ch.isdigit() for ch in l):
                    continue
                if len(l.split()) >= 2:
                    data['address'] = l
                    break

    for k, v in list(data.items()):
        if isinstance(v, str):
            data[k] = v.strip().rstrip(':').strip()

    return data



from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
import re

@csrf_exempt
@require_POST
def emirates_id_upload(request):
    """
    Multi-file front/back upload using OCR.space.
    Handles multiple files in one upload, auto-detects front/back sides,
    merges extracted data into one EmiratesIDRecord,
    and syncs extracted fields with ChatSession.
    """
    from .ocr_space import ocr_space_file_multi_lang, detect_document_side

    session_id = request.POST.get("session_id")
    if not session_id:
        return JsonResponse({"error": "session_id required"}, status=400)

    try:
        session = ChatSession.objects.get(session_id=session_id)
    except ChatSession.DoesNotExist:
        return JsonResponse({"error": "Chat session not found"}, status=404)

    # Collect all uploaded files
    files = request.FILES.getlist("file")
    if not files:
        return JsonResponse({"error": "No files received"}, status=400)

    # Process each file with OCR.space
    processed_files = []
    for file_obj in files:
        is_pdf = file_obj.name.lower().endswith(".pdf")

        # Use the improved multi-language OCR function
        text, code, err = ocr_space_file_multi_lang(file_obj, is_pdf)
        if code != 0:
            return JsonResponse({"error": f"OCR.space failed for {file_obj.name}: {err}"}, status=500)

        # Detect document side
        side = detect_document_side(text)

        processed_files.append({
            'name': file_obj.name,
            'text': text,
            'side': side,
            'is_pdf': is_pdf
        })

    # Separate front and back side text
    front_texts = [f['text'] for f in processed_files if f['side'] in ['front', 'unknown']]
    back_texts = [f['text'] for f in processed_files if f['side'] == 'back']

    # Combine texts
    front_combined = " ".join(front_texts) if front_texts else ""
    back_combined = " ".join(back_texts) if back_texts else ""

    # If no clear side detection, use heuristics
    if not front_combined and back_combined:
        front_combined = back_combined
    elif not back_combined and front_combined:
        back_combined = ""

    # Parse fields from combined text
    front_data = parse_fields(front_combined)
    back_data = parse_back_side_fields(back_combined)

    # Handle family sponsor follow-up
    family_sponsor_name = None
    if back_data.get("family_sponsor") == "Yes":
        name_match = re.search(r'(?:family sponsor name|اسم الكفيل|كفيل)\s*[:\-]?\s*([^\n\r]{2,80})',
                               back_combined, re.IGNORECASE)
        if name_match:
            family_sponsor_name = name_match.group(1).strip()

    # Get or create EmiratesIDRecord
    record, created = EmiratesIDRecord.objects.get_or_create(
        chat_session=session,
        defaults={
            "emirates_id": front_data.get("emirates_id"),
            "name": front_data.get("name"),
            "dob": front_data.get("dob"),
            "nationality": front_data.get("nationality"),
            "gender": front_data.get("gender"),
            "address": front_data.get("address"),
            "issuing_date": front_data.get("issuing_date"),
            "expiry_date": front_data.get("expiry_date"),
            "occupation": back_data.get("occupation"),
            "employer": back_data.get("employer"),
            "issuing_place": back_data.get("issuing_place"),
            "family_sponsor": back_data.get("family_sponsor"),
            "family_sponsor_name": family_sponsor_name,
            "raw_response": {
                "front_text": front_combined[:2000],
                "back_text": back_combined[:2000],
                "processed_files": [f['name'] for f in processed_files]
            }
        }
    )

    if not created:
        # Update existing record
        update_fields = []
        for field, value in {**front_data, **back_data}.items():
            if hasattr(record, field) and value:
                setattr(record, field, value)
                update_fields.append(field)

        if family_sponsor_name:
            record.family_sponsor_name = family_sponsor_name
            update_fields.append('family_sponsor_name')

        if update_fields:
            record.save(update_fields=update_fields)

    # ✅ Sync extracted fields into ChatSession
    if record:
        update_fields = []

        if record.name and not session.full_name:
            session.full_name = record.name
            update_fields.append('full_name')

        if record.emirates_id and not session.emirates_id_number:
            session.emirates_id_number = record.emirates_id
            update_fields.append('emirates_id_number')

        if record.dob and not session.dob:
            session.dob = record.dob
            update_fields.append('dob')

        if record.expiry_date and not session.expiry:
            session.expiry = record.expiry_date
            update_fields.append('expiry')

        if record.nationality and not session.nationality:
            session.nationality = record.nationality
            update_fields.append('nationality')

        if record.occupation and not session.occupation:
            session.occupation = record.occupation
            update_fields.append('occupation')

        session.emirates_id_uploaded = True
        update_fields.append('emirates_id_uploaded')

        if update_fields:
            session.save(update_fields=update_fields)

    # Compute missing fields
    expected_front = [
        "emirates_id", "name", "dob", "nationality", "gender",
        "address", "issuing_date", "expiry_date"
    ]
    expected_back = [
        "occupation", "employer", "issuing_place"
    ]

    # missing = [
    #     field for field in expected_front + expected_back
    #     if not getattr(record, field, None)
    # ]
    # if record.family_sponsor == "Yes" and not record.family_sponsor_name:
    #     missing.append("family_sponsor_name")

    # for field in expected_front:
    #     if not getattr(record, field, None):
    #         missing.append(field)

    # for field in expected_back:
    #     if not getattr(record, field, None):
    #         missing.append(field)

    # if record.family_sponsor == "Yes" and not record.family_sponsor_name:
    #     missing.append("family_sponsor_name")

    missing = [
        field for field in expected_front + expected_back
        if not getattr(record, field, None)
    ]
    if record.family_sponsor == "Yes" and not record.family_sponsor_name:
        missing.append("family_sponsor_name")


    # Determine next step
    has_back_side = any(f['side'] == 'back' for f in processed_files)
    next_step = "complete" 

    if has_back_side and session.step in ["awaiting_id_frontside", "awaiting_id_backside"]:
        session.step = "complete"
        session.save()

    # ✅ Use the new helper function here
    products, message = compute_products_based_on_data(session, record)

    # Normalize issuing place (robust to common OCR errors)
    issuing_raw = (record.issuing_place or "") if record else ""
    issuing_norm = issuing_raw.strip().lower()

    # canonicalize common OCR variations
    place = None
    if issuing_norm:
        # Abu Dhabi has many OCR variants like 'abu', 'abu dabhi', 'abudhabi', etc.
        if ("abu" in issuing_norm and "dub" not in issuing_norm) or "adhabi" in issuing_norm or "adabi" in issuing_norm or "abudhabi" in issuing_norm or "abud" in issuing_norm:
            place = "Abu Dhabi"
        elif "duba" in issuing_norm or "dubai" in issuing_norm or "dubi" in issuing_norm:
            place = "Dubai"
        else:
            # fallback: exact words
            if "abu dhabi" in issuing_norm:
                place = "Abu Dhabi"
            elif "dubai" in issuing_norm:
                place = "Dubai"
            else:
                place = issuing_raw.strip().title() if issuing_raw else None

    # Normalize salary selection from session
    salary_raw = (session.salary or "").lower() if session else ""
    salary_key = None

    # look for the canonical options user should have selected:
    # "below 4000 AED", "4000 - 5000 AED", "above 5000 AED"
    if "below" in salary_raw and "4000" in salary_raw:
        salary_key = "below_4000"
    elif "4000" in salary_raw and "5000" in salary_raw:
        salary_key = "4000_5000"
    elif "above" in salary_raw and ("5000" in salary_raw or "5000" in salary_raw):
        salary_key = "above_5000"
    else:
        # fuzzy/fallback parsing
        if "below 4000" in salary_raw or salary_raw.strip() == "below 4000 aed" or salary_raw.strip() == "below 4000":
            salary_key = "below_4000"
        elif "4000 - 5000" in salary_raw or "4000-5000" in salary_raw or "4000 to 5000" in salary_raw:
            salary_key = "4000_5000"
        elif "above 5000" in salary_raw or "more than 5000" in salary_raw or "5000+" in salary_raw or "5000 =" in salary_raw:
            salary_key = "above_5000"

    # final fallback: if salary contains digits, try to infer numerically
    if salary_key is None and salary_raw:
        digits = re.findall(r'\d+', salary_raw)
        if digits:
            nums = [int(d) for d in digits]
            # take median-ish or first number
            n = nums[0]
            if n < 4000:
                salary_key = "below_4000"
            elif 4000 <= n <= 5000:
                salary_key = "4000_5000"
            elif n > 5000:
                salary_key = "above_5000"

    # Build products or message based on rules you requested
    products = []
    message = None

    if place == "Dubai":
        if salary_key == "below_4000":
            # single demo product
            products = [
                {"name": "DHA-Basic", "price": "864.00", "plan": "NLSB"}
            ]
        elif salary_key in ("4000_5000", "above_5000"):
            products = [
                {"name": "DHA-Basic", "price": "864.00", "plan": "NLSB"},
                {"name": "DHA-Basic", "price": "1893.00", "plan": "LSB"}
            ]
        else:
            # salary unknown — give conservative suggestion
            products = [
                {"name": "DHA-Basic", "price": "1893.00", "plan": "LSB"}
            ]

    elif place == "Abu Dhabi":
        if salary_key in ("below_4000", "4000_5000"):
            message = "The minimum requirement for this product plan is above 5000 AED"
        elif salary_key == "above_5000":
            products = [
                {"name": "Abu Dhabi Eligible Plan", "price": "1350.00 AED", "plan": "Premium"}
            ]
        else:
            message = "Please ensure your salary is above 5000 AED to see available plans for Abu Dhabi."

    else:
        # unknown issuing place -> be conservative
        if salary_key == "above_5000":
            products = [
                {"name": "General Plan (Eligible)", "price": "999.00 AED", "plan": "Standard"}
            ]
        else:
            message = "We couldn't determine issuing place precisely. If you'd like, please confirm the issuing place (Dubai or Abu Dhabi)."

    ask_mobile = not bool(session.mobile)

    # In the emirates_id_upload function, update the product URL generation:
    # In the emirates_id_upload function, update the product URL generation:
    for p in products:
        # Ensure URL is always a string and properly formatted
        if "DHA-Basic" in p["name"]:
            if p["plan"] == "NLSB":
                p["url"] = "https://gia-insurance-provider.com/dha-basic-nlsb"
            elif p["plan"] == "LSB":
                p["url"] = "https://gia-insurance-provider.com/dha-basic-lsb"
            else:
                p["url"] = "https://gia-insurance-provider.com/dha-basic"
        elif "Abu Dhabi" in p["name"]:
            p["url"] = "https://gia-insurance-provider.com/abu-dhabi-premium"
        else:
            p["url"] = "https://gia-insurance-provider.com/general-plan"
        
        # Double-check URL format
        if not p["url"].startswith(('http://', 'https://')):
            p["url"] = "https://" + p["url"]

    # --- Build response (include new keys) ---
    resp_payload = {
        "ok": True,
        "id": record.id,
        "fields": {**front_data, **back_data},
        "missing_fields": missing,
        "sides_detected": {
            "front": bool(front_texts),
            "back": bool(back_texts)
        },
        "next_step": next_step,
        "files_processed": len(processed_files),
        # New keys consumed by front-end
        "issuing_place_detected": place,
        "products": products,
        "message": message,
        "ask_mobile": ask_mobile,
    }

    return JsonResponse(resp_payload)






from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import uuid
import json
from .models import ChatSession


@csrf_exempt
def insurance_chat(request):
    """
    Dynamic OpenAI-driven conversational flow for medical-insurance onboarding.
    """
    # ---------- GET: fetch chat history ----------
    if request.method == "GET":
        session_id = request.GET.get("session_id")
        if not session_id:
            return JsonResponse({"error": "session_id required"}, status=400)

        try:
            session = ChatSession.objects.get(session_id=session_id)
        except ChatSession.DoesNotExist:
            return JsonResponse({"error": "Session not found"}, status=404)

        messages = ChatMessage.objects.filter(session=session).order_by("created_at")
        history = [
            {"role": m.role, "content": m.content, "created_at": m.created_at}
            for m in messages
        ]
        return JsonResponse({"session_id": session.session_id, "messages": history})

    # ---------- POST: handle new message ----------
    if request.method == "POST":
        try:
            payload = json.loads(request.body.decode("utf-8"))
        except Exception:
            return JsonResponse({"error": "Malformed JSON"}, status=400)

        user_text = payload.get("user_text", "").strip()
        session_id = payload.get("session_id") or str(uuid.uuid4())
        user = request.user if request.user.is_authenticated else None

        # Get or create session
        try:
            session = ChatSession.objects.get(session_id=session_id)
            if user and session.user is None:  # attach user if logged in later
                session.user = user
                session.save(update_fields=["user"])
        except ChatSession.DoesNotExist:
            session = ChatSession.objects.create(
                session_id=session_id,
                user=user
            )

        # Prepare context for OpenAI
        context = {
            "step": session.step or "start",
            "looking_for_insurance": session.looking_for_insurance,
            "role": session.role,
            "salary": session.salary,
            "depender_type": session.depender_type,
            "is_completed": session.is_completed,
        }

        # Add Emirates ID data if available
        emirates_id_data = {}
        try:
            if hasattr(session, 'emirates_id_record'):
                emr = session.emirates_id_record
                emirates_id_data = {
                    "full_name": emr.name,
                    "emirates_id_number": emr.emirates_id,
                    "dob": emr.dob,
                    "expiry": emr.expiry_date,
                    "nationality": emr.nationality,
                    "occupation": emr.occupation,
                    "gender": emr.gender,
                    "address": emr.address,
                    "employer": emr.employer,
                    "issuing_place": emr.issuing_place,
                    "family_sponsor": emr.family_sponsor,
                }
        except EmiratesIDRecord.DoesNotExist:
            pass
        
        context["emirates_id_data"] = emirates_id_data

        if session.step == "passport_verified":
            try:
                record = EmiratesIDRecord.objects.filter(chat_session=session).first()
                products, message = compute_products_based_on_data(session, record)
                
                # Save message if user sent text
                if user_text:
                    ChatMessage.objects.create(session=session, role="user", content=user_text)
                
                # Prepare product response
                reply = "Based on your Emirates ID and salary, here are the available products:"
                if message:
                    reply = message
                
                # If products found, return them
                if products and len(products) > 0:
                    return JsonResponse({
                        "reply": reply,
                        "options": [],
                        "session_id": session.session_id,
                        "step": session.step,
                        "products": products
                    })
                else:
                    # No products available
                    return JsonResponse({
                        "reply": reply or "No products available at this time.",
                        "options": [],
                        "session_id": session.session_id,
                        "step": session.step,
                        "products": []
                    })
            except Exception as e:
                print(f"Error computing products after passport verification: {e}")
                # Fall through to normal flow if error

        # Prepare messages for OpenAI
        messages_for_openai = [
            {"role": "system", "content": INSURANCE_ONBOARDING_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({
                "context": context,
                "last_user_message": user_text
            })}
        ]

        try:
            # ---------- Call DeepSeek ----------
            reply_content = deepseek_chat(
                messages=messages_for_openai,  # message format is compatible
                max_tokens=500,
            )

            # ---------- Parse JSON response ----------
            try:
                ai_response = json.loads(reply_content)
            except json.JSONDecodeError:
                # DeepSeek returned non-JSON, fallback to FSM
                print("DeepSeek returned invalid JSON, falling back to FSM")
                return fallback_to_fsm(session, user_text, user)

            # ---------- Extract AI response fields ----------
            reply = ai_response.get("reply", "")
            options = ai_response.get("options", [])
            session_updates = ai_response.get("session_updates", {})
            complete = ai_response.get("complete", False)

            # ---------- Apply session updates ----------
            for field, value in session_updates.items():
                if hasattr(session, field):
                    setattr(session, field, value)

            # ---------- Handle completion ----------
            if complete:
                session.is_completed = True
                session.step = "complete"

            session.save()

        except Exception as e:
            # ---------- Hard fallback (DeepSeek error, timeout, network, etc.) ----------
            print(f"DeepSeek error: {e}, falling back to FSM")
            return fallback_to_fsm(session, user_text, user)

        # ---------- Save messages ----------
        if user_text:
            ChatMessage.objects.create(
                session=session,
                role="user",
                content=user_text
            )

        if reply:
            ChatMessage.objects.create(
                session=session,
                role="bot",
                content=reply
            )

        # ---------- Final response ----------
        return JsonResponse({
            "reply": reply,
            "options": options,
            "session_id": session.session_id,
            "step": session.step
        })


    return JsonResponse({"error": "Method not allowed"}, status=405)


# Add this fallback function to maintain the original FSM as backup
def fallback_to_fsm(session, user_text, user):
    """
    Fallback to the original FSM flow if OpenAI fails
    """
    reply, options = "", []
    step = session.step or "start"

    # ---------- Original FSM logic (keep as backup) ----------
    if step == "start":
        reply = "Are you looking for medical insurance?"
        options = ["Yes", "No"]
        session.step = "q1"

    elif step == "q1":
        if user_text.lower().startswith("y"):
            if not user or not user.is_authenticated:
                return JsonResponse(
                    {"error": "Authentication required", "login_required": True, "session_id": session.session_id},
                    status=401
                )
            reply = "Are you an employee or a depender?"
            options = ["Employee", "Depender"]
            session.looking_for_insurance = "Yes"
            session.step = "q2"
        else:
            reply = "Okay. Let me know if you need help later."
            session.looking_for_insurance = "No"
            session.is_completed = True

    elif step == "q2":
        text = user_text.lower()
        if text in ["e", "employee"]:
            session.role = "Employee"
            reply = "What is your current monthly salary?"
            options = ["below 4000 AED", "4000 - 5000 AED", "above 5000 AED"]
            session.step = "q3"
        elif text in ["d", "depender"]:
            session.role = "Depender"
            reply = "What type of depender are you?"
            options = ["Spouse", "Child"]
            session.step = "q2a"
        else:
            reply = "Please select Employee or Depender."
            options = ["Employee", "Depender"]

    elif step == "q2a":
        text = user_text.lower()
        if text in ["s", "spouse"]:
            session.depender_type = "Spouse"
            reply = "What is your sponsor's current monthly salary?"
            options = ["below 4000 AED", "4000 - 5000 AED", "above 5000 AED"]
            session.step = "q3"
        elif text in ["c", "child"]:
            session.depender_type = "Child"
            reply = "What is your sponsor's current monthly salary?"
            options = ["below 4000 AED", "4000 - 5000 AED", "above 5000 AED"]
            session.step = "q3"
        else:
            reply = "Please select Spouse or Child."
            options = ["Spouse", "Child"]

    elif step == "q3":
        session.salary = user_text
        reply = "Please upload the insured person's valid Emirates ID."
        session.step = "awaiting_id_frontside"

    elif step == "awaiting_id_frontside":
        reply = "I see you're trying to upload an Emirates ID. Please use the file upload button to submit the front side of your Emirates ID."

    elif step == "awaiting_id_backside":
        reply = "I see you're trying to upload an Emirates ID. Please use the file upload button to submit the back side of your Emirates ID."

    elif step == "passport_verified":
        # After passport verification, compute and return products
        try:
            record = EmiratesIDRecord.objects.filter(chat_session=session).first()
            products, message = compute_products_based_on_data(session, record)
            
            reply = "Based on your Emirates ID and salary, here are the available products:"
            if message:
                reply = message
            options = []
            
            # Return products in response (will be handled by caller)
            session.save()
            if user_text:
                ChatMessage.objects.create(session=session, role="user", content=user_text)
            if reply:
                ChatMessage.objects.create(session=session, role="bot", content=reply)
            
            return JsonResponse({
                "reply": reply,
                "options": options,
                "session_id": session.session_id,
                "step": session.step,
                "products": products or []
            })
        except Exception as e:
            reply = f"Sorry, there was an error fetching products. Error: {str(e)}"
            options = []

    elif step == "complete":
        # Free chat mode with original SYSTEM_PROMPT
        try:
            resp = client.chat.completions.create(
                model="openai",
                messages=[
                    {"role": "system", "content": INSURANCE_ONBOARDING_SYSTEM_PROMPT},
                    {"role": "user", "content": user_text}
                ],
                max_tokens=500,

            )
            # reply = resp.choices[0].message.content.strip()
            reply = deepseek_chat(
                messages=[
                    {"role": "system", "content": INSURANCE_ONBOARDING_SYSTEM_PROMPT},
                    {"role": "user", "content": user_text},
                ]
            )

        except Exception as e:
            reply = f"Sorry, I'm having trouble right now. Error: {str(e)}"

    else:  # Unknown step → reset
        reply = "I seem to have lost track. Let's start over. Are you looking for medical insurance?"
        options = ["Yes", "No"]
        session.step = "start"

    session.save()

    # Save messages
    if user_text:
        ChatMessage.objects.create(session=session, role="user", content=user_text)
    if reply:
        ChatMessage.objects.create(session=session, role="bot", content=reply)

    return JsonResponse({
        "reply": reply,
        "options": options,
        "session_id": session.session_id,
        "step": session.step
    })



@csrf_exempt
@require_POST
def update_emirates_id_record(request):
    import json
    payload = json.loads(request.body.decode("utf-8"))
    rec_id = payload.get("id")
    field  = (payload.get("field") or "").strip().lower()
    value  = (payload.get("value") or "").strip()

    if not rec_id or not field:
        return JsonResponse({"error": "id and field required"}, status=400)

    rec = EmiratesIDRecord.objects.filter(id=rec_id).first()
    if not rec:
        return JsonResponse({"error": "record not found"}, status=404)

    # Normalize field name - handle both 'sex' and 'gender'
    if field in ("sex", "gender"):
        v = value.lower()
        if v in ("m", "male", "ذكر"):
            canon = "Male"
        elif v in ("f", "female", "أنثى"):
            canon = "Female"
        else:
            canon = value.title() if value else ""

        rec.gender = canon
        if hasattr(rec, 'sex'):
            rec.sex = canon
            
        rec.save(update_fields=["gender", "sex"] if hasattr(rec, 'sex') else ["gender"])
        
        ack = f"Thank you, gender saved as {canon}." if canon else "Gender information saved."
        return JsonResponse({"ok": True, "message": ack, "fields": {"gender": rec.gender}})

    allowed = {
        "emirates_id", "name", "address", "expiry_date", "dob", "issuing_date", 
        "nationality", "occupation", "employer", "issuing_place", "family_sponsor",
        "family_sponsor_name"  # ADD THIS
    }
    if field not in allowed:
        return JsonResponse({"error": "field not allowed"}, status=400)

    setattr(rec, field, value)
    rec.save(update_fields=[field])
    return JsonResponse({"ok": True, "message": "Saved", "fields": {field: value}})


@csrf_exempt
def get_user_session_data(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    try:
        payload = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    session_id = payload.get("session_id")
    if not session_id:
        return JsonResponse({"error": "session_id required"}, status=400)

    # ---------- look up by session_id, NOT user_id ----------
    try:
        chat_session = ChatSession.objects.get(session_id=session_id)
    except ChatSession.DoesNotExist:
        return JsonResponse({"error": "Session not found"}, status=404)

    # ---------- build response ----------
    response = {
        "chat_session": {
            "id": chat_session.id,
            "uuid": str(chat_session.uuid),
            "session_id": chat_session.session_id,
            "step": chat_session.step,
            "is_completed": chat_session.is_completed,
            "looking_for_insurance": chat_session.looking_for_insurance,
            "role": chat_session.role,
            "salary": chat_session.salary,
            "depender_type": chat_session.depender_type,
            "created_at": chat_session.created_at.isoformat(),
            "updated_at": chat_session.updated_at.isoformat(),
        }
    }

    # optional emirates record
    if hasattr(chat_session, "emirates_id_record"):
        emr = chat_session.emirates_id_record
        response["emirates_record"] = {
            "id": emr.id,
            "uuid": str(emr.uuid),
            "emirates_id": emr.emirates_id,
            "name": emr.name,
            "dob": emr.dob,
            "issuing_date": emr.issuing_date,
            "issuing_place": emr.issuing_place,
            "nationality": emr.nationality,
            "gender": emr.gender,
            "address": emr.address,
            "expiry_date": emr.expiry_date,
            "created_at": emr.created_at.isoformat(),
            "updated_at": emr.updated_at.isoformat(),
        }

    return JsonResponse(response)
    

def parse_back_side_fields(text):
    import re
    data = {}
    raw = text or ""
    
    # Occupation
    occupation_patterns = [
        r'(?i)(?:Occupation|المهنة|Profession|الوظيفة)\s*[:\-]?\s*([^\n\r]{2,80})',
        r'(?i)(?:Occupation|المهنة|Profession|الوظيفة)[\s\S]{1,100}?([^\n\r]{2,80})(?=\n|$)'
    ]
    for pattern in occupation_patterns:
        m = re.search(pattern, raw)
        if m:
            occupation = m.group(1).strip()
            # Clean up common OCR artifacts
            occupation = re.sub(r'[^\w\s\-]', '', occupation)
            if len(occupation) > 1:
                data['occupation'] = occupation
                break
    
    # Employer
    employer_patterns = [
        r'(?i)(?:Employer|صاحب العمل|Company|الشركة|جهة العمل)\s*[:\-]?\s*([^\n\r]{2,80})',
        r'(?i)(?:Employer|صاحب العمل|Company|الشركة|جهة العمل)[\s\S]{1,100}?([^\n\r]{2,80})(?=\n|$)'
    ]
    for pattern in employer_patterns:
        m = re.search(pattern, raw)
        if m:
            employer = m.group(1).strip()
            employer = re.sub(r'[^\w\s\-]', '', employer)
            if len(employer) > 1:
                data['employer'] = employer
                break
    
    # Issuing Place
    issuing_place_patterns = [
        r'(?i)(?:Issuing Place|مكان الإصدار|Place of Issue|جهة الإصدار)\s*[:\-]?\s*([^\n\r]{2,80})',
        r'(?i)(?:Issuing Place|مكان الإصدار|Place of Issue|جهة الإصدار)[\s\S]{1,100}?([^\n\r]{2,80})(?=\n|$)'
    ]
    for pattern in issuing_place_patterns:
        m = re.search(pattern, raw)
        if m:
            issuing_place = m.group(1).strip()
            issuing_place = re.sub(r'[^\w\s\-]', '', issuing_place)
            if len(issuing_place) > 1:
                data['issuing_place'] = issuing_place
                break
    
    # Family Sponsor (Yes/No)
    family_sponsor_patterns = [
        r'(?i)(?:Family Sponsor|كفالة عائلية|Sponsor|كفالة|الکفالة)\s*[:\-]?\s*(Yes|No|نعم|لا|Y|N)',
        r'(?i)(?:Family Sponsor|كفالة عائلية|Sponsor|كفالة|الکفالة)[\s\S]{1,50}?(Yes|No|نعم|لا|Y|N)'
    ]
    for pattern in family_sponsor_patterns:
        m = re.search(pattern, raw)
        if m:
            value = m.group(1).strip().lower()
            if value in ['yes', 'y', 'نعم']:
                data['family_sponsor'] = 'Yes'
            elif value in ['no', 'n', 'لا']:
                data['family_sponsor'] = 'No'
            break
    
    return data


@csrf_exempt
def check_session_status(request):
    """Check if session needs backside upload"""
    import json
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)
    
    payload = json.loads(request.body.decode("utf-8"))
    session_id = payload.get("session_id")
    
    if not session_id:
        return JsonResponse({"error": "session_id required"}, status=400)
    
    try:
        chat_session = ChatSession.objects.get(user_id=session_id)
        needs_backside = chat_session.step == "awaiting_id_backside"
        
        return JsonResponse({
            "needs_backside": needs_backside,
            "step": chat_session.step
        })
        
    except ChatSession.DoesNotExist:
        return JsonResponse({"error": "Session not found"}, status=404)
    

    
from django.contrib.auth import login, authenticate
from rest_framework.authtoken.models import Token
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from django.core.mail import send_mail
from .models import User, OTP


@csrf_exempt
@api_view(['POST'])
@permission_classes([AllowAny])
def send_otp(request):
    email = request.data.get('email')
    
    if not email:
        return Response({'error': 'Email is required'}, status=400)
    
    try:
        user, created = User.objects.get_or_create(
            email=email,
            defaults={'username': email}
        )
        
        # Generate OTP
        otp = OTP.generate_otp(user)
        
        # Send OTP via email
        send_mail(
            'Your OTP Code',
            f'Your OTP code is: {otp.otp_code}. It will expire in 10 minutes.',
            'amalikka0@gmail.com',
            [email],
            fail_silently=False,
        )
        
        return Response({
            'message': 'OTP sent successfully',
            'user_id': user.id,
            'is_new_user': created
        })
        
    except Exception as e:
        return Response({'error': str(e)}, status=500)


@csrf_exempt
@api_view(['POST'])
@permission_classes([AllowAny])
def verify_otp(request):
    email = request.data.get('email')
    otp_code = request.data.get('otp_code')
    
    if not email or not otp_code:
        return Response({'error': 'Email and OTP code are required'}, status=400)
    
    try:
        user = User.objects.get(email=email)
        otp = OTP.objects.filter(user=user, otp_code=otp_code, is_used=False).first()
        
        if not otp or not otp.is_valid():
            return Response({'error': 'Invalid or expired OTP'}, status=400)
        
        # Mark OTP as used
        otp.is_used = True
        otp.save()
        
        # Update user verification status
        user.email_verified = True
        user.save()
        
        # Login the user
        login(request, user)
        
        # Create or get authentication token
        token, created = Token.objects.get_or_create(user=user)
        
        return Response({
            'message': 'OTP verified successfully',
            'token': token.key,
            'user_id': user.id,
            'session_id': str(uuid.uuid4()) 
        })
        
    except User.DoesNotExist:
        return Response({'error': 'User not found'}, status=404)
    except Exception as e:
        return Response({'error': str(e)}, status=500)

@csrf_exempt
@api_view(['GET'])
def get_user_chat_history(request, session_id=None):
    user = request.user
    if not user.is_authenticated:
        return Response({'error': 'Authentication required'}, status=401)
    
    # Get all chat sessions for this user
    chat_sessions = ChatSession.objects.filter(user=user).order_by('-created_at')
    
    sessions_data = []
    for session in chat_sessions:
        sessions_data.append({
            'session_id': session.session_id,
            'step': session.step,
            'created_at': session.created_at,
            'is_completed': session.is_completed,
            'emirates_id': session.emirates_id_record.emirates_id if hasattr(session, 'emirates_id_record') else None
        })
    
    return Response({'chat_sessions': sessions_data})



@csrf_exempt
def save_mobile(request):
    import json
    from django.http import JsonResponse
    from .models import ChatSession, EmiratesIDRecord

    data = json.loads(request.body.decode("utf-8"))
    session_id = data.get("session_id")
    mobile = data.get("mobile")

    if not session_id or not mobile:
        return JsonResponse({"error": "session_id and mobile required"}, status=400)

    try:
        session = ChatSession.objects.get(session_id=session_id)
    except ChatSession.DoesNotExist:
        return JsonResponse({"error": "Session not found"}, status=404)

    # Save the mobile number
    session.mobile = mobile
    session.save(update_fields=["mobile"])

    # ✅ Get related Emirates ID record
    record = EmiratesIDRecord.objects.filter(chat_session=session).first()

    products, message = compute_products_based_on_data(session, record)

    response = {
        "ok": True,
        "message": f"Thanks! I've updated your mobile number to {mobile}.",
        "products": products or [],
        "info_message": message or None,
    }

    return JsonResponse(response)



from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .models import EmiratesIDRecord, PassportRecord 

@csrf_exempt
def save_missing_field(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request"}, status=405)

    try:
        data = json.loads(request.body.decode("utf-8"))
        record_id = data.get("record_id")
        field_name = data.get("field_name")
        value = data.get("value")

        if not record_id or not field_name or not value:
            return JsonResponse({"error": "Missing data"}, status=400)

        record = EmiratesIDRecord.objects.get(id=record_id)
        setattr(record, field_name, value)
        record.save()

        # Check if there are still missing fields
        missing_fields = []
        for f in ["mobile", "salary", "family_sponsor_name"]:  # adjust to your model
            if not getattr(record, f, None):
                missing_fields.append(f)

        # If all fields filled, return products (or empty if none)
        products = []
        if not missing_fields:
            products = [
                {"name": "Standard Plan", "price": "AED 599", "plan": "Basic"},
                {"name": "Gold Plan", "price": "AED 899", "plan": "Premium"},
            ]

        return JsonResponse({
            "ok": True,
            "next_missing_field": missing_fields[0] if missing_fields else None,
            "products": products
        })

    except EmiratesIDRecord.DoesNotExist:
        return JsonResponse({"error": "Record not found"}, status=404)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


import re

def compute_products_based_on_data(session, record):
    # Normalize issuing place (robust to common OCR errors)
    issuing_raw = (record.issuing_place or "") if record else ""
    issuing_norm = issuing_raw.strip().lower()

    # canonicalize common OCR variations
    place = None
    if issuing_norm:
        if ("abu" in issuing_norm and "dub" not in issuing_norm) or "adhabi" in issuing_norm or "adabi" in issuing_norm or "abudhabi" in issuing_norm or "abud" in issuing_norm:
            place = "Abu Dhabi"
        elif "duba" in issuing_norm or "dubai" in issuing_norm or "dubi" in issuing_norm:
            place = "Dubai"
        else:
            if "abu dhabi" in issuing_norm:
                place = "Abu Dhabi"
            elif "dubai" in issuing_norm:
                place = "Dubai"
            else:
                place = issuing_raw.strip().title() if issuing_raw else None

    # Normalize salary selection from session
    salary_raw = (session.salary or "").lower() if session else ""
    salary_key = None

    if "below" in salary_raw and "4000" in salary_raw:
        salary_key = "below_4000"
    elif "4000" in salary_raw and "5000" in salary_raw:
        salary_key = "4000_5000"
    elif "above" in salary_raw and ("5000" in salary_raw or "5000" in salary_raw):
        salary_key = "above_5000"
    else:
        if "below 4000" in salary_raw or salary_raw.strip() == "below 4000 aed" or salary_raw.strip() == "below 4000":
            salary_key = "below_4000"
        elif "4000 - 5000" in salary_raw or "4000-5000" in salary_raw or "4000 to 5000" in salary_raw:
            salary_key = "4000_5000"
        elif "above 5000" in salary_raw or "more than 5000" in salary_raw or "5000+" in salary_raw or "5000 =" in salary_raw:
            salary_key = "above_5000"

    # final fallback: numeric inference
    if salary_key is None and salary_raw:
        digits = re.findall(r'\d+', salary_raw)
        if digits:
            nums = [int(d) for d in digits]
            n = nums[0]
            if n < 4000:
                salary_key = "below_4000"
            elif 4000 <= n <= 5000:
                salary_key = "4000_5000"
            elif n > 5000:
                salary_key = "above_5000"

    # Build products or message
    products = []
    message = None

    if place == "Dubai":
        if salary_key == "below_4000":
            products = [
                {"name": "DHA-Basic", "price": "864.00", "plan": "NLSB", "url": "https://gia-insurance-provider.com/dha-basic-nlsb"}
            ]
        elif salary_key in ("4000_5000", "above_5000"):
            products = [
                {"name": "DHA-Basic", "price": "864.00", "plan": "NLSB", "url": "https://gia-insurance-provider.com/dha-basic-nlsb"},
                {"name": "DHA-Basic", "price": "1893.00", "plan": "LSB", "url": "https://gia-insurance-provider.com/dha-basic-lsb"}
            ]
        else:
            products = [
                {"name": "DHA-Basic", "price": "1893.00", "plan": "LSB", "url": "https://gia-insurance-provider.com/dha-basic-lsb"}
            ]

    elif place == "Abu Dhabi":
        if salary_key in ("below_4000", "4000_5000"):
            message = "The minimum requirement for this product plan is above 5000 AED"
        elif salary_key == "above_5000":
            products = [
                {"name": "Abu Dhabi Eligible Plan", "price": "1350.00 AED", "plan": "Premium", "url": "https://gia-insurance-provider.com/abu-dhabi-premium"}
            ]
        else:
            message = "Please ensure your salary is above 5000 AED to see available plans for Abu Dhabi."

    else:
        if salary_key == "above_5000":
            products = [
                {"name": "General Plan (Eligible)", "price": "999.00 AED", "plan": "Standard", "url": "https://gia-insurance-provider.com/general-plan"}
            ]
        else:
            message = "We couldn't determine issuing place precisely. If you'd like, please confirm the issuing place (Dubai or Abu Dhabi)."

    return products, message



# -------------------- Mindee Passport Upload (Testing Version) --------
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.http import JsonResponse
from mindee import ClientV2, InferenceParameters, BytesInput
import re

from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from django.http import JsonResponse

@csrf_exempt
@api_view(["POST"])
@permission_classes([AllowAny])
def passport_upload(request):
    
    try:
        uploaded = request.FILES.get("file")
        if not uploaded:
            return JsonResponse({"error": "No file uploaded"}, status=400)

        # Prefer env, fall back to your test keys so nothing breaks while you tinker
        api_key = os.getenv("MINDEE_API_KEY", "md_bqhpQSjYVJUmdydlOHoc9TBDEbqkfuSs")
        model_id = os.getenv("MINDEE_PASSPORT_MODEL_ID", "a534a5a4-47ca-481e-924e-c9ea69f693dd")

        client = ClientV2(api_key)
        params = InferenceParameters(model_id=model_id, confidence=True)

        # Read uploaded file bytes
        input_source = BytesInput(uploaded.read(), filename=uploaded.name)

        # OCR
        response = client.enqueue_and_get_inference(input_source, params)
        fields = response.inference.result.fields

        # Helpers -------------------------------------------------------------

        def value_of(field_obj):
            """
            Mindee returns either raw strings or objects with .value.
            Return a clean string or None.
            """
            if field_obj is None:
                return None
            v = getattr(field_obj, "value", field_obj)
            if v is None:
                return None
            s = str(v).strip()
            return s if s else None

        def get_any(keys):
            for k in keys:
                if k in fields:
                    v = value_of(fields[k])
                    if v:
                        return v
            return None

        def normalize_date(s):
            if not s:
                return None
            s = str(s).strip()
            # Mindee commonly returns YYYY-MM-DD
            m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", s)
            if m:
                y, mm, dd = m.groups()
                return f"{dd}/{mm}/{y}"
            return s

        def _collapse_spaces(txt):
            return re.sub(r"\s+", " ", txt).strip()

        def parse_names_from_mrz(mrz_line_1):
            """
            MRZ line 1 format (with OCR glitches):
            'P<ISSUER[SURNAME]<<GIVEN<NAMES<<<<'
            Real world often has '<' misread as 'A', so handle that too.
            Returns (surname, given_names) or (None, None).
            """
            if not mrz_line_1:
                return None, None

            line = mrz_line_1.upper()
            # Normalize common OCR misreads of the filler char
            line = line.replace("«", "<")
            
            line_no_hdr = re.sub(r"^P[<A][A-Z]{3}", "", line)

            # Split surname and given names by the primary separator
            parts = line_no_hdr.split("<<", 1)
            if len(parts) != 2:
                return None, None

            raw_surname, raw_given = parts[0], parts[1]

            surname = _collapse_spaces(raw_surname.replace("<", " "))
            given = _collapse_spaces(raw_given.split("<<", 1)[0].replace("<", " "))

            return (surname or None), (given or None)

        # --------------------------------------------------------------------

        passport_number = get_any(["passport_number", "PassportNumber", "Document Number"])
        dob = normalize_date(get_any(["date_of_birth", "Date of Birth", "DOB"]))

        # Prefer Mindee's plural keys first, then common alternates
        given_name = get_any([
            "given_names", "given_name", "Given Names", "Given Name",
            "first_names", "first_name", "forenames", "forename", "given"
        ])
        surname = get_any([
            "surnames", "surname", "Surname", "Last Name",
            "family_name", "family_names", "lastname"
        ])

        # Fallback via MRZ if any name is missing
        if not given_name or not surname:
            mrz1 = get_any(["mrz_line_1", "mrz1", "MRZ Line 1"])
            parsed_surname, parsed_given = parse_names_from_mrz(mrz1)
            surname = surname or parsed_surname
            given_name = given_name or parsed_given

        full_name = _collapse_spaces(f"{given_name or ''} {surname or ''}")

        return JsonResponse({
            "fields": {
                "passport_number": passport_number,
                "given_name": given_name, 
                "surname": surname,
                "name": full_name,
                "dob": dob
            },
            "validation": {
                "is_valid": all([passport_number, full_name, dob]),
                "validation_message": "Successfully extracted passport details"
            }
        }, status=200)

    except Exception as e:
        # Do not leak stack traces to clients
        return JsonResponse({"error": f"OCR failed: {str(e)}"}, status=500)



from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.http import JsonResponse
from .ocr_space import ocr_space_file_multi_lang, ocr_space_pdf_all_pages
from .models import UAEDocumentVisa
import re

import re

def parse_uae_visa_fields(text: str):

    data = {}
    if not text:
        return data

    raw = text
    t = re.sub(r'[ \t]+', ' ', raw).replace('\r', '\n')

    def clean(v):
        v = v.strip()
        v = re.sub(r'^[\:\-\|\/\s]+', '', v)
        v = re.sub(r'[\s\|]+$', '', v)
        return v.strip()

    # ID Number
    id_match = re.search(r'(?i)(?:ID\s*Number|ID\s*No\.?|رقم\s*الهوية)\s*[:\-]?\s*([0-9\-]{10,25})', t)
    if not id_match:
        id_match = re.search(r'\b(784\d{9,12})\b', t)
    if id_match:
        data['id_number'] = clean(id_match.group(1))

    # File Number
    file_match = re.search(r'(?i)(?:File\s*Number|File\s*No\.?|رقم\s*الملف)\s*[:\-]?\s*([0-9\/\-]{8,25})', t)
    if not file_match:
        file_match = re.search(r'(\b\d{3,5}\/\d{4}\/\d{5,8}\b)', t)
    if file_match:
        data['file_number'] = clean(file_match.group(1))

    # ---------------- Passport Number ----------------
    pass_match = re.search(
        r'(?i)(?:Passport\s*No\.?|Passport\s*Number|رقم\s*الجواز)\s*[:\-]?\s*((?=[A-Z0-9]*\d)[A-Z0-9]{5,12})',
        t
    )
    passport_val = None
    if pass_match:
        passport_val = clean(pass_match.group(1))
    else:
        # Fallback: any token 6–12 chars with letters+digits
        candidates = re.findall(r'\b(?=[A-Z0-9]*\d)[A-Z0-9]{6,12}\b', t)
        for val in candidates:
            if re.match(r'^[A-Z][A-Z0-9]*\d', val):
                passport_val = val
                break
        if not passport_val and candidates:
            passport_val = candidates[0]

    if passport_val:
        data['passport_no'] = passport_val

    # ---------------- UID No (if present) ----------------
    uid_match = re.search(r'(?i)(?:UID\s*(?:No\.?|Number)?|Unified\s*Number|الرقم\s*الموحد)\s*[:\-]?\s*([0-9]{8,15})', t)
    if uid_match:
        data['uid_no'] = clean(uid_match.group(1))

    # ---------------- Name (Visa Holder) ----------------
    
    name_val = None
    
    # Pattern 1: Name appears after passport number line, before profession keywords
    # Match 2-4 uppercase words that form a person's name
    name_match = re.search(
        r'(?i)(?:Passport\s*No\.?|رقم\s*الجواز)[^\n]*\n\s*([A-Z][A-Z\s]{8,60}?)\s*(?=\n|HOUSE\s*WIFE|Profession|Employer|المهنة|الوظيفة)',
        t
    )
    
    if name_match:
        name_val = clean(name_match.group(1))
    
    # Pattern 2: Look for Arabic name pattern followed by English name
    if not name_val:
        name_match = re.search(
            r'[\u0600-\u06FF\s]+\n\s*([A-Z][A-Z\s]{8,60}?)\s*(?=\n|HOUSE\s*WIFE|Profession)',
            t
        )
        if name_match:
            name_val = clean(name_match.group(1))
    
    # Pattern 3: Find uppercase name before profession keywords
    if not name_val:
        name_match = re.search(
            r'\b([A-Z]{3,}(?:\s+[A-Z]{3,}){1,3})\s*(?=\n|HOUSE\s*WIFE|Profession|Employer)',
            t
        )
        if name_match:
            name_val = clean(name_match.group(1))
    
    # Clean and validate the extracted name
    if name_val:
        # Remove any trailing junk
        name_val = re.sub(r'\s*(ame|Profession|HOUSE\s*WIFE|Employer).*$', '', name_val, flags=re.IGNORECASE)
        name_val = clean(name_val)
        
        # Validate: should be 2-4 words, each at least 2 chars, no common false positives
        words = name_val.split()
        excluded_words = {'UNITED', 'ARAB', 'EMIRATES', 'HOUSE', 'WIFE', 'PROFESSION', 'EMPLOYER', 'PASSPORT', 'NUMBER'}
        
        if (2 <= len(words) <= 4 and 
            all(len(w) >= 2 for w in words) and 
            not any(w.upper() in excluded_words for w in words)):
            data['employer_name'] = name_val




    # ---------------- Dates: Issue / Expiry ----------------
    date_rx = r'(\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2}|\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})'

    def normalize_date(s: str) -> str:
        s = clean(s).replace('-', '/')
        parts = s.split('/')
        if len(parts) == 3:
            if len(parts[0]) == 4:   
                y, m, d = parts
            else:                    
                d, m, y = parts
                if len(y) == 2:
                    y = f"{2000+int(y):04d}" if int(y) < 30 else f"{1900+int(y):04d}"
            try:
                return f"{int(y):04d}/{int(m):02d}/{int(d):02d}"
            except:
                return s
        return s

    # labeled first, English + Arabic, allow a bit of junk between label and digits
    issue_match = re.search(r'(?i)(?:Issu(?:e|ing)\s*Date|Date\s*of\s*Issue|تاريخ\s*الإصدار)[^\d]{0,12}' + date_rx, t)
    exp_match   = re.search(r'(?i)(?:Expiry\s*Date|Expires|تاريخ\s*الانتهاء)[^\d]{0,12}' + date_rx, t)

    if issue_match:
        data['issuing_date'] = normalize_date(issue_match.group(1))
    if exp_match:
        data['expiry_date'] = normalize_date(exp_match.group(1))

    # fallback: pick earliest and latest reasonable dates in the text
    if 'issuing_date' not in data or 'expiry_date' not in data:
        all_dates = [normalize_date(d) for d in re.findall(date_rx, t)]
        def year_ok(ds):
            try:
                y = int(ds.split('/')[0])
                return 2000 <= y <= 2100
            except:
                return False
        cand = sorted(set([d for d in all_dates if year_ok(d)]))
        if cand:
            data.setdefault('issuing_date', cand[0])
            data.setdefault('expiry_date', cand[-1])

    return data



from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from django.http import JsonResponse

@csrf_exempt
@api_view(["POST"])
@permission_classes([AllowAny])
def uae_visa_upload(request):
    """
    Handles UAE Visa uploads (image/pdf) using OCR.space API.
    """
    f = request.FILES.get('file')
    if not f:
        return JsonResponse({"error": "No file uploaded"}, status=400)

    name = (f.name or "").lower()
    is_pdf = name.endswith(".pdf") or (f.content_type == "application/pdf")

    # OCR process
    if is_pdf:
        text, code, err = ocr_space_pdf_all_pages(f)
    else:
        text, code, err = ocr_space_file_multi_lang(f, False)

    if code != 0:
        return JsonResponse({"error": f"OCR failed: {err or 'unknown error'}"}, status=500)

    fields = parse_uae_visa_fields(text or "")

    # Save record
    record = UAEDocumentVisa.objects.create(
        id_number=fields.get("id_number"),
        file_number=fields.get("file_number"),
        passport_no=fields.get("passport_no"),
        employer_name=fields.get("employer_name"),
        uid_no=fields.get("uid_no"),
        issuing_date=fields.get("issuing_date"),
        expiry_date=fields.get("expiry_date"),
        raw_text=text
    )

    return JsonResponse({
        "ok": True,
        "record_id": record.id,
        "fields": {
            "id_number": record.id_number,
            "file_number": record.file_number,
            "passport_no": record.passport_no,
            "employer_name": record.employer_name,
            "uid_no": record.uid_no,
            "issuing_date": record.issuing_date,
            "expiry_date": record.expiry_date,
        },
        "raw_text": record.raw_text[:2000]  # print trimmed full OCR result
    }, status=200)


from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from django.http import JsonResponse

@csrf_exempt
@api_view(["POST"])
@permission_classes([AllowAny])
def emirates_id_upload_test(request):
    """
    Global Emirates ID OCR test.
    Returns ONLY essential fields requested by the user.
    """
    from .ocr_space import ocr_space_file_multi_lang

    files = request.FILES.getlist("file")
    if not files:
        return JsonResponse({"error": "No files received"}, status=400)

    combined_text = ""
    for file_obj in files:
        is_pdf = file_obj.name.lower().endswith(".pdf")
        text, code, err = ocr_space_file_multi_lang(file_obj, is_pdf)

        if code != 0:
            return JsonResponse({"error": f"OCR failed: {err}"}, status=500)

        combined_text += " " + text

    # Extract data using your original logic
    front_fields = parse_fields(combined_text)

    try:
        back_fields = parse_back_side_fields(combined_text)
    except:
        back_fields = {}

    merged = {**front_fields, **back_fields}

    # FILTERED FINAL OUTPUT (Only fields you want)
    filtered = {
        "emirates_id": merged.get("emirates_id"),
        "name": merged.get("name"),
        "dob": merged.get("dob"),
        "expiry_date": merged.get("expiry_date"),
        "issuing_date": merged.get("issuing_date"),
        "nationality": merged.get("nationality"),
        "gender": merged.get("gender"),
        "occupation": merged.get("occupation"),
        "employer": merged.get("employer"),
        "issuing_place": merged.get("issuing_place"),
    }

    return JsonResponse({
        "ok": True,
        "fields": filtered,
    })
