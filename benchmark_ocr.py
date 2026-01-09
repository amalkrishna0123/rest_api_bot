import time
import io
import os
import sys

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
import django
django.setup()

from api.ocr_space import process_pdf_page_by_page, ocr_space_file_multi_lang
from django.core.files.uploadedfile import SimpleUploadedFile
from PIL import Image
import numpy as np
import fitz

def create_dummy_pdf(num_pages=2):
    """Creates a dummy PDF file for testing flow and overhead"""
    doc = fitz.open()
    for i in range(num_pages):
        page = doc.new_page()
        page.insert_text((50, 50), f"Emirates ID Side {i+1}")
        page.insert_text((50, 100), "Nationality: Indian")
        page.insert_text((50, 150), "784-1234-1234567-1")
    
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes

def run_benchmark():
    print("\n--- OCR PERFORMANCE BENCHMARK ---")
    
    # 1. Warm up (Initialize models)
    print("Warming up models...")
    dummy_pdf = create_dummy_pdf(num_pages=1)
    file_obj = SimpleUploadedFile("warmup.pdf", dummy_pdf, content_type="application/pdf")
    process_pdf_page_by_page(file_obj)
    print("Warmup complete.")

    # 2. Test 2-page PDF (Typical Emirates ID)
    pdf_content = create_dummy_pdf(num_pages=2)
    file_obj = SimpleUploadedFile("test_id.pdf", pdf_content, content_type="application/pdf")
    
    print("\nStarting 2-page PDF processing...")
    start_time = time.time()
    result = process_pdf_page_by_page(file_obj)
    elapsed = time.time() - start_time
    
    print(f"2-page processing took: {elapsed:.2f} seconds")
    print(f"Pages detected: {len(result)}")
    
    # 3. Test Image processing
    img = Image.new('RGB', (1000, 1000), color=(255, 255, 255))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_content = img_byte_arr.getvalue()
    
    img_file = SimpleUploadedFile("test_id.jpg", img_content, content_type="image/jpeg")
    
    print("\nStarting single image processing...")
    start_time = time.time()
    result_text, code, err = ocr_space_file_multi_lang(img_file, is_pdf=False)
    img_elapsed = time.time() - start_time
    
    print(f"Single image processing took: {img_elapsed:.2f} seconds")
    
    print("\n--- SUMMARY ---")
    print(f"PDF Benchmark: {elapsed:.2f}s")
    print(f"Image Benchmark: {img_elapsed:.2f}s")
    print(f"Is PDF time < 4s? {'✅ YES' if elapsed < 4.0 else '❌ NO (Limited by CPU/Env)'}")
    print("---------------------------------\n")

if __name__ == "__main__":
    run_benchmark()
