import time
import io
import os
import requests
from django.test import TestCase
from api.ocr_space import process_pdf_page_by_page, paddle_ocr_file
from django.core.files.uploadedfile import SimpleUploadedFile
from PIL import Image
import numpy as np

class OCRPerformanceBenchmark(TestCase):
    """
    Benchmark test to verify OCR performance improvements.
    Note: Real performance depends on the presence of a GPU.
    """
    
    def create_dummy_pdf(self, num_pages=2):
        """Creates a dummy PDF file for testing flow and overhead"""
        import fitz
        doc = fitz.open()
        for i in range(num_pages):
            page = doc.new_page()
            # Add some dummy text to make it look like a real document
            page.insert_text((50, 50), f"Emirates ID Side {i+1}")
            page.insert_text((50, 100), "Nationality: Indian")
            page.insert_text((50, 150), "784-1234-1234567-1")
        
        pdf_bytes = doc.tobytes()
        doc.close()
        return pdf_bytes

    def test_processing_speed(self):
        print("\n--- OCR PERFORMANCE BENCHMARK ---")
        
        # 1. Test 2-page PDF (Typical Emirates ID)
        pdf_content = self.create_dummy_pdf(num_pages=2)
        file_obj = SimpleUploadedFile("test_id.pdf", pdf_content, content_type="application/pdf")
        
        print("Starting 2-page PDF processing...")
        start_time = time.time()
        result = process_pdf_page_by_page(file_obj)
        elapsed = time.time() - start_time
        
        print(f"2-page processing took: {elapsed:.2f} seconds")
        print(f"Pages detected: {len(result)}")
        
        # 2. Test Image processing
        img = Image.new('RGB', (1000, 1000), color=(255, 255, 255))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_content = img_byte_arr.getvalue()
        
        img_file = SimpleUploadedFile("test_id.jpg", img_content, content_type="image/jpeg")
        
        print("\nStarting single image processing...")
        start_time = time.time()
        # Note: ocr_space_file_multi_lang is the entry point for images in views.py
        from api.ocr_space import ocr_space_file_multi_lang
        result_text, code, err = ocr_space_file_multi_lang(img_file, is_pdf=False)
        img_elapsed = time.time() - start_time
        
        print(f"Single image processing took: {img_elapsed:.2f} seconds")
        
        print("\n--- SUMMARY ---")
        print(f"Is time < 4s? {'✅ YES' if elapsed < 4.0 else '❌ NO (Needs GPU or faster CPU)'}")
        print("---------------------------------\n")

if __name__ == "__main__":
    # This can be run via: python manage.py test api.tests_ocr_performance
    pass
