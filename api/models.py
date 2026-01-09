from django.db import models
import uuid
import random
import datetime
from django.utils import timezone
from django.contrib.auth.models import AbstractUser


class User(AbstractUser):
    email = models.EmailField(unique=True)
    phone = models.CharField(max_length=15, blank=True, null=True)
    email_verified = models.BooleanField(default=False)
    otp_secret = models.CharField(max_length=16, blank=True, null=True)
    
    def __str__(self):
        return self.email

class ChatSession(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    session_id = models.CharField(max_length=36, unique=True, db_index=True)
    uuid = models.UUIDField(default=uuid.uuid4, editable=False, unique=True, db_index=True)
    step = models.CharField(max_length=64, default="start")   
    is_completed = models.BooleanField(default=False)
    looking_for_insurance = models.CharField(max_length=5, blank=True, null=True) 
    role = models.CharField(max_length=20, blank=True, null=True)            
    salary = models.CharField(max_length=64, blank=True, null=True)
    depender_type = models.CharField(max_length=20, blank=True, null=True)
    depender_name = models.CharField(max_length=255, blank=True, null=True)  # NEW FIELD
    
    # NEW FIELDS for OpenAI dynamic flow
    full_name = models.CharField(max_length=255, blank=True, null=True)
    emirates_id_number = models.CharField(max_length=20, blank=True, null=True)
    dob = models.CharField(max_length=32, blank=True, null=True)
    expiry = models.CharField(max_length=32, blank=True, null=True)
    nationality = models.CharField(max_length=128, blank=True, null=True)
    occupation = models.CharField(max_length=128, blank=True, null=True)
    mobile = models.CharField(max_length=20, blank=True, null=True)
    emirates_id_uploaded = models.BooleanField(default=False)
    
    updated_at = models.DateTimeField(auto_now=True) 
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Session {self.user_id} - Step: {self.step}"
    
class ChatMessage(models.Model):
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name="messages")
    role = models.CharField(max_length=20)   # "user" or "bot"
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    
class OTP(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    otp_code = models.CharField(max_length=6)
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()
    is_used = models.BooleanField(default=False)
    
    def is_valid(self):
        return not self.is_used and timezone.now() < self.expires_at
    
    @classmethod
    def generate_otp(cls, user):
        # Delete any existing OTPs for this user
        cls.objects.filter(user=user).delete()
        
        # Generate 6-digit OTP
        otp_code = str(random.randint(100000, 999999))
        expires_at = timezone.now() + datetime.timedelta(minutes=10)
        
        return cls.objects.create(
            user=user,
            otp_code=otp_code,
            expires_at=expires_at
        )
    
    def __str__(self):
        return f"{self.user.email} - {self.otp_code}"



# models.py - Update EmiratesIDRecord model
class EmiratesIDRecord(models.Model):
    uuid = models.UUIDField(default=uuid.uuid4, editable=False, unique=True, db_index=True) 
    chat_session = models.OneToOneField(
        ChatSession, on_delete=models.CASCADE, related_name='emirates_id_record',
        blank=True, null=True
    )
    emirates_id = models.CharField(max_length=32, blank=True, null=True, db_index=True)
    name = models.CharField(max_length=256, blank=True, null=True)
    dob = models.CharField(max_length=32, blank=True, null=True)  
    issuing_date = models.CharField(max_length=32, blank=True, null=True)  
    nationality = models.CharField(max_length=128, blank=True, null=True)
    gender = models.CharField(max_length=16, blank=True, null=True)
    address = models.TextField(blank=True, null=True)
    occupation = models.CharField(max_length=128, blank=True, null=True)
    employer = models.CharField(max_length=256, blank=True, null=True)
    issuing_place = models.CharField(max_length=128, blank=True, null=True)
    family_sponsor = models.CharField(max_length=5, blank=True, null=True)  # Yes/No
    family_sponsor_name = models.CharField(max_length=128, blank=True, null=True)  # NEW FIELD
    expiry_date = models.CharField(max_length=32, blank=True, null=True)  
    raw_response = models.JSONField(blank=True, null=True)
    status = models.CharField(max_length=32, default="pending")  
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)             

    def __str__(self):
        return f"{self.emirates_id or 'No ID'} - {self.name or 'Unknown'}"


# Passport Record Model
class PassportRecord(models.Model):
    uuid = models.UUIDField(default=uuid.uuid4, editable=False, unique=True, db_index=True)
    chat_session = models.OneToOneField(
        ChatSession, on_delete=models.CASCADE, related_name='passport_record',
        blank=True, null=True
    )
    passport_number = models.CharField(max_length=32, blank=True, null=True, db_index=True)
    name = models.CharField(max_length=256, blank=True, null=True)
    given_name = models.CharField(max_length=256, blank=True, null=True)  # First name
    surname = models.CharField(max_length=256, blank=True, null=True)  # Last name
    dob = models.CharField(max_length=32, blank=True, null=True)
    nationality = models.CharField(max_length=128, blank=True, null=True)
    gender = models.CharField(max_length=16, blank=True, null=True)
    place_of_birth = models.CharField(max_length=256, blank=True, null=True)
    date_of_issue = models.CharField(max_length=32, blank=True, null=True)
    date_of_expiry = models.CharField(max_length=32, blank=True, null=True)
    issuing_authority = models.CharField(max_length=128, blank=True, null=True)
    passport_type = models.CharField(max_length=16, blank=True, null=True)  # P, D, etc.
    is_valid = models.BooleanField(default=False)  # Validation result
    validation_message = models.TextField(blank=True, null=True)  # Validation details
    name_match = models.BooleanField(default=False)  # Does name match Emirates ID
    expiry_valid = models.BooleanField(default=False)  # Is passport not expired
    raw_response = models.JSONField(blank=True, null=True)
    status = models.CharField(max_length=32, default="pending")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.passport_number or 'No Passport'} - {self.name or 'Unknown'}"


class UAEDocumentVisa(models.Model):
    id_number = models.CharField(max_length=50, null=True, blank=True)
    file_number = models.CharField(max_length=50, null=True, blank=True)
    passport_no = models.CharField(max_length=50, null=True, blank=True)
    employer_name = models.CharField(max_length=200, null=True, blank=True)
    uid_no = models.CharField(max_length=50, null=True, blank=True)
    issuing_date = models.CharField(max_length=50, null=True, blank=True)
    expiry_date = models.CharField(max_length=50, null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    raw_text = models.TextField(blank=True, null=True)

    def __str__(self):
        return f"UAE Visa ({self.file_number or 'N/A'})"
    

import secrets

class APIToken(models.Model):
    name = models.CharField(max_length=100)
    token = models.CharField(max_length=64, unique=True, db_index=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    @staticmethod
    def generate():
        return secrets.token_hex(32)

    def __str__(self):
        return self.name
