from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('transcribe/', views.transcribe_audio, name='transcribe'),
    path('chat/', views.chat_reply, name='chat_reply'),
    path("emirates-id/", views.emirates_id_upload, name="emirates_id_upload"), 
    path('insurance-chat/', views.insurance_chat, name='insurance_chat'),
    path("emirates-id/update/", views.update_emirates_id_record, name="update_emirates_id_record"),
    path('get-session/', views.get_user_session_data, name='get_session_data'),
    path('check-session/', views.check_session_status, name='check_session_status'),
    # Add authentication endpoints
    path('auth/send-otp/', views.send_otp, name='send_otp'),
    path('auth/verify-otp/', views.verify_otp, name='verify_otp'),
    path('auth/chat-history/', views.get_user_chat_history, name='get_chat_history'),
    path("api/save-missing-field/", views.save_missing_field, name="save_missing_field"),
    path("save-mobile/", views.save_mobile, name="save_mobile"),
    # path("passport/", views.passport_upload, name="passport_upload"),
    path("passport/", views.passport_upload, name="passport_upload"),
    path("uae-visa/", views.uae_visa_upload, name="uae_visa_upload"),
    path("emirates-id-test/", views.emirates_id_upload_test, name="emirates_id_upload_test"),
]
