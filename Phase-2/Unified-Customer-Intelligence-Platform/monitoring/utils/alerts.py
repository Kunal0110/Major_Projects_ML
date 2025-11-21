import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

def send_email_alert(to_email: str, subject: str, message: str):
    """Send email alert using SMTP"""
    try:
        sender_email = os.getenv("ALERT_EMAIL", "alerts@example.com")
        sender_password = os.getenv("ALERT_EMAIL_PASSWORD", "")
        
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(message, 'plain'))
        
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        
        print(f"Email sent to {to_email}")
        return True
    except Exception as e:
        print(f"Email failed: {str(e)}")
        return False

def send_sms_alert(phone_number: str, message: str):
    """Send SMS alert using Twilio"""
    try:
        from twilio.rest import Client
        
        account_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN", "")
        twilio_phone = os.getenv("TWILIO_PHONE_NUMBER", "")
        
        if not all([account_sid, auth_token, twilio_phone]):
            print("Twilio credentials not configured")
            return False
        
        client = Client(account_sid, auth_token)
        message = client.messages.create(
            body=message,
            from_=twilio_phone,
            to=phone_number
        )
        
        print(f"SMS sent to {phone_number}")
        return True
    except Exception as e:
        print(f"SMS failed: {str(e)}")
        return False

def notify(message: str, email: str = None, phone: str = None):
    """Send notification via email and/or SMS"""
    results = []
    
    if email:
        results.append(send_email_alert(email, "Alert Notification", message))
    
    if phone:
        results.append(send_sms_alert(phone, message))
    
    return any(results) if results else False
