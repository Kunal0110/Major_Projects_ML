import streamlit as st
import json
from pathlib import Path
import hashlib

USER_DB_PATH = Path("streamlit_app/users.json")

def hash_password(password: str) -> str:
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    """Load users from JSON file"""
    if USER_DB_PATH.exists():
        with open(USER_DB_PATH, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    """Save users to JSON file"""
    USER_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(USER_DB_PATH, 'w') as f:
        json.dump(users, f, indent=2)

def signup(name: str, email: str, phone: str, password: str):
    """Register new user"""
    users = load_users()
    
    if email in users:
        return False, "Email already registered"
    
    users[email] = {
        "name": name,
        "email": email,
        "phone": phone,
        "password": hash_password(password)
    }
    
    save_users(users)
    return True, "Signup successful"

def login(email: str, password: str):
    """Authenticate user"""
    users = load_users()
    
    if email not in users:
        return False, None
    
    if users[email]["password"] == hash_password(password):
        return True, users[email]
    
    return False, None

def is_authenticated():
    """Check if user is logged in"""
    return st.session_state.get("authenticated", False)

def get_current_user():
    """Get current logged in user"""
    return st.session_state.get("user", None)

def logout():
    """Logout current user"""
    st.session_state["authenticated"] = False
    st.session_state["user"] = None
