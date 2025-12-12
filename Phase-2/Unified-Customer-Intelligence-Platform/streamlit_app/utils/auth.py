import streamlit as st
import sqlite3
from pathlib import Path
import bcrypt

USER_DB_PATH = Path("streamlit_app/users.db")

def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect(USER_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            phone TEXT,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password: str, hash: str) -> bool:
    """Verify password against hash"""
    return bcrypt.checkpw(password.encode(), hash.encode())

def signup(name: str, email: str, phone: str, password: str):
    """Register new user"""
    init_db()
    conn = sqlite3.connect(USER_DB_PATH)
    try:
        conn.execute(
            "INSERT INTO users (name, email, phone, password_hash) VALUES (?, ?, ?, ?)",
            (name, email, phone, hash_password(password))
        )
        conn.commit()
        return True, "Signup successful"
    except sqlite3.IntegrityError:
        return False, "Email already registered"
    finally:
        conn.close()

def login(email: str, password: str):
    """Authenticate user"""
    init_db()
    conn = sqlite3.connect(USER_DB_PATH)
    try:
        cursor = conn.execute(
            "SELECT name, email, phone, password_hash FROM users WHERE email = ?",
            (email,)
        )
        user = cursor.fetchone()
        
        if user and verify_password(password, user[3]):
            return True, {
                "name": user[0],
                "email": user[1],
                "phone": user[2]
            }
        return False, None
    finally:
        conn.close()

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
