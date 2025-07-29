import random
from itsdangerous import URLSafeSerializer, BadSignature
from datetime import datetime, timedelta

SECRET_KEY = "super-secret-key"
SESSION_COOKIE = "session_token"
SESSION_DURATION_MINUTES = 30  # Session expiry duration


serializer = URLSafeSerializer(SECRET_KEY, salt="session")

def create_secure_cookie(data: dict) -> str:
    session = {
        **data,
        "exp": (datetime.utcnow() + timedelta(minutes=SESSION_DURATION_MINUTES)).timestamp()
    }
    return serializer.dumps(session)

def decode_secure_cookie(cookie: str) -> dict | None:
    try:
        session = serializer.loads(cookie)
        exp = datetime.fromtimestamp(session["exp"])
        if datetime.utcnow() > exp:
            return None  # Session expired
        return session
    except (BadSignature, KeyError):
        return None

def parse_cookie_header(header: str) -> dict:
    return {
        kv.split("=", 1)[0].strip(): kv.split("=", 1)[1].strip()
        for kv in header.split(";")
        if "=" in kv
    }

def generate_otp() -> str:
    return str(random.randint(100000, 999999))
