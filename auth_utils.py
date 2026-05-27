from passlib.context import CryptContext
from jose import jwt
from datetime import datetime, timedelta

SECRET_KEY = "mysecretkey"

# NO BCRYPT
pwd_context = CryptContext(
    schemes=["pbkdf2_sha256"],
    deprecated="auto"
)

# =========================
# HASH PASSWORD
# =========================

def hash_password(password: str):

    return pwd_context.hash(password)

# =========================
# VERIFY PASSWORD
# =========================

def verify_password(
    plain_password: str,
    hashed_password: str
):

    return pwd_context.verify(
        plain_password,
        hashed_password
    )

# =========================
# CREATE TOKEN
# =========================

def create_token(data: dict):

    payload = data.copy()

    payload["exp"] = (
        datetime.utcnow() + timedelta(days=7)
    )

    token = jwt.encode(
        payload,
        SECRET_KEY,
        algorithm="HS256"
    )

    return token