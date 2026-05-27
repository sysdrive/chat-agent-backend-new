
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from database import users_collection
from auth_utils import hash_password, verify_password, create_token

router = APIRouter()

class Register(BaseModel):
    name:str
    email:str
    password:str

class Login(BaseModel):
    email:str
    password:str

@router.post("/register")
def register(data:Register):
    existing = users_collection.find_one({"email":data.email})

    if existing:
        raise HTTPException(status_code=400, detail="Email exists")

    users_collection.insert_one({
        "name":data.name,
        "email":data.email,
        "password":hash_password(data.password)
    })

    return {"message":"Registered"}

@router.post("/login")
def login(data:Login):
    user = users_collection.find_one({"email":data.email})

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not verify_password(data.password, user["password"]):
        raise HTTPException(status_code=401, detail="Wrong password")

    token = create_token({"email":user["email"]})

    return {"token":token}
