
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

client = MongoClient(os.getenv("MONGO_URI"))
db = client["medical_agent"]

users_collection = db["users"]
chat_collection = db["chats"]
