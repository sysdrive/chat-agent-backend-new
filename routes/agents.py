from fastapi import APIRouter
from pydantic import BaseModel
from database import agents_collection
import uuid

router = APIRouter()

class Agent(BaseModel):
    name: str
    description: str
    category: str

@router.post("/agents/create")
def create_agent(data: Agent):

    agent_id = str(uuid.uuid4())

    agent = {
        "agent_id": agent_id,
        "name": data.name,
        "description": data.description,
        "category": data.category
    }

    agents_collection.insert_one(agent)

    return {
        "message": "Agent created",
        "agent": agent
    }

@router.get("/agents")
def get_agents():

    agents = list(
        agents_collection.find({}, {"_id": 0})
    )

    return {
        "agents": agents
    }

@router.delete("/agents/{agent_id}")
def delete_agent(agent_id: str):

    agents_collection.delete_one({
        "agent_id": agent_id
    })

    return {
        "message": "Agent deleted"
    }