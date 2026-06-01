# src/agent/agent_config.py

import os
from dotenv import load_dotenv
from langsmith import Client

#Activate langSmith
client = Client()
#Load env variables
load_dotenv()

# --- API KEYS ---
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- LLM MODEL ---
LLM_MODEL = "llama-3.1-8b-instant"

# --- AGENT PARAMETERS ---
THRESHOLD_APPROVE = 0.30
THRESHOLD_REJECT = 0.70

# --- VALIDATION ---
if not LANGSMITH_API_KEY:
    raise ValueError("Missing LANGSMITH_API_KEY in .env")

if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY in .env")
