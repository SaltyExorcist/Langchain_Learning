import os
from dotenv import load_dotenv

load_dotenv()

print("Token loaded:", os.getenv("HUGGINGFACEHUB_API_TOKEN") is not None)
