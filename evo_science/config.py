from dotenv import load_dotenv
import os

load_dotenv()


DATA_ROOT = os.getenv("DATA_ROOT") or "data"
