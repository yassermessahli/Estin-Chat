from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()
PORT = os.getenv("HOST_PORT", None)
API_KEY = os.getenv("API_KEY", None)


client = InferenceClient(
    base_url=f"http://172.17.0.1:{PORT}/embed",
    api_key=API_KEY,
)


if __name__ == "__main__":
    # Example usage
    embedding = client.feature_extraction("What is deep learning?")
    print(embedding[0])