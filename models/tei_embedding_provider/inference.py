from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import time

load_dotenv()
PORT = os.getenv("HOST_PORT", None)
API_KEY = os.getenv("API_KEY", None)


client = InferenceClient(
    base_url=f"http://localhost:{PORT}/embed",
    api_key=API_KEY,
)


if __name__ == "__main__":
    # Example usage
    st = time.time()
    embedding = client.feature_extraction("What is deep learning?")
    print(embedding[0][:10])
    et = time.time()
    print(f"generated in {et - st:.4f} seconds")