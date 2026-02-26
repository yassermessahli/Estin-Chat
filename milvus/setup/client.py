from pymilvus import MilvusClient
from dotenv import load_dotenv
import os


# Load environment variables
load_dotenv()

# Setup Connection
# Connect to Milvus server as root
client = MilvusClient(uri=os.getenv("MILVUS_HOST"), token="root:Milvus")


# Create and use the 'core_db' database
db_name = "core_db"
if db_name not in client.list_databases():
    print(f"Database '{db_name}' not found, creating new one...")
    client.create_database(db_name)

client.use_database(db_name)