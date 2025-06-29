from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility,
)
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class MilvusClient:
    def __init__(
        self,
        host: str = "localhost",
        port: str = "19530",
        collection_names: List[str] = [
            "texts_collection",
            "tables_collection",
            "images_collection",
        ],
    ):
        self.host = host
        self.port = port
        self.collections = {}
        self.connect()
        for name in collection_names:
            if utility.has_collection(name):
                self.collections[name] = Collection(name)

    def connect(self):
        """Connect to Milvus instance"""
        try:
            connections.connect(alias="default", host=self.host, port=self.port)
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise

    def insert_chunks(self, collection_name: str, chunks_data: List[Dict[str, Any]]):
        """Insert document chunks into a specified Milvus collection"""
        if not chunks_data:
            return

        if collection_name not in self.collections:
            logger.error(f"Collection {collection_name} does not exist.")
            raise ValueError(f"Collection {collection_name} not found.")

        collection = self.collections[collection_name]

        # Dynamically prepare data for insertion based on collection schema
        field_names = [field.name for field in collection.schema.fields]
        data = []
        for field_name in field_names:
            if field_name == "id":
                continue  # Skip auto-id field
            # This part assumes chunk_data keys match field names.
            # For text/table/image description, the key in chunk_data should be correct.
            data.append([chunk[field_name] for chunk in chunks_data])

        collection.insert(data)
        collection.flush()
        logger.info(f"Inserted {len(chunks_data)} chunks into {collection_name}")
            FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="year", dtype=DataType.INT64),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=65535),
        ]

        for name, specific_fields in collection_schemas.items():
            if utility.has_collection(name):
                logger.info(f"Collection {name} already exists")
                self.collections[name] = Collection(name)
                continue

            schema = CollectionSchema(
                common_fields + specific_fields, f"Collection for {name}"
            )
            collection = Collection(name, schema)
            self.collections[name] = collection

            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128},
            }
            collection.create_index("vector", index_params)
            logger.info(f"Created collection {name}")

    def insert_chunks(self, collection_name: str, chunks_data: List[Dict[str, Any]]):
        """Insert document chunks into a specified Milvus collection"""
        if not chunks_data:
            return

        if collection_name not in self.collections:
            logger.error(f"Collection {collection_name} does not exist.")
            raise ValueError(f"Collection {collection_name} not found.")

        collection = self.collections[collection_name]

        # Dynamically prepare data for insertion based on collection schema
        field_names = [field.name for field in collection.schema.fields]
        data = []
        for field_name in field_names:
            if field_name == "id":
                continue  # Skip auto-id field
            # This part assumes chunk_data keys match field names.
            # For text/table/image description, the key in chunk_data should be correct.
            data.append([chunk[field_name] for chunk in chunks_data])

        collection.insert(data)
        collection.flush()
        logger.info(f"Inserted {len(chunks_data)} chunks into {collection_name}")

    def load_collections(self):
        """Load all collections into memory for search"""
        for name, collection in self.collections.items():
            collection.load()
            logger.info(f"Loaded collection {name}")
