# Milvus Setup & Data Management

Quick guide to set up Milvus vector database, create collections, and efficiently populate with bulk data.

## Quick Start

### 1. Run Milvus Instance

```bash
# Start Milvus with docker compose if it is not running
cd milvus
docker compose up -d
```

### 2. Create Collection & Print Stats

```bash
# Execute from repo root - create collection
# NOTE: do not execute this command if the collection already exists with inserted data.
python -m milvus.setup.collection

# View statistics
python -m milvus.setup.stats
```

### 3. Bulk Data Population

```bash
# Execute from repo root - use CLI to populate Milvus with bulk data

# 1. Extract records from JSONL files (Output of OpenAI Batch API) to JSON records
python -m milvus.data_upload.cli --get-records-from-jsonl <openai-batch-output-folder> <jsons-records-folder>

# 2. Push JSON records to MinIO storage
# (This will return the created parquet file name in the MinIO bucket)
python -m milvus.data_upload.cli --push-bulk-to-minio <jsons-records-folder>

# 3. Import bulk data from MinIO to Milvus
# Use the returned parquet file name from the previous step
# (You can also specify multiple files if needed)
python -m milvus.data_upload.cli --populate-milvus <created-parquet-files-from-minio>
```

## Folder Structure

### `setup/` - Collection Management

**Purpose**: Create and configure Milvus collections with proper schema and indexing.

**Files**:

- `client.py` - Milvus client connection and database setup
- `collection.py` - Create collection with schema and index parameters
- `schema.py` - Define collection schema and field configurations
- `stats.py` - Collection statistics and record count utilities

**Usage**:

```bash
# Create new collection (drops existing if found)
python -m milvus.setup.collection

# View collection statistics
python -m milvus.setup.stats
```

### `data_upload/` - Bulk Data Operations

**Purpose**: Efficiently handle large-scale data imports using Milvus BulkWriter API.

**Files**:

- `cli.py` - Command-line interface for data operations
- `utils.py` - BulkWriter utilities and import functions

**Usage**:

```bash
# Extract records from JSONL files to JSON format
python -m milvus.data_upload.cli --get-records-from-jsonl INPUT_FOLDER OUTPUT_FOLDER

# Push processed data to MinIO storage
python -m milvus.data_upload.cli --push-bulk-to-minio INPUT_FOLDER

# Import bulk data from MinIO to Milvus
python -m milvus.data_upload.cli --populate-milvus [BATCH_FILES...]

# Check import job status
python -m milvus.data_upload.cli --get-import-status JOB_ID
```

### `utils/` - Database Utilities

**Purpose**: Collection verification, data loading, and validation tools.

**Files**:

- `verify_and_load.py` - Load collection and verify data accessibility
- `load_optimized.py` - Optimized data loading procedures
- `print_record.py` - Record inspection and debugging
- `verify_vectors.py` - Vector data validation

**Usage**:

```bash
# Verify and load collection for searching
python -m milvus.utils.verify_and_load

# Validate vector data integrity
python -m milvus.utils.verify_vectors
```

## Configuration

Ask the administrator to provide the `.env` file with necessary environment variables.
Or:

1. Copy `.env.milvus.example` to `.env`
2. Configure environment variables yourself

## Core Files

- `compose.yml` - Docker Compose configuration for Milvus stack (etcd, MinIO, Milvus)
- `milvus.yaml` - Milvus server configuration
- `.env` - Environment variables (create from `.env.milvus.example`)

## Workflow Summary

1. **Setup**: `docker compose up -d` → `python -m milvus.setup.collection`
2. **Data Prep**: `python -m milvus.data_upload.cli --get-records-from-jsonl`
3. **Bulk Import**: `python -m milvus.data_upload.cli --push-bulk-to-minio` → `--populate-milvus`
4. **Verify**: `python -m milvus.setup.stats` → `python -m milvus.utils.verify_and_load`
