# vLLM Inference Server

This directory contains the configuration for running a vLLM-based OpenAI-compatible API server using Docker Compose.

## Prerequisites

- Docker
- Docker Compose
- NVIDIA Container Toolkit
- A Hugging Face Hub token

## Setup

1.  **Configure Environment Variables:**
    Create a `.env` file in this directory by copying the provided `.env.example` or by creating a new one. Add your Hugging Face Hub token to it:

    ```env
    HUGGING_FACE_HUB_TOKEN=<your-hugging-face-token>
    ```

2.  **Start the Server:**
    Open a terminal in this directory and run the following command:
    ```bash
    docker-compose up -d
    ```
    This will pull the vLLM Docker image and start the server in detached mode. The first time you run this, it will download the model, which may take some time depending on your internet connection.

## Usage

### Check Logs

To monitor the server and see the logs, run:

```bash
docker-compose logs -f
```

### Test the API

Once the server is running, you can interact with it using the OpenAI API format. Here is an example using `curl`:

```bash
curl http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
    "model": "TheBloke/Mistral-7B-v0.1-AWQ",
    "prompt": "San Francisco is a",
    "max_tokens": 7,
    "temperature": 0
}'
```

### Stop the Server

To stop the vLLM server, run:

```bash
docker-compose down
```
