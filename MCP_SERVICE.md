# MCP StreamableHTTP Service (Step 4 Only)

This repo includes a small MCP service that exposes a single tool which performs **Step 4** from `main.py` (building the LangChain SQL agent) and **does not** run the demo questions from Step 5.

## Files

- `mcp_fastmcp_server.py` — MCP server (FastMCP + StreamableHTTP transport)
- `Procfile` — deployment entrypoint (`web: python mcp_fastmcp_server.py`)
- `main.py` — optional demo script (not required by the MCP server)

## MCP Tool

- Tool name: `ask_budget`
- Args:
  - `question` (required): natural-language question to analyze
  - `database_url` (optional): overrides `DATABASE_URL`
  - `model_name`, `temperature` (optional): LLM settings
  - `verbose` (optional): agent debug logging
  - `force_rebuild` (optional): rebuild cached agent

## Environment

- `OPENAI_API_KEY`: required by `langchain_openai.ChatOpenAI`
- `DATABASE_URL`: optional, defaults to `postgresql://postgres:postgres@localhost:5432/ocfl`
- `HOST`: bind host for the MCP service (default `0.0.0.0`)
- `PORT`: HTTP port for the MCP service (default `8000`)
- `AUTH_TOKEN`: optional shared secret; when set, client must send `Authorization: Bearer <AUTH_TOKEN>` (or `X-Auth-Token: <AUTH_TOKEN>`) for `/mcp` requests
- `MCP_ALLOWED_HOSTS`: optional, comma-separated allowlist for `Host` header (needed for ngrok/custom domains)
- `MCP_ALLOWED_ORIGINS`: optional, comma-separated allowlist for `Origin` header (needed for browser-based clients)
- `OPENAI_TIMEOUT`: optional request timeout seconds (default `60`)
- `OPENAI_MAX_RETRIES`: optional retry count (default `2`)
- `DEBUG_SQL_SCHEMA`: set to `1` to print DB schema/table samples

## Local Run

1. Install Python deps: `python -m pip install -r requirements.txt`
2. Start server: `HOST=127.0.0.1 PORT=8000 uvicorn mcp_fastmcp_server:app --host 127.0.0.1 --port 8000`

## URL

The StreamableHTTP endpoint is:

- `http://localhost:8000/mcp`
- Health check: `http://localhost:8000/healthz`

## Ngrok / Public URL

If you expose the server via ngrok and see `Invalid Host header` / `421 Misdirected Request`,
set `MCP_ALLOWED_HOSTS` (and `MCP_ALLOWED_ORIGINS` if your client sends an `Origin` header):

- `MCP_ALLOWED_HOSTS=YOUR_SUBDOMAIN.ngrok-free.app`
- `MCP_ALLOWED_ORIGINS=https://YOUR_SUBDOMAIN.ngrok-free.app`
