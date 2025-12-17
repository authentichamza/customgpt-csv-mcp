"""
MCP StreamableHTTP service (Python) using FastMCP.

Exposes a single tool that runs a LangChain SQL agent against the configured DB.
"""

from __future__ import annotations

import os
import logging
import threading
from typing import Any

import anyio
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route

from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AuthTokenMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        *,
        token: str,
        protected_prefix: str = "/mcp",
        exempt_paths: set[str] | None = None,
    ):
        super().__init__(app)
        self.token = token
        self.protected_prefix = protected_prefix
        self.exempt_paths = exempt_paths or set()

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path in self.exempt_paths or not path.startswith(self.protected_prefix):
            return await call_next(request)

        if request.method == "OPTIONS":
            return await call_next(request)

        auth_header = request.headers.get("authorization", "")
        token_header = request.headers.get("x-auth-token", "")

        expected_bearer = f"Bearer {self.token}"
        if auth_header == expected_bearer or auth_header == self.token or token_header == self.token:
            return await call_next(request)

        return JSONResponse(
            {"error": "Unauthorized"},
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )

STEP4_PREFIX = """You are analyzing county budget data across three tables:

1. actual_expenses: Contains actual spending by fiscal year
2. adopted_budget: Contains adopted/planned budgets by fiscal year
3. budget_roll: Contains budget rollover amounts by fiscal year

CRITICAL SQL RULES - READ CAREFULLY:
==================================
1. PostgreSQL is CASE-SENSITIVE for column names
2. ALL column names with uppercase letters or spaces MUST be double-quoted
3. NEVER use unquoted uppercase column names - they will fail
4. ALWAYS use double quotes (") for column names, not single quotes (')

CORRECT EXAMPLES:
  ✓ SELECT "DEPARTMENT", "UNIT NAME" FROM actual_expenses
  ✓ WHERE "DEPARTMENT" = 'HR' AND "fiscal_year" = '2024'
  ✓ SELECT SUM("amount") FROM actual_expenses WHERE "OBJECT CODE" = 1120

INCORRECT EXAMPLES (WILL FAIL):
  ✗ SELECT DEPARTMENT FROM actual_expenses (missing quotes)
  ✗ WHERE 'DEPARTMENT' = 'HR' (wrong quote type)
  ✗ WHERE DEPARTMENT = 'HR' (no quotes on column name)

KEY COLUMNS (always use double quotes):
- "DEPARTMENT" - organizational department
- "DIVISION" - sub-department division
- "UNIT NAME" - specific unit name
- "OBJECT CODE" - expense type code
- "OBJECT NAME" - expense type description
- "OBJECT GROUP" - expense category (Salaries, Operating, Capital)
- "APPROPRIATION" - budget category
- "DEPT CODE", "UNIT", "FUND" - numeric identifiers
- "amount" - dollar amount
- "fiscal_year" - year as text (2019-2025)

QUERY STRATEGY:
1. Start by examining the schema carefully
2. Before writing SQL, identify which columns you need
3. ALWAYS double-quote column names with uppercase or spaces
4. Use single quotes for string values (e.g., '2024', 'HR')
5. When checking for departments, search BOTH "DEPARTMENT" and "UNIT NAME"

Common department abbreviations:
- ISS or Information Systems Services
- HR or Human Resources

BEFORE executing any query, mentally verify:
□ Are ALL column names double-quoted?
□ Are string values single-quoted?
□ Did I check both "DEPARTMENT" and "UNIT NAME" for department filters?

Always format currency amounts clearly with dollar signs and commas in your final answer.
"""


def build_agent_executor(
    *,
    database_url: str | None = None,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0,
    sample_rows_in_table_info: int = 2,
    verbose: bool = False,
    timeout: float | None = None,
    max_retries: int | None = 2,
):
    """Build the LangChain SQL agent executor."""
    logger.info(f"Building agent executor with model={model_name}")
    
    db_url = database_url or os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/ocfl",
    )
    # Heroku commonly provides DATABASE_URL as `postgres://...` but SQLAlchemy expects `postgresql://...`
    if db_url.startswith("postgres://"):
        db_url = "postgresql://" + db_url[len("postgres://") :]
    
    logger.info(f"Connecting to database: {db_url.split('@')[-1]}")  # Log without credentials
    
    db = SQLDatabase.from_uri(
        db_url,
        sample_rows_in_table_info=sample_rows_in_table_info,
    )

    if os.getenv("DEBUG_SQL_SCHEMA") == "1":
        logger.info("SQL Schema Info:")
        logger.info(db.get_table_info())

    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
    )
    logger.info(f"Created LLM: {llm.model_name}")

    agent = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="openai-tools",
        verbose=verbose,
        prefix=STEP4_PREFIX,
    )
    
    logger.info("Agent executor created successfully")
    return agent


def _csv_env(name: str) -> list[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


allowed_hosts = _csv_env("MCP_ALLOWED_HOSTS")
allowed_origins = _csv_env("MCP_ALLOWED_ORIGINS")
transport_security = (
    TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=allowed_hosts,
        allowed_origins=allowed_origins,
    )
    if (allowed_hosts or allowed_origins)
    else None
)

mcp = FastMCP(
    "csv-llm-step4",
    json_response=True,
    host=os.getenv("HOST", "0.0.0.0"),
    port=int(os.getenv("PORT", "8000")),
    transport_security=transport_security,
)

# Per-process cache (note: with multiple Heroku workers, each worker has its own cache)
_agent_executor: Any | None = None
_agent_config: tuple[Any, ...] | None = None
_agent_lock = threading.Lock()


def _get_agent_executor(
    *,
    database_url: str | None,
    model_name: str,
    temperature: float,
    sample_rows_in_table_info: int,
    verbose: bool,
    force_rebuild: bool = False,
):
    """Get or create the agent executor (with simple caching)."""
    global _agent_executor, _agent_config

    config = (
        database_url or os.getenv("DATABASE_URL"),
        model_name,
        float(temperature),
        int(sample_rows_in_table_info),
        bool(verbose),
    )

    with _agent_lock:
        if force_rebuild or _agent_executor is None or _agent_config != config:
            logger.info(f"Building new agent (force_rebuild={force_rebuild})")
            _agent_executor = build_agent_executor(
                database_url=config[0],
                model_name=model_name,
                temperature=temperature,
                sample_rows_in_table_info=sample_rows_in_table_info,
                verbose=verbose,
                timeout=float(os.getenv("OPENAI_TIMEOUT", "60")),
                max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "2")),
            )
            _agent_config = config
        else:
            logger.info("Using cached agent executor")

    return _agent_executor


@mcp.tool()
async def ask_budget(
    question: str,
    database_url: str | None = None,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0,
    sample_rows_in_table_info: int = 2,
    verbose: bool = True,  # Default to True for debugging
    force_rebuild: bool = False,
) -> dict:
    """
    Analyze a natural-language question using the SQL agent.

    This uses the Step 4 agent (LLM + SQLDatabase tools) to generate SQL,
    run it against the database, and return the final answer text.
    
    Args:
        question: Natural language question about the budget data
        database_url: PostgreSQL connection string (defaults to DATABASE_URL env var)
        model_name: OpenAI model to use (default: gpt-4o-mini)
        temperature: LLM temperature (default: 0 for deterministic output)
        sample_rows_in_table_info: Number of sample rows to include in schema (default: 2)
        verbose: Whether to log agent's thinking process (default: True)
        force_rebuild: Force rebuilding the agent (default: False)
    
    Returns:
        dict with 'ok' status and either 'output' (success) or 'error' (failure)
    """
    logger.info("="*80)
    logger.info(f"ask_budget called")
    logger.info(f"Question: {question}")
    logger.info(f"Model: {model_name}, Temp: {temperature}, Verbose: {verbose}")
    logger.info("="*80)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY is not set")
        return {"ok": False, "error": "OPENAI_API_KEY is not set"}

    try:
        def _run_in_thread():
            agent_executor = _get_agent_executor(
                database_url=database_url,
                model_name=model_name,
                temperature=temperature,
                sample_rows_in_table_info=sample_rows_in_table_info,
                verbose=verbose,
                force_rebuild=force_rebuild,
            )
            logger.info(f"Invoking agent with question: {question}")
            return agent_executor.invoke({"input": question})

        result = await anyio.to_thread.run_sync(_run_in_thread)
        
        logger.info(f"Agent completed. Result type: {type(result)}")
        
        # Extract output
        output = result.get("output") if isinstance(result, dict) else str(result)
        
        logger.info(f"Output length: {len(output)} characters")
        logger.info(f"Output preview: {output[:200]}...")
        
        return {"ok": True, "output": output}
        
    except Exception as e:
        logger.error(f"Error in ask_budget: {str(e)}", exc_info=True)
        return {"ok": False, "error": str(e)}


def create_app():
    """
    Create the Starlette ASGI app for StreamableHTTP.
    Exporting a module-level `app` allows Uvicorn/Gunicorn workers on Heroku.
    """
    app = mcp.streamable_http_app()

    async def healthz(_request: Request):
        return PlainTextResponse("ok")

    app.router.routes.append(Route("/healthz", endpoint=healthz, methods=["GET"]))

    auth_token = os.getenv("AUTH_TOKEN", "").strip()
    if auth_token:
        app.add_middleware(
            AuthTokenMiddleware,
            token=auth_token,
            protected_prefix=mcp.settings.streamable_http_path,
            exempt_paths={"/healthz"},
        )

    return app


# ASGI app entrypoint for Procfile: `uvicorn mcp_fastmcp_server:app --port $PORT`
app = create_app()


if __name__ == "__main__":
    logger.info("Starting MCP server...")
    import uvicorn

    uvicorn.run(
        "mcp_fastmcp_server:app",
        host=mcp.settings.host,
        port=mcp.settings.port,
        log_level=mcp.settings.log_level.lower(),
        workers=1,
    )