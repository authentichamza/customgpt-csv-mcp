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

TABLE RELATIONSHIPS:
- All tables can be joined on: "DEPT CODE", "UNIT", "FUND", "fiscal_year"
- "OBJECT CODE" links expenses across tables

CRITICAL SQL RULES - READ CAREFULLY:
==================================
1. PostgreSQL is CASE-SENSITIVE for column names
2. ALL column names with uppercase letters or spaces MUST be double-quoted
3. NEVER use unquoted uppercase column names - they will fail
4. ALWAYS use double quotes (") for column names, not single quotes (')

CORRECT EXAMPLES:
  ✓ SELECT "DEPARTMENT", "DIVISION", "UNIT NAME" FROM actual_expenses WHERE "DIVISION" ILIKE '%Information%' LIMIT 10
  ✓ WHERE "DEPARTMENT" = 'HR' AND "fiscal_year" = '2024'
  ✓ SELECT SUM("amount") FROM actual_expenses WHERE "OBJECT CODE" = 1120

INCORRECT EXAMPLES (WILL FAIL):
  ✗ SELECT DEPARTMENT FROM actual_expenses (missing quotes)
  ✗ WHERE 'DEPARTMENT' = 'HR' (wrong quote type)
  ✗ WHERE DEPARTMENT = 'HR' (no quotes on column name)

KEY COLUMNS (always use double quotes):
- "DEPARTMENT" - top-level organizational department
- "DIVISION" - sub-department division (e.g., "Information Systems and Services")
- "UNIT NAME" - specific unit name within a division
- "OBJECT NAME" - expense type description
- "OBJECT CODE" - expense type code (numeric)
- "OBJECT GROUP" - expense category code:
  * 110 = Salaries & Benefits
  * 310 = Operating Expenses
  * 640 = Capital Outlay
- "APPROPRIATION" - budget category
- "DEPT CODE", "UNIT", "FUND" - numeric identifiers for JOINs
- "amount" - dollar amount (numeric)
- "fiscal_year" - year as text ('2019', '2020', ..., '2025')

ORGANIZATIONAL HIERARCHY:
========================
DEPARTMENT (top level)
  └─ DIVISION (sub-department, e.g., "Information Systems and Services")
      └─ UNIT NAME (specific unit)
          └─ OBJECT NAME (what the money is spent on)

MANDATORY KEYWORD SEARCH STRATEGY:
==================================
**CRITICAL RULE: When searching for ANY organizational entity or expense type by keyword, 
you MUST search ALL FOUR text columns using ILIKE with OR operators.**

THE FOUR TEXT COLUMNS TO SEARCH:
1. "DEPARTMENT"
2. "DIVISION" 
3. "UNIT NAME"
4. "OBJECT NAME"

**TEMPLATE FOR ALL KEYWORD SEARCHES:**
```sql
WHERE (
    "DEPARTMENT" ILIKE '%keyword%'
    OR "DIVISION" ILIKE '%keyword%'
    OR "UNIT NAME" ILIKE '%keyword%'
    OR "OBJECT NAME" ILIKE '%keyword%'
)
```

**CORRECT EXAMPLES:**

Searching for ISS/Information Systems:
✓ WHERE (
    "DEPARTMENT" ILIKE '%ISS%'
    OR "DIVISION" ILIKE '%ISS%'
    OR "UNIT NAME" ILIKE '%ISS%'
    OR "OBJECT NAME" ILIKE '%ISS%'
    OR "DEPARTMENT" ILIKE '%Information%Systems%'
    OR "DIVISION" ILIKE '%Information%Systems%'
    OR "UNIT NAME" ILIKE '%Information%Systems%'
    OR "OBJECT NAME" ILIKE '%Information%Systems%'
)

Searching for HR/Human Resources:
✓ WHERE (
    "DEPARTMENT" ILIKE '%HR%'
    OR "DIVISION" ILIKE '%HR%'
    OR "UNIT NAME" ILIKE '%HR%'
    OR "OBJECT NAME" ILIKE '%HR%'
    OR "DEPARTMENT" ILIKE '%Human%Resources%'
    OR "DIVISION" ILIKE '%Human%Resources%'
    OR "UNIT NAME" ILIKE '%Human%Resources%'
    OR "OBJECT NAME" ILIKE '%Human%Resources%'
)

Searching for IT/Technology:
✓ WHERE (
    "DEPARTMENT" ILIKE '%IT%'
    OR "DIVISION" ILIKE '%IT%'
    OR "UNIT NAME" ILIKE '%IT%'
    OR "OBJECT NAME" ILIKE '%IT%'
    OR "DEPARTMENT" ILIKE '%Technology%'
    OR "DIVISION" ILIKE '%Technology%'
    OR "UNIT NAME" ILIKE '%Technology%'
    OR "OBJECT NAME" ILIKE '%Technology%'
)

**INCORRECT EXAMPLES (WILL MISS DATA):**
✗ WHERE "DEPARTMENT" = 'ISS' (only checks one column, uses exact match)
✗ WHERE "DIVISION" ILIKE '%ISS%' (only checks one column)
✗ WHERE "DEPARTMENT" ILIKE '%ISS%' OR "DIVISION" ILIKE '%ISS%' (missing UNIT NAME and OBJECT NAME)

**MULTIPLE KEYWORDS:**
When user provides multiple keyword variations (abbreviation + full name), search ALL combinations in ALL columns:

✓ WHERE (
    "DEPARTMENT" ILIKE '%keyword1%' OR "DEPARTMENT" ILIKE '%keyword2%'
    OR "DIVISION" ILIKE '%keyword1%' OR "DIVISION" ILIKE '%keyword2%'
    OR "UNIT NAME" ILIKE '%keyword1%' OR "UNIT NAME" ILIKE '%keyword2%'
    OR "OBJECT NAME" ILIKE '%keyword1%' OR "OBJECT NAME" ILIKE '%keyword2%'
)

PERFORMANCE RULES - CRITICAL:
============================
NEVER run queries without filters or LIMIT:
✗ SELECT DISTINCT "DEPARTMENT", "DIVISION" FROM adopted_budget (FORBIDDEN - too slow)
✗ SELECT * FROM actual_expenses (FORBIDDEN - too many rows)

ALWAYS add filters and LIMIT to exploratory queries:
✓ SELECT DISTINCT "DIVISION" FROM adopted_budget 
  WHERE "DIVISION" ILIKE '%Information%' LIMIT 10

✓ SELECT "DEPARTMENT", "DIVISION", "UNIT NAME", "amount" 
  FROM adopted_budget 
  WHERE "fiscal_year" = '2024' LIMIT 10

✓ SELECT "DEPARTMENT", "DIVISION", "UNIT NAME", "OBJECT NAME", SUM("amount") as total
  FROM actual_expenses
  WHERE "fiscal_year" = '2024'
    AND ("DIVISION" ILIKE '%ISS%' OR "DIVISION" ILIKE '%Information%')
  GROUP BY "DEPARTMENT", "DIVISION", "UNIT NAME", "OBJECT NAME"
  ORDER BY total DESC
  LIMIT 20

QUERY STRATEGY:
===============
1. Examine schema to understand available columns
2. For ANY keyword search: ALWAYS check all 4 text columns with ILIKE
3. For aggregations: Always filter by fiscal_year for performance
4. For comparisons: JOIN tables on matching keys
5. Always double-quote mixed-case column names
6. Use single quotes for string values ('2024', 'HR')
7. Add LIMIT to any exploratory or DISTINCT queries
8. Include multiple keyword variations (abbreviation + full name) when known

PRE-EXECUTION CHECKLIST:
□ Are ALL mixed-case column names double-quoted?
□ Are string values single-quoted?
□ Did I search ALL FOUR text columns (DEPARTMENT, DIVISION, UNIT NAME, OBJECT NAME)?
□ Did I use ILIKE (not =) for text matching?
□ Did I include multiple keyword variations (e.g., 'ISS' AND 'Information Systems')?
□ Did I add LIMIT to exploratory queries?
□ Did I filter by fiscal_year for performance?
□ Did I use OR between all column searches?

KNOWN ENTITIES (use all variations in searches):
- ISS / Information Systems and Services / Information Systems / Info Systems
- HR / Human Resources
- Public Works / PW
- Finance / Financial Services
- Planning / Planning Department

OUTPUT FORMATTING:
Always format currency with dollar signs and commas (e.g., $1,234,567.89).
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
    stateless_http=os.getenv("MCP_STATELESS_HTTP", "1") == "1",
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
