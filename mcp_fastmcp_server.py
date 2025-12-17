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

from prompt_step4 import STEP4_PREFIX

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


LEGACY_STEP4_PREFIX = """You are analyzing county budget data across three tables:

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

MANDATORY KEYWORD SEARCH STRATEGY - CRITICAL:
==============================================
**ABSOLUTE RULE: When searching for ANY organizational entity by keyword, you MUST:**
1. Search ALL FOUR text columns: "DEPARTMENT", "DIVISION", "UNIT NAME", "OBJECT NAME"
2. Search BOTH the abbreviation AND the full name(s)
3. Use ILIKE with % wildcards
4. Connect everything with OR operators

THE FOUR TEXT COLUMNS TO ALWAYS SEARCH:
1. "DEPARTMENT"
2. "DIVISION" 
3. "UNIT NAME"
4. "OBJECT NAME"

**CRITICAL: ALWAYS SEARCH BOTH ABBREVIATION AND FULL NAME**

When searching for an entity, you must include:
- The abbreviation (e.g., 'ISS', 'HR', 'PW')
- The full name (e.g., 'Information Systems and Services', 'Human Resources')
- Common variations (e.g., 'Information Systems', 'Info Systems')

**TEMPLATE - USE THIS EXACT PATTERN:**
```sql
WHERE (
    -- Search for abbreviation in all 4 columns
    "DEPARTMENT" ILIKE '%abbreviation%'
    OR "DIVISION" ILIKE '%abbreviation%'
    OR "UNIT NAME" ILIKE '%abbreviation%'
    OR "OBJECT NAME" ILIKE '%abbreviation%'
    -- Search for full name in all 4 columns
    OR "DEPARTMENT" ILIKE '%full%name%'
    OR "DIVISION" ILIKE '%full%name%'
    OR "UNIT NAME" ILIKE '%full%name%'
    OR "OBJECT NAME" ILIKE '%full%name%'
    -- Search for variations in all 4 columns
    OR "DEPARTMENT" ILIKE '%variation%'
    OR "DIVISION" ILIKE '%variation%'
    OR "UNIT NAME" ILIKE '%variation%'
    OR "OBJECT NAME" ILIKE '%variation%'
)
```

**CORRECT EXAMPLE - ISS/Information Systems:**
```sql
WHERE (
    -- Abbreviation: ISS
    "DEPARTMENT" ILIKE '%ISS%'
    OR "DIVISION" ILIKE '%ISS%'
    OR "UNIT NAME" ILIKE '%ISS%'
    OR "OBJECT NAME" ILIKE '%ISS%'
    -- Full name: Information Systems and Services
    OR "DEPARTMENT" ILIKE '%Information%Systems%Services%'
    OR "DIVISION" ILIKE '%Information%Systems%Services%'
    OR "UNIT NAME" ILIKE '%Information%Systems%Services%'
    OR "OBJECT NAME" ILIKE '%Information%Systems%Services%'
    -- Variation: Information Systems
    OR "DEPARTMENT" ILIKE '%Information%Systems%'
    OR "DIVISION" ILIKE '%Information%Systems%'
    OR "UNIT NAME" ILIKE '%Information%Systems%'
    OR "OBJECT NAME" ILIKE '%Information%Systems%'
)
```

**CORRECT EXAMPLE - HR/Human Resources:**
```sql
WHERE (
    -- Abbreviation: HR
    "DEPARTMENT" ILIKE '%HR%'
    OR "DIVISION" ILIKE '%HR%'
    OR "UNIT NAME" ILIKE '%HR%'
    OR "OBJECT NAME" ILIKE '%HR%'
    -- Full name: Human Resources
    OR "DEPARTMENT" ILIKE '%Human%Resources%'
    OR "DIVISION" ILIKE '%Human%Resources%'
    OR "UNIT NAME" ILIKE '%Human%Resources%'
    OR "OBJECT NAME" ILIKE '%Human%Resources%'
)
```

**CORRECT EXAMPLE - Public Works:**
```sql
WHERE (
    -- Abbreviation: PW
    "DEPARTMENT" ILIKE '%PW%'
    OR "DIVISION" ILIKE '%PW%'
    OR "UNIT NAME" ILIKE '%PW%'
    OR "OBJECT NAME" ILIKE '%PW%'
    -- Full name: Public Works
    OR "DEPARTMENT" ILIKE '%Public%Works%'
    OR "DIVISION" ILIKE '%Public%Works%'
    OR "UNIT NAME" ILIKE '%Public%Works%'
    OR "OBJECT NAME" ILIKE '%Public%Works%'
)
```

**INCORRECT EXAMPLES (WILL MISS DATA):**
✗ WHERE "DIVISION" ILIKE '%ISS%' 
   (Missing: other 3 columns, full name, variations)

✗ WHERE ("DEPARTMENT" ILIKE '%ISS%' OR "DIVISION" ILIKE '%ISS%' 
         OR "UNIT NAME" ILIKE '%ISS%' OR "OBJECT NAME" ILIKE '%ISS%')
   (Missing: full name "Information Systems and Services")

✗ WHERE "DIVISION" ILIKE '%Information%Systems%'
   (Missing: other 3 columns, abbreviation "ISS")

**WHY THIS MATTERS:**
- Data may use "ISS" in some records and "Information Systems and Services" in others
- Different tables or fiscal years may use different naming conventions
- "ISS" might appear in DIVISION while "Information Systems" appears in UNIT NAME
- Searching only abbreviation OR only full name will miss records

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

QUERY STRATEGY:
===============
1. Examine schema to understand available columns
2. For ANY keyword search: 
   - Identify the abbreviation (e.g., ISS, HR, PW)
   - Identify the full name (e.g., Information Systems and Services, Human Resources)
   - Identify common variations (e.g., Information Systems, Info Systems)
   - Search ALL 4 text columns for EACH variation using ILIKE with OR
3. For aggregations: Always filter by fiscal_year for performance
4. For comparisons: JOIN tables on matching keys
5. Always double-quote mixed-case column names
6. Use single quotes for string values ('2024', 'HR')
7. Add LIMIT to any exploratory or DISTINCT queries

PRE-EXECUTION CHECKLIST - READ EVERY TIME:
□ Did I search ALL FOUR text columns (DEPARTMENT, DIVISION, UNIT NAME, OBJECT NAME)?
□ Did I search the ABBREVIATION (e.g., 'ISS', 'HR')?
□ Did I search the FULL NAME (e.g., 'Information Systems and Services', 'Human Resources')?
□ Did I search COMMON VARIATIONS (e.g., 'Information Systems')?
□ Did I use ILIKE (not =) with % wildcards?
□ Did I connect all searches with OR?
□ Are ALL mixed-case column names double-quoted?
□ Are string values single-quoted?
□ Did I add LIMIT to exploratory queries?
□ Did I filter by fiscal_year for aggregation queries?

KNOWN ENTITIES - ALWAYS SEARCH ALL FORMS:
==========================================
1. ISS:
   - Abbreviation: 'ISS'
   - Full: 'Information Systems and Services'
   - Variations: 'Information Systems', 'Info Systems'

2. HR:
   - Abbreviation: 'HR'
   - Full: 'Human Resources'

3. Public Works:
   - Abbreviation: 'PW'
   - Full: 'Public Works'

4. Finance:
   - Full: 'Finance', 'Financial Services'

5. Planning:
   - Full: 'Planning', 'Planning Department'

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
    stateless_http=False,
)

# Per-process cache (note: with multiple Heroku workers, each worker has its own cache)
_agent_executor: Any | None = None
_agent_config: tuple[Any, ...] | None = None

def _get_agent_executor(
    *,
    database_url: str | None,
    model_name: str,
    temperature: float,
    sample_rows_in_table_info: int,
    verbose: bool,
    force_rebuild: bool = False,
):
    """Get or create the agent executor (built once per process by default)."""
    global _agent_executor, _agent_config

    config = (
        database_url or os.getenv("DATABASE_URL"),
        model_name,
        float(temperature),
        int(sample_rows_in_table_info),
        bool(verbose),
    )

    logger.info("Building agent executor (first initialization)")
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

    return _agent_executor

_get_agent_executor(
    database_url=os.getenv("PG_DATABASE_URL"),
    model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=float(os.getenv("OPENAI_TEMPERATURE", "0")),
    sample_rows_in_table_info=int(os.getenv("SQL_SAMPLE_ROWS_IN_TABLE_INFO", "2")),
    verbose=os.getenv("AGENT_VERBOSE", "0") == "1",
    force_rebuild=True,
)

@mcp.tool()
async def ask_budget(
    question: str,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0,
    verbose: bool = True,  # Default to True for debugging
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
        result = _agent_executor.run(question)
        
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
