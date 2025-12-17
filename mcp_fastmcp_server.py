"""
MCP StreamableHTTP service (Python) using FastMCP.

Exposes a single tool that runs a LangChain SQL agent against the configured DB.
"""

from __future__ import annotations

import os
import logging
from typing import Any

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

STEP4_PREFIX = """You are analyzing county budget data across three tables:

1. actual_expenses: Contains actual spending by fiscal year
2. adopted_budget: Contains adopted/planned budgets by fiscal year
3. budget_roll: Contains budget rollover amounts by fiscal year

Key information:
- fiscal_year column contains the year (2019-2025)
- DEPARTMENT and DIVISION identify organizational units
- UNIT and UNIT NAME are more specific organizational identifiers
- OBJECT CODE and OBJECT NAME categorize expense types
- OBJECT GROUP categorizes expenses into Salaries, Operating, Capital, etc.
- amount column contains the dollar values
- APPROPRIATION indicates the budget category

Common department abbreviations:
- ISS = Information Systems Services
- HR = Human Resources

When asked for breakdowns by category (salaries, operating, capital), use the OBJECT GROUP column.
When filtering by department, check both DEPARTMENT and UNIT NAME columns.
Always format currency amounts clearly with dollar signs and commas.
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
        "postgresql://postgres:postgres@localhost:5432/ocfl"
    )
    
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

# Simple global cache - no threading needed
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
    """Get or create the agent executor (with simple caching)."""
    global _agent_executor, _agent_config

    config = (
        database_url or os.getenv("DATABASE_URL"),
        model_name,
        float(temperature),
        int(sample_rows_in_table_info),
        bool(verbose),
    )

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
def ask_budget(
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
        # Get the agent executor
        agent_executor = _get_agent_executor(
            database_url=database_url,
            model_name=model_name,
            temperature=temperature,
            sample_rows_in_table_info=sample_rows_in_table_info,
            verbose=verbose,
            force_rebuild=force_rebuild,
        )
        
        logger.info(f"Invoking agent with question: {question}")
        
        # Invoke the agent synchronously
        result = agent_executor.invoke({"input": question})
        
        logger.info(f"Agent completed. Result type: {type(result)}")
        
        # Extract output
        output = result.get("output") if isinstance(result, dict) else str(result)
        
        logger.info(f"Output length: {len(output)} characters")
        logger.info(f"Output preview: {output[:200]}...")
        
        return {"ok": True, "output": output}
        
    except Exception as e:
        logger.error(f"Error in ask_budget: {str(e)}", exc_info=True)
        return {"ok": False, "error": str(e)}


if __name__ == "__main__":
    logger.info("Starting MCP server...")
    mcp.run(transport="streamable-http")