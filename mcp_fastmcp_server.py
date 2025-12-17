"""
MCP service using FastMCP with optimized prompt including preloaded schema.
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

from prompt import get_formatted_prompt, generate_schema_info

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_agent_executor(
    *,
    database_url: str,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0,
    sample_rows_in_table_info: int = 3,  # Increased from 2 to 3 for better examples
    verbose: bool = False,
    use_dynamic_schema: bool = True,  # New parameter to control schema generation
):
    """Build the LangChain SQL agent executor with optimized prompt."""
    logger.info(f"Building agent with model={model_name}")
    
    # Handle Heroku postgres:// -> postgresql:// conversion
    if database_url.startswith("postgres://"):
        database_url = "postgresql://" + database_url[len("postgres://"):]
    
    # Create database connection
    db = SQLDatabase.from_uri(
        database_url,
        sample_rows_in_table_info=sample_rows_in_table_info,
    )
    
    logger.info(f"Connected to database: {db.dialect}")
    logger.info(f"Available tables: {db.get_usable_table_names()}")

    # Generate schema information dynamically if enabled
    if use_dynamic_schema:
        logger.info("Generating dynamic schema information...")
        try:
            schema_info = generate_schema_info(db)
            logger.info(f"Schema info generated successfully ({len(schema_info)} characters)")
        except Exception as e:
            logger.warning(f"Failed to generate dynamic schema: {e}. Using static schema.")
            schema_info = None
    else:
        schema_info = None
        logger.info("Using static schema information from prompt")

    # Create LLM
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
    )
    
    # Get the formatted prompt with schema information
    # Note: We don't pass dialect and top_k here since they're not in the new format
    if use_dynamic_schema and schema_info:
        agent_prefix = get_formatted_prompt(db)
    else:
        agent_prefix = get_formatted_prompt()
    
    logger.info(f"Prompt length: {len(agent_prefix)} characters")
    
    # Create agent with the optimized prefix
    agent = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="openai-tools",
        verbose=verbose,
        prefix=agent_prefix,
    )
    
    logger.info("Agent created successfully with preloaded schema")
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

# Initialize FastMCP
mcp = FastMCP("budget-sql-agent-optimized", transport_security=transport_security, port=os.getenv("PORT", 8000), host=os.getenv("HOST", "0.0.0.0"))

# Global agent cache (initialized once per process)
_agent: Any | None = None


def get_agent():
    """Get or create the agent (singleton per process)."""
    global _agent
    
    if _agent is None:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL environment variable is required")
        
        # Check if dynamic schema generation is enabled
        use_dynamic_schema = os.getenv("USE_DYNAMIC_SCHEMA", "true").lower() == "true"
        
        _agent = build_agent_executor(
            database_url=database_url,
            model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0")),
            sample_rows_in_table_info=int(os.getenv("SAMPLE_ROWS", "3")),
            verbose=os.getenv("VERBOSE", "true").lower() == "true",
            use_dynamic_schema=use_dynamic_schema,
        )
    
    return _agent


@mcp.tool()
def ask_budget(question: str) -> dict:
    """
    Answer a natural language question about budget data using SQL.
    
    This optimized version includes preloaded schema information, so the agent
    can skip the initial schema exploration steps and respond faster.
    
    The agent will:
    1. Use preloaded schema information (no need to query schema)
    2. Generate appropriate SQL queries based on the question
    3. Validate queries using the query checker
    4. Execute the queries
    5. Format and return the results
    
    Args:
        question: Natural language question about the budget data
    
    Returns:
        dict with 'success' status and either 'answer' or 'error'
    """
    logger.info("="*80)
    logger.info(f"Question: {question}")
    logger.info("="*80)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set")
        return {"success": False, "error": "OPENAI_API_KEY is not set"}
    
    try:
        agent = get_agent()
        logger.info("Invoking agent with preloaded schema...")
        
        # Run the agent
        result = agent.invoke({"input": question})
        
        # Extract the output
        if isinstance(result, dict):
            output = result.get("output", str(result))
        else:
            output = str(result)
        
        logger.info(f"Agent completed successfully")
        logger.info(f"Output: {output[:200]}...")
        
        return {"success": True, "answer": output}
        
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    # Run the MCP server
    mcp.run(transport="streamable-http")