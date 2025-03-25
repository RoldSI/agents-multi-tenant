#!/usr/bin/env python
# coding=utf-8

"""
Storage module for handling database operations in the multi-tenant agent framework.
"""

import os
import json
import sqlite3
from typing import Dict, List, Any, Optional, Union, Tuple
from contextlib import contextmanager

# Default database path
DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'database', 'multi_tenant.db')

@contextmanager
def get_db_connection(db_path: str = DEFAULT_DB_PATH):
    """
    Context manager for database connections.
    
    Args:
        db_path: Path to the SQLite database file
    
    Yields:
        SQLite connection object
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    try:
        yield conn
    finally:
        conn.close()

def initialize_database(db_path: str = DEFAULT_DB_PATH):
    """
    Initialize the database by creating necessary tables.
    
    Args:
        db_path: Path to the SQLite database file
    """
    # Get schema file
    schema_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'database', 'schema.sql')
    
    # Read schema
    with open(schema_path, 'r') as f:
        schema = f.read()
    
    # Create tables
    with get_db_connection(db_path) as conn:
        conn.executescript(schema)
        conn.commit()

# Tool storage operations
def save_tool_metadata(
    tool_id: str, 
    name: str, 
    description: str,
    dependencies: List[str],
    db_path: str = DEFAULT_DB_PATH
) -> str:
    """
    Save tool metadata to database.
    
    Args:
        tool_id: The unique ID of the tool
        name: The tool name
        description: The tool description
        dependencies: List of dependency packages
        db_path: Path to the SQLite database file
    
    Returns:
        The tool_id
    """
    # Convert dependencies to JSON
    dependencies_json = json.dumps(dependencies)
    
    with get_db_connection(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO tools 
            (tool_id, name, description, dependencies)
            VALUES (?, ?, ?, ?)
            """,
            (tool_id, name, description, dependencies_json)
        )
        conn.commit()
    
    return tool_id

def get_tool_metadata(tool_id: str, db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    """
    Get tool metadata from database.
    
    Args:
        tool_id: The unique ID of the tool
        db_path: Path to the SQLite database file
    
    Returns:
        Dictionary with tool metadata or None if not found
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.execute(
            "SELECT * FROM tools WHERE tool_id = ?",
            (tool_id,)
        )
        row = cursor.fetchone()
    
    if row is None:
        return None
    
    # Convert row to dict and parse dependencies
    tool_metadata = dict(row)
    tool_metadata['dependencies'] = json.loads(tool_metadata['dependencies'])
    
    return tool_metadata

def list_tools(db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """
    List all tools in the database.
    
    Args:
        db_path: Path to the SQLite database file
    
    Returns:
        List of tool metadata dictionaries
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.execute("SELECT tool_id, name, description FROM tools")
        tools = [dict(row) for row in cursor.fetchall()]
    
    return tools

def delete_tool(tool_id: str, db_path: str = DEFAULT_DB_PATH) -> bool:
    """
    Delete a tool from the database.
    
    Args:
        tool_id: The unique ID of the tool
        db_path: Path to the SQLite database file
    
    Returns:
        True if deleted, False if not found
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.execute("DELETE FROM tools WHERE tool_id = ?", (tool_id,))
        conn.commit()
        return cursor.rowcount > 0

# LLM storage operations
def save_llm(
    llm_id: str,
    name: str,
    base_url: str,
    api_key: str,
    model_name: str,
    db_path: str = DEFAULT_DB_PATH
) -> str:
    """
    Save LLM configuration to database.
    
    Args:
        llm_id: The unique ID of the LLM
        name: A friendly name for the LLM configuration
        base_url: The base URL for the LLM API
        api_key: The API key for the LLM
        model_name: The model name to use
        db_path: Path to the SQLite database file
    
    Returns:
        The llm_id
    """
    with get_db_connection(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO llms
            (llm_id, name, base_url, api_key, model_name)
            VALUES (?, ?, ?, ?, ?)
            """,
            (llm_id, name, base_url, api_key, model_name)
        )
        conn.commit()
    
    return llm_id

def get_llm(llm_id: str, db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    """
    Get LLM configuration from database.
    
    Args:
        llm_id: The unique ID of the LLM
        db_path: Path to the SQLite database file
    
    Returns:
        Dictionary with LLM configuration or None if not found
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.execute(
            "SELECT * FROM llms WHERE llm_id = ?",
            (llm_id,)
        )
        row = cursor.fetchone()
    
    if row is None:
        return None
    
    return dict(row)

def list_llms(db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, str]]:
    """
    List all LLMs in the database.
    
    Args:
        db_path: Path to the SQLite database file
    
    Returns:
        List of LLM configuration dictionaries
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.execute("SELECT llm_id, name FROM llms")
        llms = [dict(row) for row in cursor.fetchall()]
    
    return llms

def delete_llm(llm_id: str, db_path: str = DEFAULT_DB_PATH) -> bool:
    """
    Delete an LLM configuration from the database.
    
    Args:
        llm_id: The unique ID of the LLM
        db_path: Path to the SQLite database file
    
    Returns:
        True if deleted, False if not found
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.execute("DELETE FROM llms WHERE llm_id = ?", (llm_id,))
        conn.commit()
        return cursor.rowcount > 0 