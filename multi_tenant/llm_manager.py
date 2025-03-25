#!/usr/bin/env python
# coding=utf-8

"""
LLM management module for the multi-tenant agent framework.
"""

import os
from typing import Dict, List, Any, Optional, Union

from .base import generate_id
from .storage import save_llm, get_llm, list_llms, delete_llm

def add_llm(
    base_url: str,
    api_key: str,
    model_name: str,
    llm_name: str
) -> str:
    """
    Add a new LLM configuration.
    
    Args:
        base_url: The base URL for the LLM API
        api_key: The API key
        model_name: The model name
        llm_name: A friendly name for this LLM configuration
    
    Returns:
        The llm_id
    """
    # Validate inputs
    if not base_url:
        raise ValueError("Base URL is required")
    if not api_key:
        raise ValueError("API key is required")
    if not model_name:
        raise ValueError("Model name is required")
    if not llm_name:
        raise ValueError("LLM name is required")
    
    # Generate LLM ID
    sanitized_name = llm_name.lower().replace(' ', '_')
    llm_id = generate_id(f"{sanitized_name}_")
    
    # Save to database
    save_llm(llm_id, llm_name, base_url, api_key, model_name)
    
    return llm_id

def get_llm_config(llm_id: str) -> Dict[str, str]:
    """
    Retrieve LLM configuration.
    
    Args:
        llm_id: The LLM ID
    
    Returns:
        Dictionary with LLM configuration
    """
    # Retrieve from database
    llm_config = get_llm(llm_id)
    
    if llm_config is None:
        raise ValueError(f"LLM with ID {llm_id} not found")
    
    return {
        'llm_name': llm_config['name'],
        'base_url': llm_config['base_url'],
        'model_name': llm_config['model_name'],
        'api_key': llm_config['api_key']
    }

def get_llms() -> List[Dict[str, str]]:
    """
    List all available LLMs.
    
    Returns:
        List of dictionaries with llm_id and llm_name
    """
    return list_llms()

def create_llm_client(llm_id: str) -> Any:
    """
    Create an LLM client compatible with smolagents using OpenAIServerModel.
    
    Args:
        llm_id: The LLM ID
    
    Returns:
        A smolagents.models.OpenAIServerModel instance
    """
    # Get LLM configuration
    llm_config = get_llm_config(llm_id)
    
    # Import smolagents
    try:
        from smolagents.models import OpenAIServerModel
    except ImportError:
        raise ImportError(
            "smolagents is required to create an LLM client. "
            "Install it with: pip install smolagents"
        )
    
    # Create OpenAIServerModel with the correct parameters
    # The OpenAIServerModel constructor uses api_base instead of base_url
    model = OpenAIServerModel(
        api_base=llm_config['base_url'],  # Changed from base_url to api_base
        api_key=llm_config['api_key'],
        model_id=llm_config['model_name']
    )
    
    return model 