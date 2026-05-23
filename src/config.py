# Configuration file for LLM Metrics Dashboard
# This file contains all configuration values used throughout the application

import os
from dotenv import load_dotenv

# Load environment variables from .env file
# The load_dotenv() function reads the .env file and adds its variables to os.environ
load_dotenv()

# LLM Model Configuration
# List of supported LLM models (will be expanded as we add more providers)
SUPPORTED_MODELS = [
    "gpt-5.2",            # OpenAI
    "gpt-4o-mini",        # OpenAI
    "mock-fast-model",    # Local mock model
    "mock-quality-model", # Local mock model
]

# Default model to use when no model is specified
# This is loaded from the .env file, with a fallback to a current OpenAI model
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-5.2")

# API Configuration
# These store API keys loaded from environment variables
# IMPORTANT: API keys are sensitive! Never print or expose them in the UI
API_KEYS = {
    "openai": os.getenv("OPENAI_API_KEY"),        # Loaded from .env file
    "anthropic": os.getenv("ANTHROPIC_API_KEY"),  # Loaded from .env file (if available)
}

# Pricing Configuration
# Price per 1K tokens for different models (in USD)
# Format: {"model_name": {"prompt": price_per_1k, "completion": price_per_1k}}
MODEL_PRICING = {
    "gpt-5.2": {
        "prompt": 0.0015,        # $0.0015 per 1K prompt tokens
        "completion": 0.002,     # $0.002 per 1K completion tokens
    },
    "gpt-4o-mini": {
        "prompt": 0.00015,
        "completion": 0.0006,
    },
}

# Database Configuration
# Will store the path to the SQLite database
DATABASE_PATH = "llm_metrics.db"

# Application Settings
MAX_PROMPT_LENGTH = 4000        # Maximum characters for a prompt
ENABLE_CACHING = False          # Enable response caching (future feature)
DEBUG_MODE = True               # Enable debug logging

# Mock LLM Mode Configuration
# When set to True, the app will use fake LLM responses instead of calling the real OpenAI API
# This is useful for testing and development without incurring API costs
USE_MOCK_LLM = os.getenv("USE_MOCK_LLM", "true").lower() == "true"


def is_api_key_valid(api_key: str) -> bool:
    """
    Validate that the API key is properly configured.
    
    The API key is considered INVALID if:
    - It is None or empty string
    - It equals the placeholder text "your_api_key_here"
    - It contains the word "your" (indicates it's still a placeholder)
    
    Args:
        api_key (str or None): The API key to validate
    
    Returns:
        bool: True if the API key appears to be properly configured, False otherwise
    """
    # Check if API key is None or empty
    if not api_key:
        return False
    
    # Convert to string and check for placeholder text
    api_key_str = str(api_key).strip()
    
    # Check if it's the exact placeholder
    if api_key_str.lower() == "your_api_key_here":
        return False
    
    # Check if it contains the word "your" (common placeholder pattern)
    if "your" in api_key_str.lower():
        return False
    
    # If none of the invalid conditions are met, consider it valid
    return True
