# Database Module
# This module handles all database operations for storing and retrieving metrics
# Uses SQLite for local data persistence

import sqlite3
from datetime import datetime

# Path to the SQLite database file
DATABASE_PATH = "llm_metrics.db"


def initialize_database() -> bool:
    """
    Initialize the SQLite database and create the prompt_logs table if it doesn't exist.
    
    The table stores detailed information about each LLM API call:
    - id: Unique identifier for each log
    - timestamp: When the request was made
    - prompt: The user's input prompt
    - response: The model's response
    - model: The model name used
    - temperature: Temperature setting used
    - latency_seconds: How long it took to get a response
    - input_tokens: Number of tokens in the prompt
    - output_tokens: Number of tokens in the response
    - total_tokens: Total tokens (input + output)
    - input_cost: Estimated cost for input tokens
    - output_cost: Estimated cost for output tokens
    - total_cost: Total estimated cost
    - mode: "mock" or "real" to indicate if this was a real API call or mock
    
    Returns:
        bool: True if initialization was successful, False otherwise
    """
    try:
        # Connect to the SQLite database. SQLite creates the file automatically
        # if it does not already exist.
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            
            # Create the prompt_logs table if it doesn't already exist.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prompt_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    response TEXT NOT NULL,
                    model TEXT NOT NULL,
                    temperature REAL NOT NULL,
                    latency_seconds REAL NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    input_cost REAL NOT NULL,
                    output_cost REAL NOT NULL,
                    total_cost REAL NOT NULL,
                    mode TEXT NOT NULL
                )
            """)
            
            # Commit the table creation.
            conn.commit()
        
        return True
    
    except Exception as e:
        # If there's an error, print it for debugging
        print(f"Error initializing database: {e}")
        return False


def save_prompt_log(
    prompt: str,
    response: str,
    model: str,
    temperature: float,
    latency_seconds: float,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    input_cost: float,
    output_cost: float,
    total_cost: float,
    mode: str = "mock"
) -> bool:
    """
    Save a single prompt run to the database.
    
    Args:
        prompt (str): The user's input prompt
        response (str): The model's response
        model (str): The model name used (e.g., "gpt-5.2")
        temperature (float): Temperature setting used (0.0 - 1.0)
        latency_seconds (float): Response time in seconds
        input_tokens (int): Number of tokens in the prompt
        output_tokens (int): Number of tokens in the response
        total_tokens (int): Total tokens used
        input_cost (float): Estimated cost for input in USD
        output_cost (float): Estimated cost for output in USD
        total_cost (float): Total estimated cost in USD
        mode (str): "mock" for mock LLM, "real" for real API
    
    Returns:
        bool: True if save was successful, False otherwise
    """
    try:
        # Get current timestamp
        timestamp = datetime.now().isoformat()
        
        # Connect to the database and insert one prompt run.
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            
            # Question marks are placeholders. They keep the data separate
            # from the SQL statement and make inserts safer.
            cursor.execute("""
                INSERT INTO prompt_logs (
                    timestamp, prompt, response, model, temperature,
                    latency_seconds, input_tokens, output_tokens, total_tokens,
                    input_cost, output_cost, total_cost, mode
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp, prompt, response, model, temperature,
                latency_seconds, input_tokens, output_tokens, total_tokens,
                input_cost, output_cost, total_cost, mode
            ))
            
            # Commit the new row.
            conn.commit()
        
        return True
    
    except Exception as e:
        # If there's an error, print it for debugging
        print(f"Error saving prompt log: {e}")
        return False


def get_all_prompt_logs() -> list:
    """
    Retrieve all prompt logs from the database.
    
    Returns:
        list: List of all prompt log records as dictionaries
    """
    try:
        # Connect to the database and return rows that behave like dictionaries.
        with sqlite3.connect(DATABASE_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Query all logs from the database, ordered by most recent first.
            cursor.execute("SELECT * FROM prompt_logs ORDER BY timestamp DESC")
            logs = [dict(row) for row in cursor.fetchall()]
        
        return logs
    
    except Exception as e:
        # If there's an error, print it for debugging
        print(f"Error retrieving prompt logs: {e}")
        return []


def clear_prompt_logs() -> bool:
    """
    Delete all saved prompt logs from the database.
    
    Returns:
        bool: True if delete was successful, False otherwise
    """
    try:
        # Delete the saved rows, but keep the prompt_logs table ready for
        # future prompt runs.
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM prompt_logs")
            conn.commit()
        
        return True
    
    except Exception as e:
        # If there's an error, print it for debugging
        print(f"Error deleting prompt logs: {e}")
        return False


def delete_all_prompt_logs() -> bool:
    """
    Backward-compatible wrapper for clearing prompt logs.
    
    New code should use clear_prompt_logs().
    """
    return clear_prompt_logs()


def get_metrics_by_model(model: str) -> list:
    """
    Retrieve metric records for a specific model.
    
    Args:
        model (str): Model name to filter by
    
    Returns:
        list: List of metric records for that model
    
    TODO: Query metrics table with model filter
    """
    pass


def get_metrics_by_date_range(start_date, end_date) -> list:
    """
    Retrieve metric records within a specific date range.
    
    Args:
        start_date: Start date for filtering
        end_date: End date for filtering
    
    Returns:
        list: List of metric records within the date range
    
    TODO: Query metrics table with date filter
    """
    pass


def delete_old_metrics(days: int) -> int:
    """
    Delete metric records older than a specified number of days.
    
    Args:
        days (int): Number of days to keep (delete older)
    
    Returns:
        int: Number of records deleted
    
    TODO: Delete records with timestamp older than days
    """
    pass


def close_database():
    """
    Close the database connection.
    
    TODO: Implement connection cleanup
    """
    pass
