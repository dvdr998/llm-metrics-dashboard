# Charts Module
# This module prepares saved prompt logs for dashboard charts and tables.


def sort_logs_oldest_first(prompt_logs: list) -> list:
    """
    Sort prompt logs so time-based charts move from left to right.
    
    Args:
        prompt_logs (list): Saved prompt runs from the SQLite database
    
    Returns:
        list: Prompt logs sorted from oldest to newest
    """
    return sorted(prompt_logs, key=lambda log: log["timestamp"])


def create_summary_metrics_display(prompt_logs: list) -> dict:
    """
    Calculate the dashboard summary numbers.
    
    Args:
        prompt_logs (list): Saved prompt runs from the SQLite database
    
    Returns:
        dict: Summary numbers for Streamlit metric cards
    """
    if not prompt_logs:
        return {
            "total_prompt_runs": 0,
            "average_latency": 0,
            "total_tokens": 0,
            "total_cost": 0,
        }
    
    total_prompt_runs = len(prompt_logs)
    total_latency = sum(log["latency_seconds"] for log in prompt_logs)
    total_tokens = sum(log["total_tokens"] for log in prompt_logs)
    total_cost = sum(log["total_cost"] for log in prompt_logs)
    
    return {
        "total_prompt_runs": total_prompt_runs,
        "average_latency": total_latency / total_prompt_runs,
        "total_tokens": total_tokens,
        "total_cost": total_cost,
    }


def create_latency_chart_data(prompt_logs: list) -> list:
    """
    Prepare latency data for the dashboard.
    """
    chart_rows = []
    
    for log in sort_logs_oldest_first(prompt_logs):
        chart_rows.append({
            "timestamp": log["timestamp"],
            "latency_seconds": log["latency_seconds"],
        })
    
    return chart_rows


def create_cost_chart_data(prompt_logs: list) -> list:
    """
    Prepare total cost data for the dashboard.
    """
    chart_rows = []
    
    for log in sort_logs_oldest_first(prompt_logs):
        chart_rows.append({
            "timestamp": log["timestamp"],
            "total_cost": log["total_cost"],
        })
    
    return chart_rows


def create_token_usage_chart_data(prompt_logs: list) -> list:
    """
    Prepare token usage data for the dashboard.
    """
    chart_rows = []
    
    for log in sort_logs_oldest_first(prompt_logs):
        chart_rows.append({
            "timestamp": log["timestamp"],
            "input_tokens": log["input_tokens"],
            "output_tokens": log["output_tokens"],
            "total_tokens": log["total_tokens"],
        })
    
    return chart_rows


def create_model_usage_chart_data(prompt_logs: list) -> list:
    """
    Prepare model usage counts for the dashboard.
    """
    model_counts = {}
    
    for log in prompt_logs:
        model = log["model"]
        model_counts[model] = model_counts.get(model, 0) + 1
    
    chart_rows = []
    
    for model, count in sorted(model_counts.items()):
        chart_rows.append({
            "model": model,
            "prompt_runs": count,
        })
    
    return chart_rows
