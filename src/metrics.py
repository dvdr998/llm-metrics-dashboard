# Metrics Module
# This module calculates and tracks metrics for LLM API calls

# Pricing dictionary for different models
# IMPORTANT: These are placeholder prices for demonstration purposes!
# Update these with real prices from the official provider pricing page:
# OpenAI: https://openai.com/pricing
# Anthropic: https://www.anthropic.com/pricing
# Format: {"model_name": {"input": price_per_1k_tokens, "output": price_per_1k_tokens}}
MODEL_PRICING = {
    "gpt-5.2": {
        "input": 0.0015,      # Placeholder pricing for portfolio demo
        "output": 0.002,
    },
    "gpt-4o-mini": {
        "input": 0.00015,     # Lower cost model
        "output": 0.0006,
    },
    "mock-fast-model": {
        "input": 0.001,       # Fictional pricing for mock models
        "output": 0.002,
    },
    "mock-quality-model": {
        "input": 0.003,       # Higher price for "quality"
        "output": 0.005,
    },
}


def estimate_tokens(text: str, model: str = "gpt-5.2") -> int:
    """
    Estimate how many tokens are in a piece of text.
    
    This tries to use tiktoken for a more realistic estimate. If tiktoken is
    not installed or cannot find an encoding for the model, it falls back to
    a simple approximation of 1 token per 4 characters.
    
    Args:
        text (str): The text to count
        model (str): The model name used for selecting a tokenizer
    
    Returns:
        int: Estimated token count
    """
    if not text:
        return 0
    
    try:
        import tiktoken
        
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Some mock or newer model names may not be known to tiktoken.
            encoding = tiktoken.get_encoding("cl100k_base")
        
        return len(encoding.encode(text))
    
    except Exception:
        # Fallback: estimate about 1 token per 4 characters.
        # Non-empty text should always count as at least 1 token.
        return max(1, len(text) // 4)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> dict:
    """
    Calculate the estimated cost of an API call based on token usage and model.
    
    Args:
        model (str): The LLM model used
        input_tokens (int): Number of tokens in the prompt
        output_tokens (int): Number of tokens in the response
    
    Returns:
        dict: A dictionary containing:
            - "input_cost": Estimated cost of input tokens in USD
            - "output_cost": Estimated cost of output tokens in USD
            - "total_cost": Total estimated cost in USD
    
    Note: In Mock Mode, these are simulated costs. Real costs depend on actual API provider pricing.
    """
    # Get pricing for the model, or use default pricing if model not found
    if model not in MODEL_PRICING:
        # Default to gpt-5.2 placeholder pricing if model is unknown
        pricing = MODEL_PRICING.get("gpt-5.2", {"input": 0.0015, "output": 0.002})
    else:
        pricing = MODEL_PRICING[model]
    
    # Calculate costs
    # Pricing is per 1K tokens, so divide token count by 1000
    input_cost = (input_tokens / 1000) * pricing["input"]
    output_cost = (output_tokens / 1000) * pricing["output"]
    total_cost = input_cost + output_cost
    
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }


def calculate_token_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    """
    Calculate the estimated cost of an API call based on token usage.
    (Legacy function - use estimate_cost() instead)
    
    Args:
        prompt_tokens (int): Number of tokens in the prompt
        completion_tokens (int): Number of tokens in the response
        model (str): The LLM model used
    
    Returns:
        float: Estimated cost in USD
    """
    cost_dict = estimate_cost(model, prompt_tokens, completion_tokens)
    return cost_dict["total_cost"]


def create_metric_record(
    prompt: str,
    response: str,
    model: str,
    latency_ms: float,
    prompt_tokens: int,
    completion_tokens: int,
    cost_usd: float
) -> dict:
    """
    Create a structured metric record for a single LLM API call.
    
    Args:
        prompt (str): The input prompt
        response (str): The model's response
        model (str): The model used
        latency_ms (float): Response time in milliseconds
        prompt_tokens (int): Number of prompt tokens
        completion_tokens (int): Number of completion tokens
        cost_usd (float): Estimated cost in USD
    
    Returns:
        dict: A complete metric record with all information
    
    TODO: Add timestamp to record
    TODO: Add unique ID to record
    """
    pass


def calculate_average_latency(metrics_list: list) -> float:
    """
    Calculate the average latency across multiple API calls.
    
    Args:
        metrics_list (list): List of metric records
    
    Returns:
        float: Average latency in milliseconds
    
    TODO: Implement averaging logic
    """
    pass


def calculate_total_tokens(metrics_list: list) -> dict:
    """
    Calculate total tokens used across multiple API calls.
    
    Args:
        metrics_list (list): List of metric records
    
    Returns:
        dict: Dictionary with keys:
            - "total_prompt_tokens": Total prompt tokens
            - "total_completion_tokens": Total completion tokens
            - "total_tokens": Sum of both
    
    TODO: Implement token counting logic
    """
    pass


def calculate_total_cost(metrics_list: list) -> float:
    """
    Calculate the total cost across multiple API calls.
    
    Args:
        metrics_list (list): List of metric records
    
    Returns:
        float: Total cost in USD
    
    TODO: Implement cost summing logic
    """
    pass
