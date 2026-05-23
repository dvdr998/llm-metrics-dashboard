# LLM Client Module
# This module handles all interactions with LLM APIs (OpenAI, Anthropic, etc.)
# It also provides a mock LLM mode for testing without API costs

import random


def estimate_mock_tokens(text: str) -> int:
    """
    Estimate token count for Mock Mode.
    
    This is intentionally simple and deterministic: roughly 1 token per
    4 characters, with at least 1 token for non-empty text.
    """
    clean_text = text.strip()
    
    if not clean_text:
        return 0
    
    return max(1, len(clean_text) // 4)

# Mock response templates based on prompt topics
MOCK_RESPONSE_TEMPLATES = {
    "binary search": {
        "keywords": ["binary search"],
        "response": "Binary search is an efficient algorithm for finding a target value in a sorted list. It checks the middle item first. If the target is smaller, it continues searching the left half; if the target is larger, it searches the right half. Each step cuts the search space in half, so binary search runs in O(log n) time. It only works correctly when the data is sorted."
    },
    "linear search": {
        "keywords": ["linear search"],
        "response": "Linear search checks each item one by one until it finds the target or reaches the end of the list. It does not require sorted data, which makes it simple and flexible. Its time complexity is O(n) because, in the worst case, it may need to inspect every element."
    },
    "recursion": {
        "keywords": ["recursion", "recursive"],
        "response": "Recursion is a programming technique where a function solves a problem by calling itself on smaller versions of the same problem. A recursive solution needs a base case that stops the calls and a recursive case that moves closer to that base case. For example, factorial can be written recursively because n! depends on (n - 1)!."
    },
    "stack": {
        "keywords": ["stack", "stacks"],
        "response": "A stack is a data structure that follows last in, first out behavior. The most recent item added is the first one removed. Common operations are push to add an item, pop to remove the top item, and peek to view the top item. Stacks are useful for undo features, function calls, and expression parsing."
    },
    "queue": {
        "keywords": ["queue", "queues"],
        "response": "A queue is a data structure that follows first in, first out behavior. The first item added is the first item removed. Common operations are enqueue to add an item and dequeue to remove the next item. Queues are useful for scheduling, task processing, and breadth-first search."
    },
    "array": {
        "keywords": ["array", "arrays"],
        "response": "An array stores items in a fixed order and lets you access an item by its index. Index lookup is usually O(1), which makes arrays fast for reading known positions. Inserting or deleting in the middle can be O(n) because other elements may need to shift."
    },
    "linked list": {
        "keywords": ["linked list", "linked lists"],
        "response": "A linked list stores values in nodes, where each node points to the next node. It can grow easily and supports efficient insertion when you already have the right node reference. Accessing an item by position is slower than an array because you usually need to walk through the list from the head."
    },
    "sorting": {
        "keywords": ["sorting", "sort algorithm", "sorting algorithm", "sort"],
        "response": "Sorting means arranging values into an order, such as ascending or descending. Common algorithms include bubble sort, merge sort, quicksort, and insertion sort. Simple sorts are easier to understand but often run in O(n^2), while efficient sorts like merge sort and quicksort are commonly O(n log n)."
    },
    "time complexity": {
        "keywords": ["time complexity", "big o", "big-o", "runtime complexity"],
        "response": "Time complexity describes how an algorithm's running time grows as the input size grows. Big O notation focuses on the growth pattern, such as O(1), O(log n), O(n), O(n log n), or O(n^2). It helps compare algorithms without depending on a specific computer or programming language."
    },
    "code": {
        "keywords": ["code", "write", "implement", "function", "class", "python", "javascript"],
        "response": "def solve_problem(data):\n    \"\"\"Solve the problem with the provided data.\"\"\"\n    # Step 1: Validate input\n    if not data:\n        return None\n    \n    # Step 2: Process data\n    result = []\n    for item in data:\n        processed = item * 2  # Example processing\n        result.append(processed)\n    \n    # Step 3: Return results\n    return result\n\n# Example usage:\ninput_data = [1, 2, 3, 4, 5]\noutput = solve_problem(input_data)\nprint(f\"Result: {output}\")"
    },
    "summary": {
        "keywords": ["summary", "summarize", "summarise", "brief", "tldr", "overview"],
        "response": "Summary: This topic covers several key points. First, the foundational concept establishes the basis for understanding. Second, practical applications demonstrate real-world usage. Third, important considerations and limitations should be kept in mind. Finally, next steps for learning involve practice and experimentation. The main takeaway is that understanding fundamentals enables better decision-making in practice."
    },
    "comparison": {
        "keywords": ["compare", "comparison", "difference", "versus", "vs", "advantage"],
        "response": "Comparison: When comparing these options, consider the following factors: 1) Performance - Option A is faster for large datasets while Option B uses less memory. 2) Complexity - Option A requires more setup but offers more features, while Option B is simpler to implement. 3) Use cases - Option A excels in high-traffic scenarios, Option B works well for small-to-medium applications. The best choice depends on your specific requirements and constraints."
    },
    "explanation": {
        "keywords": ["explain", "what is", "how does", "describe", "meaning"],
        "response": "Explanation: This concept works by combining several principles. The core mechanism involves breaking down complex processes into manageable steps. Understanding the context helps clarify how each component contributes to the overall system. By studying examples and practicing with real scenarios, you develop both theoretical knowledge and practical skills. The key insight is that most complex systems are built from simple, well-understood fundamentals."
    },
}

# Default response if no specific template matches
DEFAULT_RESPONSE = "I do not have a specialized mock template for this prompt yet, but I can still respond based on the text you entered: \"{prompt}\". In Mock Mode, this fallback keeps the response tied to your prompt while avoiding a real API call."


def run_mock_llm(prompt: str, model: str) -> dict:
    """
    Run a mock LLM that returns realistic fake responses based on the prompt.
    This is useful for testing and development without making real API calls.
    
    Args:
        prompt (str): The user's prompt/question
        model (str): The LLM model name (used for display and latency simulation)
    
    Returns:
        dict: A dictionary containing:
            - "response": A fake response text tailored to the prompt
            - "prompt_tokens": Estimated number of prompt tokens
            - "completion_tokens": Estimated number of completion tokens
            - "total_tokens": Sum of prompt and completion tokens
            - "latency_ms": Simulated response time in milliseconds
            - "model": The model name used
    
    Note: Response quality and latency vary by model type.
    """
    # Normalize the prompt once so matching does not depend on capitalization,
    # leading/trailing spaces, or punctuation like question marks.
    normalized_prompt = prompt.lower().strip()
    clean_prompt = prompt.strip()
    
    # Select response based on prompt keywords.
    response_text = DEFAULT_RESPONSE.format(prompt=clean_prompt)
    
    # Check each template to find the best match
    for template_type, template_data in MOCK_RESPONSE_TEMPLATES.items():
        if any(keyword in normalized_prompt for keyword in template_data["keywords"]):
            response_text = template_data["response"]
            break
    
    # Estimate tokens with the Mock Mode helper. The app uses these same
    # values for display, cost estimates, and SQLite logging.
    prompt_tokens = estimate_mock_tokens(prompt)
    completion_tokens = estimate_mock_tokens(response_text)
    
    # Adjust latency based on model type
    if "fast" in model.lower():
        # Fast models: lower latency
        # Latency: 400ms - 800ms (0.4 - 0.8 seconds)
        latency_ms = random.uniform(400, 800)
    elif "quality" in model.lower():
        # Quality models: higher latency
        # Latency: 1300ms - 1800ms (1.3 - 1.8 seconds)
        latency_ms = random.uniform(1300, 1800)
    else:
        # Standard models: medium latency
        # Latency: 800ms - 1400ms (0.8 - 1.4 seconds)
        latency_ms = random.uniform(800, 1400)
    
    total_tokens = prompt_tokens + completion_tokens
    
    # Return the mock response in the same format as a real API response
    return {
        "response": response_text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "latency_ms": latency_ms,
        "model": model,
    }


def send_prompt_to_llm(prompt: str, model: str) -> dict:
    """
    Send a prompt to an LLM model and get a response.
    
    Args:
        prompt (str): The user's prompt/question
        model (str): The LLM model to use (e.g., "gpt-3.5-turbo")
    
    Returns:
        dict: A dictionary containing:
            - "response": The model's response text
            - "tokens_used": Total tokens used (prompt + completion)
            - "latency_ms": Response time in milliseconds
    
    TODO: Implement OpenAI API integration
    TODO: Add error handling for API failures
    TODO: Add support for multiple API providers
    """
    pass


def get_available_models() -> list:
    """
    Get a list of available LLM models.
    
    Returns:
        list: List of model names available for use
    
    TODO: Fetch models from API configuration
    """
    pass


def validate_api_credentials(api_key: str, provider: str) -> bool:
    """
    Validate that the provided API credentials are correct.
    
    Args:
        api_key (str): The API key to validate
        provider (str): The provider name (e.g., "openai")
    
    Returns:
        bool: True if credentials are valid, False otherwise
    
    TODO: Implement API validation
    """
    pass
