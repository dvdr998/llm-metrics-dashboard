import os
import time
from openai import OpenAI

from services.token_counter import count_text_tokens


def _get_value(obj, key, default=None):
    if obj is None:
        return default

    if isinstance(obj, dict):
        return obj.get(key, default)

    return getattr(obj, key, default)


def _extract_output_text(response):
    text = getattr(response, "output_text", None)

    if text:
        return text

    output = getattr(response, "output", None)

    if not output:
        return ""

    pieces = []

    for item in output:
        content = getattr(item, "content", None)

        if not content:
            continue

        for part in content:
            part_text = getattr(part, "text", None)

            if part_text:
                pieces.append(part_text)

    return "\n".join(pieces).strip()


def run_openai_response(
    prompt: str,
    model: str = "gpt-4o-mini",
    max_output_tokens: int = 150,
    temperature: float = 0.7,
    api_key: str | None = None
):
    """
    Runs an OpenAI hosted model and returns response + metrics.
    This measures hosted API latency and API token usage.
    It does not measure OpenAI's internal GPU memory.
    """
    key = api_key or os.getenv("OPENAI_API_KEY")

    if not key:
        raise ValueError("OPENAI_API_KEY is missing.")

    client = OpenAI(api_key=key)

    estimated_input_tokens = count_text_tokens(prompt, model)

    start_time = time.perf_counter()

    response = client.responses.create(
        model=model,
        input=prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens
    )

    end_time = time.perf_counter()

    latency_sec = end_time - start_time

    output_text = _extract_output_text(response)

    usage = getattr(response, "usage", None)

    input_tokens = _get_value(usage, "input_tokens", estimated_input_tokens)
    output_tokens = _get_value(usage, "output_tokens", count_text_tokens(output_text, model))
    total_tokens = _get_value(usage, "total_tokens", input_tokens + output_tokens)

    tokens_per_second = output_tokens / latency_sec if latency_sec > 0 else 0

    return {
        "response": output_text,
        "model_name": model,
        "runtime": "OpenAI hosted API",
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "total_tokens": int(total_tokens),
        "estimated_input_tokens": int(estimated_input_tokens),
        "latency_sec": round(latency_sec, 4),
        "tokens_per_second": round(tokens_per_second, 4),
    }
