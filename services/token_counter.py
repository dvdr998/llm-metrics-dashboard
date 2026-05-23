import tiktoken


def count_text_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """
    Counts tokens for normal text using tiktoken.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))


def count_chat_tokens(messages: list, model: str = "gpt-4o-mini") -> int:
    """
    Estimates tokens for chat-style messages.
    The final OpenAI API usage value is still the most accurate source.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens_per_message = 3
    tokens_per_name = 1

    total_tokens = 0

    for message in messages:
        total_tokens += tokens_per_message

        for key, value in message.items():
            total_tokens += len(encoding.encode(str(value)))

            if key == "name":
                total_tokens += tokens_per_name

    total_tokens += 3
    return total_tokens
