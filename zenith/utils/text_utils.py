import tiktoken
from loguru import logger

def get_encoding(model: str = "gpt-4o"):
    """
    Returns the appropriate tiktoken encoding for the specified model.
    Defaults to gpt-4o which uses o200k_base.
    """
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        logger.warning(f"Model {model} not found, falling back to cl100k_base.")
        return tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    Counts the number of tokens in a given text string.
    """
    if not text:
        return 0
    encoding = get_encoding(model)
    return len(encoding.encode(text))

def truncate_text_to_tokens(text: str, max_tokens: int, model: str = "gpt-4o") -> str:
    """
    Truncates a text string so it does not exceed the specified max_tokens.
    """
    if not text:
        return text
    
    encoding = get_encoding(model)
    tokens = encoding.encode(text)
    
    if len(tokens) <= max_tokens:
        return text
        
    logger.debug(f"Truncating text from {len(tokens)} to {max_tokens} tokens.")
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)
