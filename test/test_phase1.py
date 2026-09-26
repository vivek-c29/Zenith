from zenith.config import settings, setup_logger
from zenith.utils import count_tokens, safe_parse_json

def run_tests():
    # Test 1: Logger and Settings
    logger = setup_logger()
    logger.info(f"Loaded Settings - API_PORT: {settings.API_PORT}")
    assert settings.API_PORT == 8000, "Default port should be 8000"
    logger.success("Test 1 Passed: Logger and Settings initialized successfully.")

    # Test 2: Token Counting
    text = "Hello, world! This is Zenith, the AI agent."
    tokens = count_tokens(text)
    logger.info(f"Token count for '{text}': {tokens}")
    assert tokens > 0, "Token count should be greater than 0"
    logger.success("Test 2 Passed: Token counting works.")

    # Test 3: JSON Repair
    bad_json_string = '{"name": "Zenith", "type": "Agent",}'  # trailing comma
    parsed = safe_parse_json(bad_json_string)
    logger.info(f"Repaired JSON: {parsed}")
    assert parsed is not None, "Parsed JSON should not be None"
    assert parsed.get("name") == "Zenith", "Parsed JSON should have correct name"
    logger.success("Test 3 Passed: JSON repair works.")

if __name__ == "__main__":
    run_tests()
