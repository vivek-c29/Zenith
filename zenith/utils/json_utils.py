import json
from typing import Any, Dict, Optional
from loguru import logger
import json_repair

def safe_parse_json(json_string: str) -> Optional[Dict[str, Any]]:
    """
    Safely parses a JSON string, attempting to repair it if it's malformed
    (often the case with LLM outputs).
    
    Returns a dictionary if parsing/repair was successful, None otherwise.
    """
    if not json_string or not isinstance(json_string, str):
        logger.warning("safe_parse_json received empty or non-string input.")
        return None
        
    try:
        # First attempt a standard JSON parse (fastest)
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        logger.debug(f"Standard JSON parse failed: {e}. Attempting repair.")
        
        try:
            # Attempt repair using json_repair
            repaired = json_repair.repair_json(json_string, return_objects=True)
            if isinstance(repaired, dict) or isinstance(repaired, list):
                return repaired
            
            logger.warning(f"json_repair did not return a dict/list. Result type: {type(repaired)}")
            return None
            
        except Exception as repair_error:
            logger.error(f"Failed to repair JSON: {repair_error}\nInput snippet: {json_string[:100]}...")
            return None
