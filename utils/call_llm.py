import os
import logging
import json
from datetime import datetime
from typing import Union, List, Dict, Any

# Configure logging
log_directory = os.getenv("LOG_DIR", "logs")
os.makedirs(log_directory, exist_ok=True)
log_file = os.path.join(log_directory, f"llm_calls_{datetime.now().strftime('%Y%m%d')}.log")

# Set up logger
logger = logging.getLogger("llm_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Simple cache configuration
cache_file = "llm_cache.json"

# Learn more about calling the LLM: https://the-pocket.github.io/PocketFlow/utility_function/llm.html
def call_llm(
    prompt: Union[str, List[Dict[str, str]]], 
    model: str = "claude-sonnet-4-20250514",
    use_cache: bool = True,
    reasoning_effort: str = "high"
) -> str:
    """
    Makes API calls to language models using LiteLLM for multi-provider support.
    
    Supported Anthropic models via Vertex AI:
        - "vertex_ai/claude-3-7-sonnet@20250219" (default)
        - "vertex_ai/claude-3-opus@20240229"
        - "vertex_ai/claude-3-haiku@20240307"
        - "vertex_ai/claude-3-sonnet@20240229"
    
    You can also use other providers:
        - OpenAI: "gpt-4o", "gpt-4", "gpt-3.5-turbo"
        - Direct Anthropic: "claude-3-opus-20240229", "claude-3-sonnet-20240229"
        - Gemini: "gemini-1.5-pro", "gemini-1.5-flash"
    
    Required environment variables:
        For Vertex AI:
        - ANTHROPIC_REGION: e.g., "us-east5"
        - ANTHROPIC_PROJECT_ID: Your GCP project ID
        
        For other providers:
        - OPENAI_API_KEY: For OpenAI models
        - ANTHROPIC_API_KEY: For direct Anthropic API
        - GOOGLE_API_KEY: For Gemini models
    
    Args:
        prompt: Either a string prompt or list of messages
        model: Model name to use (default: vertex_ai/claude-3-7-sonnet@20250219)
        use_cache: Whether to use caching (default: True)
        reasoning_effort: Level of reasoning - "low", "medium", or "high" (default)
                         Maps to thinking token budgets:
                         - "low": 1024 tokens
                         - "medium": 4096 tokens  
                         - "high": 16000 tokens
    
    Returns:
        LLM response text
    """
    try:
        import litellm
    except ImportError:
        raise ImportError("Please install litellm: pip install litellm")
    
    # Log the prompt
    logger.info(f"PROMPT: {prompt}")
    logger.info(f"MODEL: {model}, REASONING: {reasoning_effort}")
    
    # Convert string prompt to messages format if needed
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = prompt
    
    # Create cache key from prompt and model
    cache_key = f"{model}:{str(prompt)}"
    
    # Check cache if enabled
    if use_cache:
        # Load cache from disk
        cache = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache = json.load(f)
            except:
                logger.warning(f"Failed to load cache, starting with empty cache")
        
        # Return from cache if exists
        if cache_key in cache:
            logger.info(f"Cache hit for prompt: {str(prompt)[:50]}...")
            return cache[cache_key]
    
    # Prepare completion parameters
    completion_params = {
        "model": model,
        "messages": messages,
        "max_tokens": 20000,
    }
    
    # Set up Vertex AI credentials if using vertex_ai models
    if model.startswith("vertex_ai/"):
        completion_params["vertex_project"] = os.getenv("ANTHROPIC_PROJECT_ID", "your-project-id")
        completion_params["vertex_location"] = os.getenv("ANTHROPIC_REGION", "us-east5")
    
    # Add thinking configuration for Claude models
    if "claude" in model.lower() and "3-7" in model:
        # Map reasoning effort to thinking budget
        thinking_budgets = {
            "low": 1024,
            "medium": 4096,
            "high": 16000
        }
        completion_params["thinking"] = {
            "type": "enabled",
            "budget_tokens": thinking_budgets.get(reasoning_effort, 16000)
        }
        # Temperature must be 1.0 when using thinking
        completion_params["temperature"] = 1.0
    elif reasoning_effort and reasoning_effort.lower() in ["low", "medium", "high"]:
        # For models that support reasoning_effort parameter
        completion_params["reasoning_effort"] = reasoning_effort.lower()
        completion_params["temperature"] = 1.0
    else:
        # Default temperature for non-reasoning models
        completion_params["temperature"] = 0.1
    
    # Call the LLM
    response = litellm.completion(**completion_params)
    
    # Extract response text
    response_text = response.choices[0].message.content
    
    # Log the response
    logger.info(f"RESPONSE: {response_text}")
    
    # Update cache if enabled
    if use_cache:
        # Load cache again to avoid overwrites
        cache = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache = json.load(f)
            except:
                pass
        
        # Add to cache and save
        cache[cache_key] = response_text
        try:
            with open(cache_file, 'w') as f:
                json.dump(cache, f, indent=2)
            logger.info(f"Added to cache")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    return response_text

def clear_cache() -> None:
    """Clear the cache file if it exists."""
    if os.path.exists(cache_file):
        os.remove(cache_file)
        logger.info("Cache cleared")

if __name__ == "__main__":
    # Test different models
    test_prompt = "Hello, how are you?"
    
    # Test Vertex AI Claude models
    vertex_models = [
        "vertex_ai/claude-3-7-sonnet@20250219",
        # "vertex_ai/claude-3-opus@20240229",
        # "vertex_ai/claude-3-haiku@20240307",
    ]
    
    for model in vertex_models:
        print(f"\nTesting {model}...")
        try:
            # First call - no cache
            response1 = call_llm(test_prompt, model=model, use_cache=False)
            print(f"Response (no cache): {response1[:100]}...")
            
            # Second call - should use cache
            response2 = call_llm(test_prompt, model=model, use_cache=True)
            print(f"Response (cached): {response2[:100]}...")
        except Exception as e:
            print(f"Error with {model}: {e}")
    
    # Test with different reasoning efforts
    print("\n\nTesting reasoning efforts...")
    complex_prompt = "Solve step by step: If a train travels 120 miles in 2 hours, then increases speed by 25% for 3 hours, how far total?"
    
    for effort in ["low", "medium", "high"]:
        print(f"\nTesting with {effort} reasoning effort...")
        try:
            response = call_llm(
                complex_prompt, 
                model="vertex_ai/claude-3-7-sonnet@20250219",
                use_cache=False,
                reasoning_effort=effort
            )
            print(f"Response ({effort}): {response[:200]}...")
        except Exception as e:
            print(f"Error with {effort} reasoning: {e}")
