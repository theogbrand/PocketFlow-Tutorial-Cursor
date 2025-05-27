import google.generativeai as genai
import PIL.Image
from typing import List, Dict, Any, Tuple
import logging
import json

logger = logging.getLogger(__name__)

def locate_objects_in_image(
    image_path: str,
    api_key: str,
    objects_to_detect: List[str]
) -> Tuple[Dict[str, Any], bool]:
    """
    Identify and locate objects within an image, counting them and determining spatial arrangement.
    
    Args:
        image_path: Path to the image file
        api_key: Google Gemini API key
        objects_to_detect: List of objects to detect and count
        
    Returns:
        Tuple of (detection_results, success)
        detection_results contains counts and spatial information for each object
    """
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-pro-preview-05-06')
        
        # Load image
        image = PIL.Image.open(image_path)
        
        # Create prompt for object detection
        objects_str = ", ".join(objects_to_detect)
        prompt = f"""You are an ObjectQuant Locator module specialized in identifying and locating objects within images.

Objects to detect: {objects_str}

For each object type listed above, please:
1. Count the total number of instances
2. Describe their spatial arrangement (e.g., clustered, scattered, aligned)
3. Note their relative positions (e.g., top-left, center, bottom-right)
4. Identify any patterns in their distribution

Output your findings in the following JSON format:
```json
{{
    "object_name": {{
        "count": number,
        "spatial_arrangement": "description",
        "positions": ["position1", "position2", ...],
        "patterns": "description of any patterns"
    }},
    ...
}}
```

Be precise in your counting and spatial descriptions."""

        # Generate response
        response = model.generate_content(
            [prompt, image],
            safety_settings={
                'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'block_none',
                'HARM_CATEGORY_HATE_SPEECH': 'block_none',
                'HARM_CATEGORY_HARASSMENT': 'block_none',
                'HARM_CATEGORY_DANGEROUS_CONTENT': 'block_none'
            }
        )
        
        response.resolve()
        response_text = response.text.strip()
        
        # Extract JSON from response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            detection_results = json.loads(json_str)
        else:
            # Fallback: try to parse the entire response
            detection_results = json.loads(response_text)
        
        logger.info(f"ObjectQuant Locator: Successfully detected objects in {image_path}")
        return detection_results, True
        
    except json.JSONDecodeError as e:
        error_msg = f"ObjectQuant Locator JSON parsing error: {str(e)}"
        logger.error(error_msg)
        # Return raw response if JSON parsing fails
        return {"raw_response": response_text, "error": error_msg}, False
    except Exception as e:
        error_msg = f"ObjectQuant Locator error: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}, False 