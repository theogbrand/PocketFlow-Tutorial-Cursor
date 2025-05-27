import google.generativeai as genai
import PIL.Image
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

def analyze_visual_content(
    image_path: str,
    api_key: str,
    question: str
) -> Tuple[str, bool]:
    """
    Process and interpret visual data to answer queries about image content.
    
    Args:
        image_path: Path to the image file
        api_key: Google Gemini API key
        question: Question about the image content
        
    Returns:
        Tuple of (analysis_result, success)
    """
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-pro-preview-05-06')
        
        # Load image
        image = PIL.Image.open(image_path)
        
        # Create prompt for visual analysis
        prompt = f"""You are a VisionIQ Analyst module specialized in processing and interpreting visual data.

Question: {question}

Please analyze the image carefully and provide a detailed answer to the question. Consider:
- Visual elements and their relationships
- Colors, shapes, and patterns
- Context and meaning
- Any relevant details that help answer the question

Provide a clear, comprehensive answer based on what you observe in the image."""

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
        analysis_result = response.text.strip()
        
        logger.info(f"VisionIQ Analyst: Successfully analyzed {image_path}")
        return analysis_result, True
        
    except Exception as e:
        error_msg = f"VisionIQ Analyst error: {str(e)}"
        logger.error(error_msg)
        return error_msg, False 