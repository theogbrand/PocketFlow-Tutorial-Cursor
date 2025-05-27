import google.generativeai as genai
import PIL.Image
from typing import Union, Tuple
import logging

logger = logging.getLogger(__name__)

def extract_text_from_image(
    image_path: str, 
    api_key: str,
    specific_task: str = "Extract all text from the image"
) -> Tuple[str, bool]:
    """
    Extract and convert text within images into editable text format.
    
    Args:
        image_path: Path to the image file
        api_key: Google Gemini API key
        specific_task: Specific text extraction task or information to extract
        
    Returns:
        Tuple of (extracted_text, success)
    """
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-pro-preview-05-06')
        
        # Load image
        image = PIL.Image.open(image_path)
        
        # Create prompt for text extraction
        prompt = f"""You are a TextIntel Extractor module specialized in extracting text from images.

Task: {specific_task}

Please extract and convert all relevant text within the image into editable text format. 
Focus on:
- All visible text elements
- Labels, titles, and annotations
- Any text within charts, graphs, or diagrams
- Numerical values and their associated labels

Format the extracted text clearly and maintain the logical structure where possible.
If the task asks for specific information, prioritize extracting that information.

Output the extracted text directly without any additional commentary."""

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
        extracted_text = response.text.strip()
        
        logger.info(f"TextIntel Extractor: Successfully extracted text from {image_path}")
        return extracted_text, True
        
    except Exception as e:
        error_msg = f"TextIntel Extractor error: {str(e)}"
        logger.error(error_msg)
        return error_msg, False 