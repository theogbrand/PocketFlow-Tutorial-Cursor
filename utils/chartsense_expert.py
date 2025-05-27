import google.generativeai as genai
import PIL.Image
from typing import Dict, Any, Tuple
import logging
import json

logger = logging.getLogger(__name__)

def analyze_chart(
    image_path: str,
    api_key: str,
    chart_aspect: str = "Extract all data points and trends from the chart"
) -> Tuple[Dict[str, Any], bool]:
    """
    Analyze and interpret information from charts and graphs.
    
    Args:
        image_path: Path to the image file containing a chart
        api_key: Google Gemini API key
        chart_aspect: Specific aspect of the chart to analyze or question about it
        
    Returns:
        Tuple of (chart_analysis, success)
        chart_analysis contains extracted data, trends, and insights
    """
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-pro-preview-05-06')
        
        # Load image
        image = PIL.Image.open(image_path)
        
        # Create prompt for chart analysis
        prompt = f"""You are a ChartSense Expert module specialized in analyzing and interpreting charts and graphs.

Task: {chart_aspect}

Please analyze the chart/graph in the image and extract:
1. Chart type (bar, line, pie, scatter, etc.)
2. Title and axis labels
3. All data points with their values
4. Trends and patterns
5. Key insights or notable features
6. Any legends or annotations

Output your analysis in the following JSON format:
```json
{{
    "chart_type": "type of chart",
    "title": "chart title",
    "axes": {{
        "x_axis": {{
            "label": "x-axis label",
            "units": "units if any"
        }},
        "y_axis": {{
            "label": "y-axis label", 
            "units": "units if any"
        }}
    }},
    "data_points": [
        {{"label": "data label", "value": numeric_value}},
        ...
    ],
    "trends": ["trend1", "trend2", ...],
    "insights": ["insight1", "insight2", ...],
    "legends": ["legend1", "legend2", ...],
    "annotations": ["annotation1", "annotation2", ...]
}}
```

Be precise with numerical values and ensure all data is accurately extracted."""

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
            chart_analysis = json.loads(json_str)
        else:
            # Fallback: try to parse the entire response
            chart_analysis = json.loads(response_text)
        
        logger.info(f"ChartSense Expert: Successfully analyzed chart in {image_path}")
        return chart_analysis, True
        
    except json.JSONDecodeError as e:
        error_msg = f"ChartSense Expert JSON parsing error: {str(e)}"
        logger.error(error_msg)
        # Return raw response if JSON parsing fails
        return {"raw_response": response_text, "error": error_msg}, False
    except Exception as e:
        error_msg = f"ChartSense Expert error: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}, False 