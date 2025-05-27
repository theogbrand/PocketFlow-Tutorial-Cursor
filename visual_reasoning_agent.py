#!/usr/bin/env python3
"""
Visual Reasoning Agent - Standalone agent for image analysis and visual content understanding.

This agent uses four specialized modules:
1. TextIntel Extractor - Extract text from images
2. ObjectQuant Locator - Identify and locate objects  
3. VisionIQ Analyst - General visual content analysis
4. ChartSense Expert - Chart and graph analysis

The agent can be used independently or integrated hierarchically with other agents.
"""

import os
import argparse
import logging
import yaml
from datetime import datetime
from typing import List, Dict, Any, Tuple

from pocketflow import Node, Flow
from utils.call_llm import call_llm
from utils.textintel_extractor import extract_text_from_image
from utils.objectquant_locator import locate_objects_in_image
from utils.visioniq_analyst import analyze_visual_content
from utils.chartsense_expert import analyze_chart

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('visual_reasoning.log')
    ]
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger('visual_reasoning_agent')

#############################################
# Visual Reasoning Decision Node
#############################################
class VisualReasoningDecisionNode(Node):
    """
    Decides which visual reasoning module to use based on the user query and context.
    """
    def prep(self, shared: Dict[str, Any]) -> Tuple[str, str, List[Dict[str, Any]]]:
        user_query = shared.get("user_query", "")
        image_path = shared.get("image_path", "")
        visual_history = shared.get("visual_history", [])
        
        if not image_path:
            raise ValueError("No image path provided for visual reasoning")
        
        if not os.path.exists(image_path):
            raise ValueError(f"Image file not found: {image_path}")
        
        return user_query, image_path, visual_history
    
    def exec(self, inputs: Tuple[str, str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        user_query, image_path, visual_history = inputs
        logger.info(f"VisualReasoningDecisionNode: Analyzing query: {user_query}")
        
        # Format visual history
        history_str = ""
        if visual_history:
            history_str = "\nPrevious visual analysis results:\n"
            for i, action in enumerate(visual_history):
                history_str += f"\nModule {i+1}: {action['module']}\n"
                history_str += f"Task: {action['task']}\n"
                history_str += f"Result: {action.get('result', 'Pending')}\n"
        
        # Create decision prompt
        prompt = f"""You are an advanced visual reasoning agent equipped with four specialized modules to analyze and respond to queries about images:

1. TextIntel Extractor: This module extracts and converts text within images into editable text format. It's particularly useful for images containing a mix of text and graphical elements. When this module is required, specify your request as: "TextIntel Extractor: <specific task or information to extract>."

2. ObjectQuant Locator: This module identifies and locates objects within an image. It's adept at counting objects and determining their spatial arrangement. When you need this module, frame your request as: "ObjectQuant Locator: <object1, object2, ..., objectN>," listing the objects you believe need detection for further analysis.

3. VisionIQ Analyst: This module processes and interprets visual data, enabling you to ask any queries related to the image's content. When information from this module is needed, phrase your request as: "VisionIQ Analyst: <your question about the image>."

4. ChartSense Expert: This module specializes in analyzing and interpreting information from charts and graphs. It can extract data points, understand trends, and identify key components such as titles, axes, labels, and legends within a chart. When you require insights from a chart or graph, specify your request as: "ChartSense Expert: <specific aspect of the chart you're interested in or question you have about the chart>."

User query: {user_query}
Image path: {image_path}
{history_str}

When faced with this question about the image, decide:

1. If the question can be answered directly based on the information already gathered, use "synthesize" to combine the results.
2. Otherwise, choose the most appropriate module and specify the task.

Respond with a YAML object:
```yaml
action: one of: textintel, objectquant, visioniq, chartsense, synthesize
module_task: |
  The specific task for the chosen module (if not synthesize)
  For synthesize: explanation of how you'll combine the gathered information
reason: |
  Detailed explanation of why you chose this action
```"""

        response = call_llm(prompt)
        
        # Extract YAML
        yaml_content = ""
        if "```yaml" in response:
            yaml_blocks = response.split("```yaml")
            if len(yaml_blocks) > 1:
                yaml_content = yaml_blocks[1].split("```")[0].strip()
        elif "```yml" in response:
            yaml_blocks = response.split("```yml")
            if len(yaml_blocks) > 1:
                yaml_content = yaml_blocks[1].split("```")[0].strip()
        
        if yaml_content:
            decision = yaml.safe_load(yaml_content)
            
            # Validate fields
            assert "action" in decision, "Action is missing"
            assert "reason" in decision, "Reason is missing"
            
            if decision["action"] != "synthesize":
                assert "module_task" in decision, "Module task is missing"
            
            return decision
        else:
            raise ValueError("No YAML object found in response")
    
    def post(self, shared: Dict[str, Any], prep_res: Any, exec_res: Dict[str, Any]) -> str:
        logger.info(f"VisualReasoningDecisionNode: Selected action: {exec_res['action']}")
        
        # Initialize visual history if not present
        if "visual_history" not in shared:
            shared["visual_history"] = []
        
        # Store current decision
        shared["current_visual_decision"] = exec_res
        
        return exec_res["action"]

#############################################
# TextIntel Extractor Action Node
#############################################
class TextIntelExtractorAction(Node):
    """
    Extracts text from images using the TextIntel Extractor module.
    """
    def prep(self, shared: Dict[str, Any]) -> Tuple[str, str, str]:
        image_path = shared.get("image_path", "")
        api_key = shared.get("gemini_api_key", "")
        current_decision = shared.get("current_visual_decision", {})
        task = current_decision.get("module_task", "Extract all text from the image")
        
        if not api_key:
            raise ValueError("Gemini API key is required for TextIntel Extractor")
        
        return image_path, api_key, task
    
    def exec(self, inputs: Tuple[str, str, str]) -> Tuple[str, bool]:
        image_path, api_key, task = inputs
        return extract_text_from_image(image_path, api_key, task)
    
    def post(self, shared: Dict[str, Any], prep_res: Any, exec_res: Tuple[str, bool]) -> str:
        result, success = exec_res
        
        # Add to visual history
        visual_history = shared.get("visual_history", [])
        visual_history.append({
            "module": "TextIntel Extractor",
            "task": prep_res[2],  # task from prep
            "result": result,
            "success": success,
            "timestamp": datetime.now().isoformat()
        })
        shared["visual_history"] = visual_history
        
        if success:
            logger.info("TextIntelExtractorAction: Successfully extracted text")
            return "continue"
        else:
            logger.error(f"TextIntelExtractorAction failed: {result}")
            return "synthesize"  # Move to synthesis even if failed

#############################################
# ObjectQuant Locator Action Node
#############################################
class ObjectQuantLocatorAction(Node):
    """
    Locates and identifies objects in images using the ObjectQuant Locator module.
    """
    def prep(self, shared: Dict[str, Any]) -> Tuple[str, str, List[str]]:
        image_path = shared.get("image_path", "")
        api_key = shared.get("gemini_api_key", "")
        current_decision = shared.get("current_visual_decision", {})
        task = current_decision.get("module_task", "")
        
        if not api_key:
            raise ValueError("Gemini API key is required for ObjectQuant Locator")
        
        # Parse objects from the task
        objects_to_detect = []
        if ":" in task:
            objects_part = task.split(":", 1)[1].strip()
            objects_to_detect = [obj.strip() for obj in objects_part.split(",")]
        
        return image_path, api_key, objects_to_detect
    
    def exec(self, inputs: Tuple[str, str, List[str]]) -> Tuple[Dict[str, Any], bool]:
        image_path, api_key, objects_to_detect = inputs
        return locate_objects_in_image(image_path, api_key, objects_to_detect)
    
    def post(self, shared: Dict[str, Any], prep_res: Any, exec_res: Tuple[Dict[str, Any], bool]) -> str:
        result, success = exec_res
        
        # Add to visual history
        visual_history = shared.get("visual_history", [])
        visual_history.append({
            "module": "ObjectQuant Locator", 
            "task": f"Locate objects: {prep_res[2]}",  # objects from prep
            "result": result,
            "success": success,
            "timestamp": datetime.now().isoformat()
        })
        shared["visual_history"] = visual_history
        
        if success:
            logger.info("ObjectQuantLocatorAction: Successfully located objects")
            return "continue"
        else:
            logger.error(f"ObjectQuantLocatorAction failed: {result}")
            return "synthesize"

#############################################
# VisionIQ Analyst Action Node
#############################################
class VisionIQAnalystAction(Node):
    """
    Analyzes visual content using the VisionIQ Analyst module.
    """
    def prep(self, shared: Dict[str, Any]) -> Tuple[str, str, str]:
        image_path = shared.get("image_path", "")
        api_key = shared.get("gemini_api_key", "")
        current_decision = shared.get("current_visual_decision", {})
        task = current_decision.get("module_task", "")
        
        if not api_key:
            raise ValueError("Gemini API key is required for VisionIQ Analyst")
        
        # Extract question from the task
        question = task
        if ":" in task:
            question = task.split(":", 1)[1].strip()
        
        return image_path, api_key, question
    
    def exec(self, inputs: Tuple[str, str, str]) -> Tuple[str, bool]:
        image_path, api_key, question = inputs
        return analyze_visual_content(image_path, api_key, question)
    
    def post(self, shared: Dict[str, Any], prep_res: Any, exec_res: Tuple[str, bool]) -> str:
        result, success = exec_res
        
        # Add to visual history
        visual_history = shared.get("visual_history", [])
        visual_history.append({
            "module": "VisionIQ Analyst",
            "task": prep_res[2],  # question from prep
            "result": result,
            "success": success,
            "timestamp": datetime.now().isoformat()
        })
        shared["visual_history"] = visual_history
        
        if success:
            logger.info("VisionIQAnalystAction: Successfully analyzed visual content")
            return "continue"
        else:
            logger.error(f"VisionIQAnalystAction failed: {result}")
            return "synthesize"

#############################################
# ChartSense Expert Action Node
#############################################
class ChartSenseExpertAction(Node):
    """
    Analyzes charts and graphs using the ChartSense Expert module.
    """
    def prep(self, shared: Dict[str, Any]) -> Tuple[str, str, str]:
        image_path = shared.get("image_path", "")
        api_key = shared.get("gemini_api_key", "")
        current_decision = shared.get("current_visual_decision", {})
        task = current_decision.get("module_task", "")
        
        if not api_key:
            raise ValueError("Gemini API key is required for ChartSense Expert")
        
        # Extract chart aspect from the task
        chart_aspect = task
        if ":" in task:
            chart_aspect = task.split(":", 1)[1].strip()
        
        return image_path, api_key, chart_aspect
    
    def exec(self, inputs: Tuple[str, str, str]) -> Tuple[Dict[str, Any], bool]:
        image_path, api_key, chart_aspect = inputs
        return analyze_chart(image_path, api_key, chart_aspect)
    
    def post(self, shared: Dict[str, Any], prep_res: Any, exec_res: Tuple[Dict[str, Any], bool]) -> str:
        result, success = exec_res
        
        # Add to visual history
        visual_history = shared.get("visual_history", [])
        visual_history.append({
            "module": "ChartSense Expert",
            "task": prep_res[2],  # chart_aspect from prep
            "result": result,
            "success": success,
            "timestamp": datetime.now().isoformat()
        })
        shared["visual_history"] = visual_history
        
        if success:
            logger.info("ChartSenseExpertAction: Successfully analyzed chart")
            return "continue"
        else:
            logger.error(f"ChartSenseExpertAction failed: {result}")
            return "synthesize"

#############################################
# Visual Synthesis Node
#############################################
class VisualSynthesisNode(Node):
    """
    Synthesizes results from multiple visual reasoning modules into a final answer.
    """
    def prep(self, shared: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
        user_query = shared.get("user_query", "")
        visual_history = shared.get("visual_history", [])
        return user_query, visual_history
    
    def exec(self, inputs: Tuple[str, List[Dict[str, Any]]]) -> str:
        user_query, visual_history = inputs
        
        if not visual_history:
            return "No visual analysis was performed. Please provide an image and try again."
        
        # Format the visual analysis results
        analysis_summary = "Visual Analysis Results:\n\n"
        for i, action in enumerate(visual_history):
            analysis_summary += f"{i+1}. {action['module']}\n"
            analysis_summary += f"   Task: {action['task']}\n"
            analysis_summary += f"   Success: {action['success']}\n"
            if action['success']:
                analysis_summary += f"   Result: {action['result']}\n"
            else:
                analysis_summary += f"   Error: {action['result']}\n"
            analysis_summary += "\n"
        
        # Create synthesis prompt
        prompt = f"""You are synthesizing the results of visual analysis modules to answer a user's question about an image.

User Question: {user_query}

{analysis_summary}

Based on the visual analysis results above, provide a comprehensive answer to the user's question. 

Guidelines:
- Synthesize information from all successful modules
- If modules failed, work with the available information
- Be specific and reference the visual elements observed
- If the question asks for a specific format (like multiple choice), follow that format
- If you need to make conclusions, clearly state your reasoning

Provide a clear, well-structured answer:"""

        # Generate synthesis
        response = call_llm(prompt)
        return response.strip()
    
    def post(self, shared: Dict[str, Any], prep_res: Any, exec_res: str) -> str:
        # Store the final synthesis result
        shared["visual_response"] = exec_res
        logger.info("VisualSynthesisNode: Successfully synthesized visual analysis results")
        return "complete"

#############################################
# Flow Creation
#############################################
def create_visual_reasoning_flow() -> Flow:
    """
    Create the visual reasoning flow with all necessary nodes and transitions.
    """
    # Create nodes
    decision_node = VisualReasoningDecisionNode()
    textintel_node = TextIntelExtractorAction()
    objectquant_node = ObjectQuantLocatorAction()
    visioniq_node = VisionIQAnalystAction()
    chartsense_node = ChartSenseExpertAction()
    synthesis_node = VisualSynthesisNode()
    
    # Set up transitions
    # From decision node to specific modules
    decision_node - "textintel" >> textintel_node
    decision_node - "objectquant" >> objectquant_node
    decision_node - "visioniq" >> visioniq_node
    decision_node - "chartsense" >> chartsense_node
    decision_node - "synthesize" >> synthesis_node
    
    # From modules back to decision or to synthesis
    textintel_node - "continue" >> decision_node
    textintel_node - "synthesize" >> synthesis_node
    objectquant_node - "continue" >> decision_node
    objectquant_node - "synthesize" >> synthesis_node
    visioniq_node - "continue" >> decision_node
    visioniq_node - "synthesize" >> synthesis_node
    chartsense_node - "continue" >> decision_node
    chartsense_node - "synthesize" >> synthesis_node
    
    # Create flow
    return Flow(start=decision_node)

#############################################
# Main Interface
#############################################
def run_visual_reasoning(
    user_query: str, 
    image_path: str, 
    gemini_api_key: str = None
) -> Dict[str, Any]:
    """
    Run visual reasoning analysis on an image.
    
    Args:
        user_query: Question or request about the image
        image_path: Path to the image file
        gemini_api_key: Google Gemini API key (defaults to environment variable)
    
    Returns:
        Dictionary containing analysis results and metadata
    """
    # Get API key from environment if not provided
    if not gemini_api_key:
        gemini_api_key = os.getenv("GEMINI_API_KEY", "")
    
    if not gemini_api_key:
        raise ValueError("Gemini API key is required. Set GEMINI_API_KEY environment variable or pass it directly.")
    
    # Validate image file
    if not os.path.exists(image_path):
        raise ValueError(f"Image file not found: {image_path}")
    
    # Initialize shared state
    shared = {
        "user_query": user_query,
        "image_path": image_path,
        "gemini_api_key": gemini_api_key,
        "visual_history": [],
        "visual_response": None
    }
    
    # Create and run the flow
    flow = create_visual_reasoning_flow()
    flow.run(shared)
    
    # Return results
    return {
        "query": user_query,
        "image_path": image_path,
        "response": shared.get("visual_response", "No response generated"),
        "visual_history": shared.get("visual_history", []),
        "success": shared.get("visual_response") is not None
    }

def main():
    """
    Command-line interface for the visual reasoning agent.
    """
    parser = argparse.ArgumentParser(description="Visual Reasoning Agent - AI-powered image analysis")
    parser.add_argument("--query", "-q", type=str, required=True, 
                        help="Question or request about the image")
    parser.add_argument("--image", "-i", type=str, required=True,
                        help="Path to the image file to analyze")
    parser.add_argument("--api-key", "-k", type=str, 
                        help="Google Gemini API key (defaults to GEMINI_API_KEY env var)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Run visual reasoning
        result = run_visual_reasoning(
            user_query=args.query,
            image_path=args.image,
            gemini_api_key=args.api_key
        )
        
        # Print results
        print("\n" + "="*60)
        print("VISUAL REASONING ANALYSIS RESULTS")
        print("="*60)
        print(f"\nQuery: {result['query']}")
        print(f"Image: {result['image_path']}")
        print(f"\nResponse:\n{result['response']}")
        
        if args.verbose and result['visual_history']:
            print(f"\n\nDetailed Analysis History:")
            print("-" * 40)
            for i, step in enumerate(result['visual_history']):
                print(f"\nStep {i+1}: {step['module']}")
                print(f"Task: {step['task']}")
                print(f"Success: {step['success']}")
                if step['success']:
                    print(f"Result: {step['result']}")
                else:
                    print(f"Error: {step['result']}")
        
        print("\n" + "="*60)
        
    except Exception as e:
        logger.error(f"Error in visual reasoning: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 