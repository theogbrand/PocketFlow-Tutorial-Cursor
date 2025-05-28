#!/usr/bin/env python3
"""
Visual Reasoning Verifier - Evaluates the validity of intermediate reasoning steps in multimodal tasks.

This verifier uses specialized modules to analyze if a given reasoning step is valid:
1. TextIntel Extractor - Extract text from images
2. ObjectQuant Locator - Identify and locate objects  
3. VisionIQ Analyst - General visual content analysis
4. ChartSense Expert - Chart and graph analysis

The verifier outputs a binary judgment on the validity of the reasoning step.
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
        logging.FileHandler('visual_verification.log')
    ]
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger('visual_reasoning_verifier')

#############################################
# Verification Decision Node
#############################################
class VerificationDecisionNode(Node):
    """
    Decides which visual reasoning module to use to verify a reasoning step.
    """
    def prep(self, shared: Dict[str, Any]) -> Tuple[str, str, str]:
        reasoning_step = shared.get("reasoning_step", "")
        image_path = shared.get("image_path", "")
        verification_history = shared.get("verification_history", [])
        
        if not reasoning_step:
            raise ValueError("No reasoning step provided for verification")
            
        if not image_path:
            raise ValueError("No image path provided for verification")
        
        if not os.path.exists(image_path):
            raise ValueError(f"Image file not found: {image_path}")
        
        return reasoning_step, image_path, str(verification_history)
    
    def exec(self, inputs: Tuple[str, str, str]) -> Dict[str, Any]:
        reasoning_step, image_path, verification_history_str = inputs
        logger.info(f"VerificationDecisionNode: Analyzing reasoning step: {reasoning_step[:50]}...")
        
        # Create decision prompt
        prompt = f"""You are a verification agent that evaluates the validity of intermediate reasoning steps in multimodal tasks.

You have access to four specialized modules to analyze images:

1. TextIntel Extractor: Extracts and converts text within images into editable text format.
2. ObjectQuant Locator: Identifies and locates objects within an image.
3. VisionIQ Analyst: Processes and interprets visual data for general content analysis.
4. ChartSense Expert: Analyzes and interprets charts and graphs.

Current task:
- Image path: {image_path}
- Reasoning step to verify: "{reasoning_step}"
- Previous verification steps: {verification_history_str}

To verify this reasoning step, decide:
1. If you have enough information already, use "synthesize" to make a judgment.
2. Otherwise, choose the most appropriate module to gather evidence.

Respond with a YAML object:
```yaml
action: one of: textintel, objectquant, visioniq, chartsense, synthesize
module_task: |
  The specific task for the chosen module (if not synthesize)
  For synthesize: explanation of how you'll evaluate the reasoning step
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
        logger.info(f"VerificationDecisionNode: Selected action: {exec_res['action']}")
        
        # Initialize verification history if not present
        if "verification_history" not in shared:
            shared["verification_history"] = []
        
        # Store current decision
        shared["current_verification_decision"] = exec_res
        
        return exec_res["action"]

#############################################
# TextIntel Extractor Action Node
#############################################
class TextIntelExtractorVerification(Node):
    """
    Uses TextIntel Extractor to verify text-related reasoning steps.
    """
    def prep(self, shared: Dict[str, Any]) -> Tuple[str, str, str, str]:
        image_path = shared.get("image_path", "")
        api_key = shared.get("gemini_api_key", "")
        reasoning_step = shared.get("reasoning_step", "")
        current_decision = shared.get("current_verification_decision", {})
        task = current_decision.get("module_task", "Extract all text from the image")
        
        if not api_key:
            raise ValueError("Gemini API key is required for TextIntel Extractor")
        
        return image_path, api_key, task, reasoning_step
    
    def exec(self, inputs: Tuple[str, str, str, str]) -> Tuple[str, bool, str]:
        image_path, api_key, task, reasoning_step = inputs
        result, success = extract_text_from_image(image_path, api_key, task)
        
        # Add verification context
        verification_context = f"Reasoning step to verify: {reasoning_step}\n\nExtracted text from image: {result}"
        
        return result, success, verification_context
    
    def post(self, shared: Dict[str, Any], prep_res: Any, exec_res: Tuple[str, bool, str]) -> str:
        result, success, verification_context = exec_res
        
        # Add to verification history
        verification_history = shared.get("verification_history", [])
        verification_history.append({
            "module": "TextIntel Extractor",
            "task": prep_res[2],  # task from prep
            "result": result,
            "success": success,
            "verification_context": verification_context,
            "timestamp": datetime.now().isoformat()
        })
        shared["verification_history"] = verification_history
        
        if success:
            logger.info("TextIntelExtractorVerification: Successfully extracted text")
            return "continue"
        else:
            logger.error(f"TextIntelExtractorVerification failed: {result}")
            return "synthesize"  # Move to synthesis even if failed

#############################################
# ObjectQuant Locator Action Node
#############################################
class ObjectQuantLocatorVerification(Node):
    """
    Uses ObjectQuant Locator to verify object-related reasoning steps.
    """
    def prep(self, shared: Dict[str, Any]) -> Tuple[str, str, List[str], str]:
        image_path = shared.get("image_path", "")
        api_key = shared.get("gemini_api_key", "")
        reasoning_step = shared.get("reasoning_step", "")
        current_decision = shared.get("current_verification_decision", {})
        task = current_decision.get("module_task", "")
        
        if not api_key:
            raise ValueError("Gemini API key is required for ObjectQuant Locator")
        
        # Parse objects from the task
        objects_to_detect = []
        if ":" in task:
            objects_part = task.split(":", 1)[1].strip()
            objects_to_detect = [obj.strip() for obj in objects_part.split(",")]
        
        return image_path, api_key, objects_to_detect, reasoning_step
    
    def exec(self, inputs: Tuple[str, str, List[str], str]) -> Tuple[Dict[str, Any], bool, str]:
        image_path, api_key, objects_to_detect, reasoning_step = inputs
        result, success = locate_objects_in_image(image_path, api_key, objects_to_detect)
        
        # Add verification context
        verification_context = f"Reasoning step to verify: {reasoning_step}\n\nObjects detected in image: {result}"
        
        return result, success, verification_context
    
    def post(self, shared: Dict[str, Any], prep_res: Any, exec_res: Tuple[Dict[str, Any], bool, str]) -> str:
        result, success, verification_context = exec_res
        
        # Add to verification history
        verification_history = shared.get("verification_history", [])
        verification_history.append({
            "module": "ObjectQuant Locator", 
            "task": f"Locate objects: {prep_res[2]}",  # objects from prep
            "result": result,
            "success": success,
            "verification_context": verification_context,
            "timestamp": datetime.now().isoformat()
        })
        shared["verification_history"] = verification_history
        
        if success:
            logger.info("ObjectQuantLocatorVerification: Successfully located objects")
            return "continue"
        else:
            logger.error(f"ObjectQuantLocatorVerification failed: {result}")
            return "synthesize"

#############################################
# VisionIQ Analyst Action Node
#############################################
class VisionIQAnalystVerification(Node):
    """
    Uses VisionIQ Analyst to verify general visual reasoning steps.
    """
    def prep(self, shared: Dict[str, Any]) -> Tuple[str, str, str, str]:
        image_path = shared.get("image_path", "")
        api_key = shared.get("gemini_api_key", "")
        reasoning_step = shared.get("reasoning_step", "")
        current_decision = shared.get("current_verification_decision", {})
        task = current_decision.get("module_task", "")
        
        if not api_key:
            raise ValueError("Gemini API key is required for VisionIQ Analyst")
        
        # Extract question from the task
        question = task
        if ":" in task:
            question = task.split(":", 1)[1].strip()
        
        return image_path, api_key, question, reasoning_step
    
    def exec(self, inputs: Tuple[str, str, str, str]) -> Tuple[str, bool, str]:
        image_path, api_key, question, reasoning_step = inputs
        result, success = analyze_visual_content(image_path, api_key, question)
        
        # Add verification context
        verification_context = f"Reasoning step to verify: {reasoning_step}\n\nVisual analysis result: {result}"
        
        return result, success, verification_context
    
    def post(self, shared: Dict[str, Any], prep_res: Any, exec_res: Tuple[str, bool, str]) -> str:
        result, success, verification_context = exec_res
        
        # Add to verification history
        verification_history = shared.get("verification_history", [])
        verification_history.append({
            "module": "VisionIQ Analyst",
            "task": prep_res[2],  # question from prep
            "result": result,
            "success": success,
            "verification_context": verification_context,
            "timestamp": datetime.now().isoformat()
        })
        shared["verification_history"] = verification_history
        
        if success:
            logger.info("VisionIQAnalystVerification: Successfully analyzed visual content")
            return "continue"
        else:
            logger.error(f"VisionIQAnalystVerification failed: {result}")
            return "synthesize"

#############################################
# ChartSense Expert Action Node
#############################################
class ChartSenseExpertVerification(Node):
    """
    Uses ChartSense Expert to verify chart-related reasoning steps.
    """
    def prep(self, shared: Dict[str, Any]) -> Tuple[str, str, str, str]:
        image_path = shared.get("image_path", "")
        api_key = shared.get("gemini_api_key", "")
        reasoning_step = shared.get("reasoning_step", "")
        current_decision = shared.get("current_verification_decision", {})
        task = current_decision.get("module_task", "")
        
        if not api_key:
            raise ValueError("Gemini API key is required for ChartSense Expert")
        
        # Extract chart aspect from the task
        chart_aspect = task
        if ":" in task:
            chart_aspect = task.split(":", 1)[1].strip()
        
        return image_path, api_key, chart_aspect, reasoning_step
    
    def exec(self, inputs: Tuple[str, str, str, str]) -> Tuple[Dict[str, Any], bool, str]:
        image_path, api_key, chart_aspect, reasoning_step = inputs
        result, success = analyze_chart(image_path, api_key, chart_aspect)
        
        # Add verification context
        verification_context = f"Reasoning step to verify: {reasoning_step}\n\nChart analysis result: {result}"
        
        return result, success, verification_context
    
    def post(self, shared: Dict[str, Any], prep_res: Any, exec_res: Tuple[Dict[str, Any], bool, str]) -> str:
        result, success, verification_context = exec_res
        
        # Add to verification history
        verification_history = shared.get("verification_history", [])
        verification_history.append({
            "module": "ChartSense Expert",
            "task": prep_res[2],  # chart_aspect from prep
            "result": result,
            "success": success,
            "verification_context": verification_context,
            "timestamp": datetime.now().isoformat()
        })
        shared["verification_history"] = verification_history
        
        if success:
            logger.info("ChartSenseExpertVerification: Successfully analyzed chart")
            return "continue"
        else:
            logger.error(f"ChartSenseExpertVerification failed: {result}")
            return "synthesize"

#############################################
# Verification Synthesis Node
#############################################
class VerificationSynthesisNode(Node):
    """
    Synthesizes results from verification modules to make a final judgment on the validity of a reasoning step.
    """
    def prep(self, shared: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
        reasoning_step = shared.get("reasoning_step", "")
        verification_history = shared.get("verification_history", [])
        return reasoning_step, verification_history
    
    def exec(self, inputs: Tuple[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        reasoning_step, verification_history = inputs
        
        if not verification_history:
            return {
                "valid": False,
                "confidence": 0.0,
                "explanation": "No verification analysis was performed. Unable to judge the reasoning step."
            }
        
        # Format the verification results
        verification_summary = "Verification Analysis Results:\n\n"
        for i, step in enumerate(verification_history):
            verification_summary += f"{i+1}. {step['module']}\n"
            verification_summary += f"   Task: {step['task']}\n"
            verification_summary += f"   Success: {step['success']}\n"
            verification_summary += f"   Verification Context: {step.get('verification_context', 'N/A')}\n"
            if step['success']:
                verification_summary += f"   Result: {step['result']}\n"
            else:
                verification_summary += f"   Error: {step['result']}\n"
            verification_summary += "\n"
        
        # Create synthesis prompt
        prompt = f"""You are a verification agent evaluating the validity of an intermediate reasoning step in a multimodal task.

Reasoning Step to Verify: "{reasoning_step}"

{verification_summary}

Based on the verification results above, make a binary judgment: Is this reasoning step valid? 

Consider:
1. Is the reasoning step factually accurate according to the image?
2. Is the reasoning logical and consistent?
3. Does it contain any errors or misconceptions about what's in the image?
4. If the step makes claims about the image, are they supported by the evidence?

Provide your judgment in the following format:
```yaml
valid: true/false (boolean)
confidence: a value between 0.0 and 1.0
explanation: |
  Detailed explanation of your judgment, citing specific evidence from the verification results
```"""

        # Generate synthesis
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
            judgment = yaml.safe_load(yaml_content)
            
            # Validate fields
            assert "valid" in judgment, "Valid judgment is missing"
            assert isinstance(judgment["valid"], bool), "Valid must be a boolean"
            assert "confidence" in judgment, "Confidence is missing"
            assert "explanation" in judgment, "Explanation is missing"
            
            return judgment
        else:
            # Fallback if YAML parsing fails
            logger.warning("YAML parsing failed, using fallback judgment")
            return {
                "valid": False,
                "confidence": 0.5,
                "explanation": "Unable to parse judgment properly. The verification process completed but produced inconclusive results."
            }
    
    def post(self, shared: Dict[str, Any], prep_res: Any, exec_res: Dict[str, Any]) -> str:
        # Format the judgment for output
        valid_box = "Yes" if exec_res["valid"] else "No"
        final_judgment = f"${valid_box}$"
        
        # Store the final verification result
        shared["verification_result"] = {
            "valid": exec_res["valid"],
            "confidence": exec_res["confidence"],
            "explanation": exec_res["explanation"],
            "formatted_judgment": final_judgment
        }
        
        logger.info(f"VerificationSynthesisNode: Final judgment - Valid: {exec_res['valid']}, Confidence: {exec_res['confidence']}")
        return "complete"

#############################################
# Flow Creation
#############################################
def create_verification_flow() -> Flow:
    """
    Create the verification flow with all necessary nodes and transitions.
    """
    # Create nodes
    decision_node = VerificationDecisionNode()
    textintel_node = TextIntelExtractorVerification()
    objectquant_node = ObjectQuantLocatorVerification()
    visioniq_node = VisionIQAnalystVerification()
    chartsense_node = ChartSenseExpertVerification()
    synthesis_node = VerificationSynthesisNode()
    
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
def verify_reasoning_step(
    reasoning_step: str, 
    image_path: str, 
    gemini_api_key: str = None
) -> Dict[str, Any]:
    """
    Verify the validity of a reasoning step against an image.
    
    Args:
        reasoning_step: The intermediate reasoning step to verify
        image_path: Path to the image file
        gemini_api_key: Google Gemini API key (defaults to environment variable)
    
    Returns:
        Dictionary containing verification result and metadata
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
        "reasoning_step": reasoning_step,
        "image_path": image_path,
        "gemini_api_key": gemini_api_key,
        "verification_history": [],
        "verification_result": None
    }
    
    # Create and run the flow
    flow = create_verification_flow()
    flow.run(shared)
    
    # Format final judgment
    verification_result = shared.get("verification_result", {})
    final_judgment = verification_result.get("formatted_judgment", "$No$")
    
    # Return results
    return {
        "reasoning_step": reasoning_step,
        "image_path": image_path,
        "valid": verification_result.get("valid", False),
        "confidence": verification_result.get("confidence", 0.0),
        "explanation": verification_result.get("explanation", "Verification failed"),
        "final_judgment": final_judgment,
        "verification_history": shared.get("verification_history", []),
        "success": verification_result is not None
    }

def main():
    """
    Command-line interface for the visual reasoning verifier.
    """
    parser = argparse.ArgumentParser(description="Visual Reasoning Verifier - Evaluate the validity of reasoning steps")
    parser.add_argument("--step", "-s", type=str, required=True, 
                        help="Reasoning step to verify")
    parser.add_argument("--image", "-i", type=str, required=True,
                        help="Path to the image file to verify against")
    parser.add_argument("--api-key", "-k", type=str, 
                        help="Google Gemini API key (defaults to GEMINI_API_KEY env var)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Run verification
        result = verify_reasoning_step(
            reasoning_step=args.step,
            image_path=args.image,
            gemini_api_key=args.api_key
        )
        
        # Print results
        print("\n" + "="*60)
        print("VISUAL REASONING VERIFICATION RESULTS")
        print("="*60)
        print(f"\nReasoning Step: {result['reasoning_step']}")
        print(f"Image: {result['image_path']}")
        print(f"\nVerdict: {result['valid']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Explanation: {result['explanation']}")
        print(f"\nFinal Judgment: {result['final_judgment']}")
        
        if args.verbose and result['verification_history']:
            print(f"\n\nDetailed Verification History:")
            print("-" * 40)
            for i, step in enumerate(result['verification_history']):
                print(f"\nStep {i+1}: {step['module']}")
                print(f"Task: {step['task']}")
                print(f"Success: {step['success']}")
                if step['success']:
                    print(f"Result: {step['result']}")
                else:
                    print(f"Error: {step['result']}")
        
        print("\n" + "="*60)
        
    except Exception as e:
        logger.error(f"Error in verification: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
