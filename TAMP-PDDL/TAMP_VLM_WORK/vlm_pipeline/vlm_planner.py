"""
VLM Planner (Module 2)
======================
Vision-Language Model planner using Qwen2-VL-7B-Instruct.
Takes visual context + state + goal and outputs action skeletons.
"""

import os
import sys
import re
import time
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# Try to import VLM dependencies
try:
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    HAS_QWEN_VL = True
except ImportError:
    HAS_QWEN_VL = False
    print("Warning: Qwen2-VL dependencies not installed.")
    print("Install with: pip install transformers qwen-vl-utils torch accelerate")


@dataclass
class ActionSkeleton:
    """Represents a single action in the plan skeleton."""
    action_name: str  # 'pick', 'place', 'open-lid'
    args: Tuple[str, ...]  # ('mug_box',) or ('mug_box', 'placement_boundary')
    
    def __str__(self):
        return f"{self.action_name}({', '.join(self.args)})"
    
    def to_tuple(self):
        """Convert to tuple format for compatibility."""
        return (self.action_name,) + self.args


@dataclass
class PlanResult:
    """Result from the VLM planner."""
    success: bool
    skeleton: List[ActionSkeleton]
    raw_output: str
    inference_time: float
    error_message: Optional[str] = None


class VLMPlanner:
    """
    Vision-Language Model planner using Qwen2-VL-7B-Instruct.
    """
    
    # Valid action patterns
    VALID_ACTIONS = {
        'pick': 1,      # pick(object)
        'place': 2,     # place(object, region)
        'open-lid': 1,  # open-lid(lid)
        'open_lid': 1,  # alternate format
    }
    
    # Known objects and regions for validation
    KNOWN_OBJECTS = {
        'mug_box', 'mug_inside_box', 'mug_table', 'mug_cupboard',
        'soup', 'mustard', 'spam', 'sugar', 'crackers', 'box_lid'
    }
    
    KNOWN_REGIONS = {
        'table', 'box-top', 'box-inside', 'placement_boundary',
        'cupboard_boundary', 'groceries_boundary'
    }
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
                 use_4bit: bool = False,  # 2B fits without quantization
                 device: str = "cuda"):
        """
        Initialize the VLM planner.
        
        Args:
            model_name: HuggingFace model name
            use_4bit: Whether to use 4-bit quantization
            device: Device to run on ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.use_4bit = use_4bit
        self.device = device
        
        self.model = None
        self.processor = None
        self.loaded = False
        
    def load_model(self) -> bool:
        """
        Load the Qwen2-VL model.
        
        Returns:
            True if successful, False otherwise
        """
        if not HAS_QWEN_VL:
            print("ERROR: Qwen2-VL dependencies not available.")
            return False
        
        print(f"Loading VLM: {self.model_name}")
        print(f"Using 4-bit quantization: {self.use_4bit}")
        
        # Clear cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        try:
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            if self.use_4bit:
                try:
                    from transformers import BitsAndBytesConfig
                    print("Configuring 4-bit quantization...")
                    
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True
                    )
                    
                    self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                        self.model_name,
                        quantization_config=bnb_config,
                        device_map="auto",
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
                except Exception as e:
                    print(f"4-bit quantization failed: {e}")
                    print("Cleaning up memory and falling back to float16...")
                    # Critical cleanup
                    self.model = None
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    self.use_4bit = False
            
            if not self.use_4bit:
                print("Loading with float16 precision...")
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            
            self.loaded = True
            print("VLM loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading VLM: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy image to base64 string."""
        from PIL import Image
        import base64
        import io
        
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
    
    def generate_plan(self, 
                      image: np.ndarray,
                      system_prompt: str,
                      user_prompt: str,
                      max_new_tokens: int = 1024,
                      temperature: float = 0.1) -> PlanResult:
        """
        Generate an action plan from visual context and prompts.
        
        Args:
            image: Composite image (numpy array)
            system_prompt: System prompt defining the task
            user_prompt: User prompt with state and goal
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            
        Returns:
            PlanResult with skeleton and metadata
        """
        if not self.loaded:
            return PlanResult(
                success=False,
                skeleton=[],
                raw_output="",
                inference_time=0,
                error_message="Model not loaded. Call load_model() first."
            )
        
        start_time = time.time()
        
        try:
            # Build messages with image
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f"data:image/png;base64,{self._image_to_base64(image)}"
                        },
                        {
                            "type": "text",
                            "text": user_prompt
                        }
                    ]
                }
            ]
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            print(f"[VLM Planner] Input text length: {len(text)}")
            print(f"[VLM Planner] Input text preview: {text[:200]}...")
            
            # Process vision info
            image_inputs, video_inputs = process_vision_info(messages)
            
            # Prepare inputs
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.device)
            print(f"[VLM Planner] Input tensors on device: {inputs.input_ids.device}, shape: {inputs.input_ids.shape}")
            
            # Generate with repetition penalty to avoid loops
            print("[VLM Planner] Starting generation...")
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=min(max_new_tokens, 512),  # Cap at 512 tokens
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    repetition_penalty=1.2,  # Penalize repeating tokens
                    no_repeat_ngram_size=3,  # Don't repeat 3-grams
                )
            print("[VLM Planner] Generation complete.")
            
            # Decode - remove input tokens
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            print(f"[VLM Planner] Raw output: {output_text}")

            
            inference_time = time.time() - start_time
            
            # Parse the output
            skeleton = self.parse_plan(output_text)
            
            return PlanResult(
                success=len(skeleton) > 0,
                skeleton=skeleton,
                raw_output=output_text,
                inference_time=inference_time
            )
            
        except Exception as e:
            inference_time = time.time() - start_time
            return PlanResult(
                success=False,
                skeleton=[],
                raw_output="",
                inference_time=inference_time,
                error_message=str(e)
            )
    
    def generate_plan_text_only(self,
                                system_prompt: str,
                                user_prompt: str,
                                max_new_tokens: int = 1024,
                                temperature: float = 0.1) -> PlanResult:
        """
        Generate plan without image (text-only mode for testing).
        Falls back to text-only LLM if VLM not available.
        """
        if not self.loaded:
            return PlanResult(
                success=False,
                skeleton=[],
                raw_output="",
                inference_time=0,
                error_message="Model not loaded."
            )
        
        start_time = time.time()
        
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=[text],
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True
            )[0]
            
            inference_time = time.time() - start_time
            skeleton = self.parse_plan(output_text)
            
            return PlanResult(
                success=len(skeleton) > 0,
                skeleton=skeleton,
                raw_output=output_text,
                inference_time=inference_time
            )
            
        except Exception as e:
            return PlanResult(
                success=False,
                skeleton=[],
                raw_output="",
                inference_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def parse_plan(self, text: str) -> List[ActionSkeleton]:
        """
        Parse VLM output text into action skeletons.
        
        Handles formats like:
        - "1. pick(mug_box)"
        - "pick(mug_box)"
        - "- pick(mug_box)"
        
        Args:
            text: Raw output text from VLM
            
        Returns:
            List of ActionSkeleton objects
        """
        actions = []
        seen_pick_place_pairs = set()  # Track (object, region) pairs to detect duplicates
        
        # Pattern to match action calls
        # Matches: pick(arg), place(arg1, arg2), open-lid(arg), open_lid(arg)
        pattern = r'(pick|place|open-lid|open_lid)\s*\(\s*([^)]+)\s*\)'
        
        matches = re.findall(pattern, text, re.IGNORECASE)
        
        for action_name, args_str in matches:
            # Normalize action name
            action_name = action_name.lower().replace('_', '-')
            
            # Parse arguments
            args = [a.strip() for a in args_str.split(',')]
            
            # Validate argument count
            expected_args = self.VALID_ACTIONS.get(action_name.replace('-', '_'), 
                                                    self.VALID_ACTIONS.get(action_name, 0))
            
            if len(args) != expected_args:
                print(f"Warning: {action_name} expects {expected_args} args, got {len(args)}")
                continue
            
            action = ActionSkeleton(
                action_name=action_name,
                args=tuple(args)
            )
            
            # De-duplicate: Check for repeated pick-place of same object to same region
            action_key = str(action)
            if action_name == 'place' and len(actions) >= 1:
                prev = actions[-1]
                if prev.action_name == 'pick':
                    pair_key = (prev.args[0], args[0] if len(args) == 1 else args[1])
                    if pair_key in seen_pick_place_pairs:
                        print(f"[Parser] Skipping duplicate pick-place: {pair_key}")
                        actions.pop()  # Remove the duplicate pick
                        continue
                    seen_pick_place_pairs.add(pair_key)
            
            actions.append(action)
            
            # Safety: Cap at 25 actions max
            if len(actions) >= 25:
                print(f"[Parser] Capping plan at 25 actions")
                break
        
        return actions
    
    def validate_plan(self, skeleton: List[ActionSkeleton]) -> Tuple[bool, List[str]]:
        """
        Validate the action skeleton against known constraints.
        
        Checks:
        1. Objects and regions are known
        2. Can't pick mug_inside_box before opening lid
        3. Can't open lid while mug_box is on top
        
        Args:
            skeleton: List of action skeletons
            
        Returns:
            (is_valid, list of error messages)
        """
        errors = []
        
        lid_opened = False
        mug_box_moved = False
        holding = None
        
        for i, action in enumerate(skeleton):
            # Check objects/regions are known
            for arg in action.args:
                if arg not in self.KNOWN_OBJECTS and arg not in self.KNOWN_REGIONS:
                    errors.append(f"Step {i+1}: Unknown object/region '{arg}'")
            
            if action.action_name == 'pick':
                obj = action.args[0]
                
                # Check if trying to pick while holding something
                if holding is not None:
                    errors.append(f"Step {i+1}: Cannot pick {obj}, already holding {holding}")
                
                # Check mug_inside_box constraint
                if obj == 'mug_inside_box' and not lid_opened:
                    errors.append(f"Step {i+1}: Cannot pick mug_inside_box - lid is not open yet")
                
                holding = obj
                if obj == 'mug_box':
                    mug_box_moved = True
                    
            elif action.action_name == 'place':
                obj = action.args[0]
                
                # Check if holding the object being placed
                if holding != obj:
                    errors.append(f"Step {i+1}: Cannot place {obj}, currently holding {holding}")
                
                holding = None
                
            elif action.action_name == 'open-lid':
                # Check if mug_box is still on top
                if not mug_box_moved:
                    errors.append(f"Step {i+1}: Cannot open lid - mug_box is still on top")
                
                # Check if hands are empty
                if holding is not None:
                    errors.append(f"Step {i+1}: Cannot open lid while holding {holding}")
                
                lid_opened = True
        
        return len(errors) == 0, errors


class MockVLMPlanner(VLMPlanner):
    """
    Mock VLM planner for testing without GPU/model.
    Returns predefined plans based on goal keywords.
    Supports replanning by detecting error context.
    """
    
    def __init__(self):
        super().__init__()
        self.loaded = True  # Pretend we're loaded
        self.replan_count = 0  # Track replans for demo
        
    def load_model(self) -> bool:
        print("MockVLMPlanner: Using mock model (no GPU required)")
        return True
    
    def generate_plan(self, 
                      image: np.ndarray,
                      system_prompt: str,
                      user_prompt: str,
                      **kwargs) -> PlanResult:
        """Return a mock plan based on goal analysis."""
        start_time = time.time()
        
        # Analyze the prompt to determine appropriate plan
        prompt_lower = user_prompt.lower()
        goal_lower = prompt_lower
        
        skeleton = []
        
        # =====================================================================
        # DETECT IF THIS IS A REPLAN (contains error context)
        # =====================================================================
        is_replan = 'replanning required' in prompt_lower or 'failed action' in prompt_lower
        
        if is_replan:
            self.replan_count += 1
            print(f"[MockVLM] Detected REPLAN request (replan #{self.replan_count})")
            
            # Check what error occurred and generate corrected plan
            if 'no pddl plan found' in prompt_lower or 'lid_closed' in prompt_lower or 'mug_inside_box' in prompt_lower:
                # The mug_inside_box failed because lid is closed
                # Generate correct sequence: move mug_box → open lid → pick mug_inside_box
                print("[MockVLM] Error was about mug_inside_box/lid - generating corrected plan")
                skeleton = [
                    ActionSkeleton('pick', ('mug_box',)),
                    ActionSkeleton('place', ('mug_box', 'placement_boundary')),
                    ActionSkeleton('open-lid', ('box_lid',)),
                    ActionSkeleton('pick', ('mug_inside_box',)),
                    ActionSkeleton('place', ('mug_inside_box', 'placement_boundary')),
                    ActionSkeleton('pick', ('mug_cupboard',)),
                    ActionSkeleton('place', ('mug_cupboard', 'placement_boundary')),
                    ActionSkeleton('pick', ('soup',)),
                    ActionSkeleton('place', ('soup', 'cupboard_boundary')),
                    ActionSkeleton('pick', ('mustard',)),
                    ActionSkeleton('place', ('mustard', 'cupboard_boundary')),
                    ActionSkeleton('pick', ('spam',)),
                    ActionSkeleton('place', ('spam', 'cupboard_boundary')),
                    ActionSkeleton('pick', ('sugar',)),
                    ActionSkeleton('place', ('sugar', 'cupboard_boundary')),
                    ActionSkeleton('pick', ('crackers',)),
                    ActionSkeleton('place', ('crackers', 'cupboard_boundary')),
                ]
            elif 'object_blocked' in prompt_lower or 'blocked by mug_box' in prompt_lower:
                # Lid was blocked by mug_box
                print("[MockVLM] Error was about blocked lid - move mug_box first")
                skeleton = [
                    ActionSkeleton('pick', ('mug_box',)),
                    ActionSkeleton('place', ('mug_box', 'placement_boundary')),
                    ActionSkeleton('open-lid', ('box_lid',)),
                ]
            else:
                # Generic replan - give full correct sequence
                print("[MockVLM] Generic replan - giving full correct sequence")
                skeleton = [
                    ActionSkeleton('pick', ('mug_box',)),
                    ActionSkeleton('place', ('mug_box', 'placement_boundary')),
                    ActionSkeleton('open-lid', ('box_lid',)),
                    ActionSkeleton('pick', ('mug_inside_box',)),
                    ActionSkeleton('place', ('mug_inside_box', 'placement_boundary')),
                    ActionSkeleton('pick', ('soup',)),
                    ActionSkeleton('place', ('soup', 'cupboard_boundary')),
                ]
        
        # =====================================================================
        # INITIAL PLAN - Deliberately make a mistake to demo replanning
        # =====================================================================
        elif 'groceries' in goal_lower and 'mugs' in goal_lower:
            # Full task - deliberately try mug_inside_box first (will fail)
            print("[MockVLM] Initial plan: Deliberately trying mug_inside_box first (will fail)")
            skeleton = [
                ActionSkeleton('pick', ('mug_inside_box',)),
                ActionSkeleton('place', ('mug_inside_box', 'placement_boundary')),
            ]
        
        # =====================================================================
        # SPECIFIC GOAL PATTERNS
        # =====================================================================
        
        # Test 1: Open lid first (WRONG - should fail with object_blocked)
        elif 'open' in goal_lower and 'lid' in goal_lower and 'mug' not in goal_lower:
            skeleton = [
                ActionSkeleton('open-lid', ('box_lid',)),
            ]
        
        # Move mug_box to table (correct)
        elif 'mug_box' in goal_lower or ('mug' in goal_lower and 'box' in goal_lower and 'table' in goal_lower):
            skeleton = [
                ActionSkeleton('pick', ('mug_box',)),
                ActionSkeleton('place', ('mug_box', 'placement_boundary')),
            ]
        
        # Mugs on table only
        elif 'mug' in goal_lower and 'table' in goal_lower:
            skeleton = [
                ActionSkeleton('pick', ('mug_box',)),
                ActionSkeleton('place', ('mug_box', 'placement_boundary')),
                ActionSkeleton('open-lid', ('box_lid',)),
                ActionSkeleton('pick', ('mug_inside_box',)),
                ActionSkeleton('place', ('mug_inside_box', 'placement_boundary')),
                ActionSkeleton('pick', ('mug_cupboard',)),
                ActionSkeleton('place', ('mug_cupboard', 'placement_boundary')),
            ]
        
        # Soup to cupboard
        elif 'soup' in goal_lower and 'cupboard' in goal_lower:
            skeleton = [
                ActionSkeleton('pick', ('soup',)),
                ActionSkeleton('place', ('soup', 'cupboard_boundary')),
            ]
        
        # Default - give complete task
        if not skeleton:
            skeleton = [
                ActionSkeleton('pick', ('mug_box',)),
                ActionSkeleton('place', ('mug_box', 'placement_boundary')),
                ActionSkeleton('open-lid', ('box_lid',)),
                ActionSkeleton('pick', ('mug_inside_box',)),
                ActionSkeleton('place', ('mug_inside_box', 'placement_boundary')),
                ActionSkeleton('pick', ('soup',)),
                ActionSkeleton('place', ('soup', 'cupboard_boundary')),
            ]
        
        raw_output = "\n".join([f"{i+1}. {a}" for i, a in enumerate(skeleton)])
        
        print(f"[MockVLM] Generated plan ({len(skeleton)} actions)")
        print(f"[MockVLM] Plan: {[str(a) for a in skeleton]}")
        
        return PlanResult(
            success=True,
            skeleton=skeleton,
            raw_output=raw_output,
            inference_time=time.time() - start_time
        )
        
        return PlanResult(
            success=True,
            skeleton=skeleton,
            raw_output=raw_output,
            inference_time=time.time() - start_time
        )


# ============================================================================
# TESTING
# ============================================================================

def test_parser():
    """Test the plan parser."""
    print("Testing plan parser...")
    
    planner = MockVLMPlanner()
    
    test_texts = [
        """1. pick(mug_box)
2. place(mug_box, placement_boundary)
3. open-lid(box_lid)
4. pick(mug_inside_box)
5. place(mug_inside_box, placement_boundary)""",
        
        """- pick(mug_box)
- place(mug_box, placement_boundary)
- open_lid(box_lid)""",
        
        """The robot should:
pick(soup)
then place(soup, cupboard_boundary)"""
    ]
    
    for text in test_texts:
        print(f"\nInput:\n{text}\n")
        actions = planner.parse_plan(text)
        print("Parsed actions:")
        for a in actions:
            print(f"  {a}")


def test_validation():
    """Test plan validation."""
    print("\nTesting plan validation...")
    
    planner = MockVLMPlanner()
    
    # Valid plan
    valid_plan = [
        ActionSkeleton('pick', ('mug_box',)),
        ActionSkeleton('place', ('mug_box', 'placement_boundary')),
        ActionSkeleton('open-lid', ('box_lid',)),
        ActionSkeleton('pick', ('mug_inside_box',)),
        ActionSkeleton('place', ('mug_inside_box', 'placement_boundary')),
    ]
    
    is_valid, errors = planner.validate_plan(valid_plan)
    print(f"\nValid plan test: {'PASS' if is_valid else 'FAIL'}")
    if errors:
        print(f"  Errors: {errors}")
    
    # Invalid plan - pick mug_inside_box before opening lid
    invalid_plan = [
        ActionSkeleton('pick', ('mug_inside_box',)),
        ActionSkeleton('place', ('mug_inside_box', 'placement_boundary')),
    ]
    
    is_valid, errors = planner.validate_plan(invalid_plan)
    print(f"\nInvalid plan test (should fail): {'PASS' if not is_valid else 'FAIL'}")
    if errors:
        print(f"  Errors: {errors}")
    
    # Invalid plan - open lid while mug_box on top
    invalid_plan2 = [
        ActionSkeleton('open-lid', ('box_lid',)),
    ]
    
    is_valid, errors = planner.validate_plan(invalid_plan2)
    print(f"\nInvalid plan test 2 (should fail): {'PASS' if not is_valid else 'FAIL'}")
    if errors:
        print(f"  Errors: {errors}")


def test_mock_planner():
    """Test mock planner generation."""
    print("\nTesting mock planner...")
    
    planner = MockVLMPlanner()
    
    # Dummy image
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    result = planner.generate_plan(
        image=image,
        system_prompt="You are a robot planner.",
        user_prompt="Move mug_box to placement_boundary, then open lid and move mug_inside_box to placement_boundary."
    )
    
    print(f"\nSuccess: {result.success}")
    print(f"Inference time: {result.inference_time:.3f}s")
    print(f"Raw output:\n{result.raw_output}")
    print(f"\nSkeleton:")
    for a in result.skeleton:
        print(f"  {a}")


if __name__ == "__main__":
    test_parser()
    test_validation()
    test_mock_planner()
