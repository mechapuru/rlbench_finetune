import os
import sys
import re
import time
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@dataclass
class ActionSkeleton:
    action_name: str
    args: Tuple[str, ...]
    
    def __str__(self):
        return f"{self.action_name}({', '.join(self.args)})"
    
    def to_tuple(self):
        return (self.action_name,) + self.args


@dataclass
class PlanResult:
    success: bool
    skeleton: List[ActionSkeleton]
    raw_output: str
    inference_time: float
    error_message: Optional[str] = None


class LLMPlanner:
    """
    Pure text-based Large Language Model planner.
    Takes semantic state + goal and outputs action skeletons.
    """
    
    VALID_ACTIONS = {
        'pick': 1,
        'place': 2,
        'open-lid': 1,
        'open_lid': 1,
    }
    
    KNOWN_OBJECTS = {
        'mug_box', 'mug_inside_box', 'mug_table', 'mug_cupboard',
        'soup', 'mustard', 'spam', 'sugar', 'crackers', 'box_lid'
    }
    
    KNOWN_REGIONS = {
        'table', 'box-top', 'box-inside', 'placement_boundary',
        'cupboard_boundary', 'groceries_boundary'
    }
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                 device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.loaded = False
        
    def load_model(self) -> bool:
        if not HAS_TRANSFORMERS:
            print("ERROR: transformers not available.")
            return False
            
        print(f"Loading Text LLM: {self.model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype="auto",
                trust_remote_code=True
            )
            self.loaded = True
            print("LLM loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading LLM: {e}")
            return False
            
    def generate_plan(self, 
                      system_prompt: str,
                      user_prompt: str,
                      max_new_tokens: int = 512,
                      temperature: float = 0.1) -> PlanResult:
        if not self.loaded:
            return PlanResult(False, [], "", 0.0, "Model not loaded.")
            
        start_time = time.time()
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.tokenizer.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True
            )[0]
            
            inference_time = time.time() - start_time
            skeleton = self.parse_plan(output_text)
            
            return PlanResult(len(skeleton) > 0, skeleton, output_text, inference_time)
            
        except Exception as e:
            return PlanResult(False, [], "", time.time() - start_time, str(e))
            
    def parse_plan(self, text: str) -> List[ActionSkeleton]:
        actions = []
        pattern = r'(pick|place|open-lid|open_lid)\s*\(\s*([^)]+)\s*\)'
        matches = re.findall(pattern, text, re.IGNORECASE)
        
        for action_name, args_str in matches:
            action_name = action_name.lower().replace('_', '-')
            args = [a.strip() for a in args_str.split(',')]
            
            expected_args = self.VALID_ACTIONS.get(action_name, 0)
            if len(args) != expected_args:
                continue
                
            actions.append(ActionSkeleton(action_name, tuple(args)))
            if len(actions) >= 25: break
            
        return actions
