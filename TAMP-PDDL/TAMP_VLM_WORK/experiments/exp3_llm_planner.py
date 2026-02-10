"""
Experiment 3: LLM-based Task Planner
====================================
Uses a local LLM (Qwen2.5-7B-Instruct) to generate action sequences
for the kitchen manipulation task.
"""

import os
import sys
import json
import time
import argparse
import random
import numpy as np
from pathlib import Path

# --- CONFIGURATION ---
EXPERIMENT_NAME = "exp3_llm_planner"

# Ground truth action sequence
GROUND_TRUTH_ACTIONS = [
    ('pick', 'mug_box'),
    ('place', 'mug_box', 'placement_boundary'),
    ('open-lid', 'box_lid'),
    ('pick', 'mug_inside_box'),
    ('place', 'mug_inside_box', 'placement_boundary'),
    ('pick', 'soup'),
    ('place', 'soup', 'cupboard_boundary'),
]

GROUND_TRUTH_SIMPLE = ['pick', 'place', 'open-lid', 'pick', 'place', 'pick', 'place']

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results", EXPERIMENT_NAME)
os.makedirs(RESULTS_DIR, exist_ok=True)


# =============================================================================
# SCENE EXTRACTION
# =============================================================================

class SceneExtractor:
    """Extract scene information from RLBench environment."""
    
    def __init__(self, env=None):
        self.env = env
    
    def get_scene_info(self):
        """
        Extract current scene state.
        For now, we use a fixed scene description since the task is deterministic.
        In a real system, this would query the actual environment state.
        """
        # Fixed scene for our kitchen task
        scene = {
            "objects": [
                {
                    "name": "mug_box",
                    "type": "mug",
                    "location": "on top of closed box",
                    "pickable": True,
                    "blocked_by": None
                },
                {
                    "name": "mug_inside_box", 
                    "type": "mug",
                    "location": "inside the box",
                    "pickable": False,  # blocked until lid opens
                    "blocked_by": "box_lid"
                },
                {
                    "name": "soup",
                    "type": "can",
                    "location": "on table",
                    "pickable": True,
                    "blocked_by": None
                },
                {
                    "name": "box_lid",
                    "type": "lid",
                    "location": "on box",
                    "state": "closed",
                    "openable": True
                }
            ],
            "regions": [
                {"name": "placement_boundary", "description": "target area for mugs"},
                {"name": "cupboard_boundary", "description": "inside the cupboard"},
                {"name": "box_top", "description": "top of the box"}
            ],
            "robot": {
                "gripper_state": "empty",
                "location": "home"
            },
            "constraints": [
                "The robot can only hold one object at a time",
                "Must place current object before picking another",
                "Cannot pick mug_inside_box while box_lid is closed",
                "Must open box_lid first to access mug_inside_box"
            ],
            "goal": {
                "mug_box": "placement_boundary",
                "mug_inside_box": "placement_boundary",
                "soup": "cupboard_boundary",
                "box_lid": "opened"
            }
        }
        return scene
    
    def get_scene_from_env(self):
        """Extract actual scene state from RLBench environment."""
        if self.env is None:
            return self.get_scene_info()
        
        env = self.env
        scene = {"objects": [], "regions": [], "robot": {}, "constraints": [], "goal": {}}
        
        # Get object positions
        try:
            mug_box = env.get_object('mug_box')
            if mug_box:
                pos = mug_box.get_position()
                scene["objects"].append({
                    "name": "mug_box",
                    "type": "mug", 
                    "position": [round(p, 3) for p in pos],
                    "location": "on top of box",
                    "pickable": True,
                    "blocked_by": None
                })
        except:
            pass
            
        try:
            mug_inside = env.get_object('mug_inside_box')
            if mug_inside:
                pos = mug_inside.get_position()
                scene["objects"].append({
                    "name": "mug_inside_box",
                    "type": "mug",
                    "position": [round(p, 3) for p in pos],
                    "location": "inside box",
                    "pickable": False,
                    "blocked_by": "box_lid"
                })
        except:
            pass
            
        try:
            soup = env.get_object('soup')
            if soup:
                pos = soup.get_position()
                scene["objects"].append({
                    "name": "soup",
                    "type": "can",
                    "position": [round(p, 3) for p in pos],
                    "location": "on table",
                    "pickable": True,
                    "blocked_by": None
                })
        except:
            pass
            
        try:
            lid = env.get_object('box_lid')
            if lid:
                pos = lid.get_position()
                # Check if lid is open by comparing position
                box = env.get_object('box_base')
                if box:
                    box_pos = box.get_position()
                    x_offset = abs(pos[0] - box_pos[0])
                    is_open = x_offset > 0.10
                else:
                    is_open = False
                scene["objects"].append({
                    "name": "box_lid",
                    "type": "lid",
                    "position": [round(p, 3) for p in pos],
                    "state": "open" if is_open else "closed",
                    "openable": True
                })
        except:
            pass
        
        # Regions
        scene["regions"] = [
            {"name": "placement_boundary", "description": "target area for mugs"},
            {"name": "cupboard_boundary", "description": "inside the cupboard"},
            {"name": "box_top", "description": "top of the box"}
        ]
        
        # Robot state
        try:
            scene["robot"] = {
                "gripper_state": "empty",  # Simplified
                "joint_positions": [round(j, 3) for j in env.get_robot_conf()]
            }
        except:
            scene["robot"] = {"gripper_state": "empty"}
        
        # Constraints (fixed)
        scene["constraints"] = [
            "The robot can only hold one object at a time",
            "Must place current object before picking another", 
            "Cannot pick mug_inside_box while box_lid is closed",
            "Must open box_lid first to access mug_inside_box"
        ]
        
        # Goal (fixed)
        scene["goal"] = {
            "mug_box": "placement_boundary",
            "mug_inside_box": "placement_boundary",
            "soup": "cupboard_boundary",
            "box_lid": "opened"
        }
        
        return scene


# =============================================================================
# PROMPT BUILDER
# =============================================================================

class PromptBuilder:
    """Build prompts for the LLM planner."""
    
    @staticmethod
    def build_planning_prompt(scene: dict) -> str:
        """Build a prompt for action sequence planning."""
        
        prompt = """You are a robot task planner. Given the scene description and goal, output the correct sequence of actions.

## SCENE DESCRIPTION

### Objects:
"""
        # Add objects
        for obj in scene["objects"]:
            if obj["type"] == "lid":
                prompt += f"- {obj['name']}: {obj['type']}, state={obj.get('state', 'unknown')}\n"
            else:
                blocked = f", BLOCKED BY {obj['blocked_by']}" if obj.get('blocked_by') else ""
                prompt += f"- {obj['name']}: {obj['type']}, location={obj['location']}{blocked}\n"
        
        prompt += "\n### Target Regions:\n"
        for region in scene["regions"]:
            prompt += f"- {region['name']}: {region['description']}\n"
        
        prompt += "\n### Robot State:\n"
        prompt += f"- Gripper: {scene['robot']['gripper_state']}\n"
        
        prompt += "\n## AVAILABLE ACTIONS\n"
        prompt += """- pick(object): Pick up an object. Cannot pick if object is blocked.
- place(object, region): Place the held object in a region.
- open-lid(lid): Open a lid to unblock objects inside.

## CONSTRAINTS
"""
        for c in scene["constraints"]:
            prompt += f"- {c}\n"
        
        prompt += "\n## GOAL\n"
        for obj, target in scene["goal"].items():
            prompt += f"- {obj} should be in/at {target}\n"
        
        prompt += """
## TASK
Output the sequence of actions to achieve the goal. Consider the constraints carefully.

IMPORTANT:
- mug_inside_box is BLOCKED by box_lid until box_lid is opened
- You must open-lid(box_lid) BEFORE you can pick(mug_inside_box)

## OUTPUT FORMAT
Output ONLY the action sequence, one action per line, in this exact format:
pick(object_name)
place(object_name, region_name)
open-lid(lid_name)

## ACTION SEQUENCE:
"""
        return prompt
    
    @staticmethod
    def build_simple_prompt(scene: dict) -> str:
        """Build a few-shot prompt with example."""
        
        prompt = """You are a robot planner. Follow the exact same format as the example.

=== EXAMPLE ===
Problem: Move block_A and block_B to goal_zone. block_A is on top of block_B.
Rules: Cannot pick block_B while block_A is on top of it.
Solution:
1. pick(block_A)
2. place(block_A, goal_zone)
3. pick(block_B)
4. place(block_B, goal_zone)

=== YOUR TASK ===
Problem: Move mug_box, mug_inside_box, and soup to their goals. mug_box is on top of box_lid. box_lid is closed, blocking mug_inside_box inside.
Rules: Cannot open box_lid while mug_box is on top. Cannot pick mug_inside_box while box_lid is closed.
Goals: mug_box to placement_boundary, mug_inside_box to placement_boundary, soup to cupboard_boundary.
Solution:
1."""
        return prompt


# =============================================================================
# LLM INTERFACE
# =============================================================================

class LLMPlanner:
    """Interface to local LLM for planning."""
    
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", use_4bit=True):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.use_4bit = use_4bit
        self.device = "cuda"
        
    def load_model(self):
        """Load the model with optional 4-bit quantization."""
        print(f"Loading model: {self.model_name}")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            if self.use_4bit:
                try:
                    from transformers import BitsAndBytesConfig
                    print("Using 4-bit quantization...")
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        quantization_config=bnb_config,
                        device_map="auto",
                        trust_remote_code=True
                    )
                except Exception as e:
                    print(f"4-bit quantization failed: {e}")
                    print("Falling back to float16...")
                    self.use_4bit = False
            
            if not self.use_4bit:
                print("Using float16 precision...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
            
            print("Model loaded successfully!")
            return True
            
        except ImportError as e:
            print(f"Error: Required packages not installed: {e}")
            print("Install with: pip install transformers torch bitsandbytes accelerate")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def generate_plan(self, prompt: str, max_new_tokens=150, temperature=0.1) -> tuple:
        """
        Generate action plan from prompt using proper chat template.
        Returns: (generated_text, inference_time)
        """
        if self.model is None:
            return None, 0
        
        try:
            import torch
            
            # Use proper chat template for Qwen/instruction models
            messages = [
                {"role": "system", "content": "You are a robot planner. Output ONLY a numbered action list. No explanations."},
                {"role": "user", "content": prompt}
            ]
            
            # Apply chat template (handles special tokens properly)
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            
            # Generate with controlled parameters
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    top_p=0.95,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            inference_time = time.time() - start_time
            
            # Decode only the NEW tokens (not the input)
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            # Clean: stop at garbage indicators
            stop_strings = ["\n\n\n", "Human:", "User:", "###", "Note:", "Explanation:"]
            for stop in stop_strings:
                if stop in response:
                    response = response.split(stop)[0].strip()
            
            return response, inference_time
            
        except Exception as e:
            print(f"Error during generation: {e}")
            return None, 0
    
    def unload_model(self):
        """Unload model to free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        try:
            import torch
            torch.cuda.empty_cache()
        except:
            pass


# =============================================================================
# PLAN PARSER & VALIDATOR
# =============================================================================

class PlanParser:
    """Parse and validate LLM-generated plans."""
    
    VALID_OBJECTS = {'mug_box', 'mug_inside_box', 'soup', 'box_lid'}
    VALID_REGIONS = {'placement_boundary', 'cupboard_boundary', 'box_top'}
    
    @staticmethod
    def parse_plan(text: str) -> list:
        """Parse action sequence from LLM output."""
        actions = []
        lines = text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip lines that look like comments or explanations
            if line.startswith('#') or line.startswith('//') or ':' in line and '(' not in line:
                continue
            
            # Try to parse action
            action = PlanParser._parse_action(line)
            if action:
                actions.append(action)
        
        return actions
    
    @staticmethod
    def _parse_action(line: str) -> tuple:
        """Parse a single action line."""
        line = line.strip().lower()
        
        # Remove leading numbers/bullets
        import re
        line = re.sub(r'^[\d\.\-\*\)]+\s*', '', line)
        
        # Parse pick(object)
        match = re.match(r'pick\s*\(\s*(\w+)\s*\)', line)
        if match:
            return ('pick', match.group(1))
        
        # Parse place(object, region)
        match = re.match(r'place\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)', line)
        if match:
            return ('place', match.group(1), match.group(2))
        
        # Parse open-lid(lid) or open_lid(lid)
        match = re.match(r'open[-_]?lid\s*\(\s*(\w+)\s*\)', line)
        if match:
            return ('open-lid', match.group(1))
        
        return None
    
    @staticmethod
    def validate_plan(actions: list, scene: dict) -> dict:
        """
        Validate if plan is logically correct.
        Returns validation result with details.
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'achieves_goal': False
        }
        
        if not actions:
            result['valid'] = False
            result['errors'].append("Empty plan")
            return result
        
        # Simulate execution
        holding = None
        lid_opened = False
        object_locations = {
            'mug_box': 'box_top',
            'mug_inside_box': 'box_inside',
            'soup': 'table'
        }
        
        for i, action in enumerate(actions):
            action_type = action[0]
            
            if action_type == 'pick':
                obj = action[1]
                
                # Check if already holding something
                if holding is not None:
                    result['errors'].append(f"Step {i+1}: Cannot pick {obj}, already holding {holding}")
                    result['valid'] = False
                    continue
                
                # Check if object is blocked
                if obj == 'mug_inside_box' and not lid_opened:
                    result['errors'].append(f"Step {i+1}: Cannot pick mug_inside_box, lid is closed")
                    result['valid'] = False
                    continue
                
                holding = obj
                
            elif action_type == 'place':
                obj = action[1]
                region = action[2]
                
                # Check if holding the right object
                if holding != obj:
                    result['errors'].append(f"Step {i+1}: Cannot place {obj}, not holding it (holding: {holding})")
                    result['valid'] = False
                    continue
                
                object_locations[obj] = region
                holding = None
                
            elif action_type == 'open-lid':
                lid = action[1]
                if holding is not None:
                    result['warnings'].append(f"Step {i+1}: Opening lid while holding {holding}")
                lid_opened = True
        
        # Check goal achievement
        goal_achieved = (
            object_locations.get('mug_box') == 'placement_boundary' and
            object_locations.get('mug_inside_box') == 'placement_boundary' and
            object_locations.get('soup') == 'cupboard_boundary' and
            lid_opened
        )
        result['achieves_goal'] = goal_achieved
        
        if not goal_achieved:
            result['warnings'].append("Plan may not achieve all goals")
        
        return result
    
    @staticmethod
    def compare_to_ground_truth(actions: list) -> dict:
        """Compare plan to ground truth."""
        # Extract just action names for comparison
        action_names = [a[0] for a in actions]
        gt_names = GROUND_TRUTH_SIMPLE
        
        exact_match = action_names == gt_names
        
        # Check if key ordering is correct (open-lid before pick mug_inside_box)
        correct_ordering = True
        open_lid_idx = None
        pick_inside_idx = None
        
        for i, action in enumerate(actions):
            if action[0] == 'open-lid':
                open_lid_idx = i
            if action[0] == 'pick' and len(action) > 1 and action[1] == 'mug_inside_box':
                pick_inside_idx = i
        
        if open_lid_idx is not None and pick_inside_idx is not None:
            correct_ordering = open_lid_idx < pick_inside_idx
        elif pick_inside_idx is not None and open_lid_idx is None:
            correct_ordering = False  # Tried to pick without opening
        
        return {
            'exact_match': exact_match,
            'action_count_match': len(actions) == len(gt_names),
            'correct_ordering': correct_ordering,
            'generated_actions': action_names,
            'ground_truth': gt_names
        }


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

class ExperimentRunner:
    """Run the LLM planning experiment."""
    
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", use_4bit=True):
        self.llm = LLMPlanner(model_name, use_4bit)
        self.scene_extractor = SceneExtractor()
        self.results = []
        
    def run_trial(self, trial_num: int, use_simple_prompt=True) -> dict:
        """Run a single trial."""
        print(f"\n--- Trial {trial_num} ---")
        
        # Get scene
        scene = self.scene_extractor.get_scene_info()
        
        # Build prompt
        if use_simple_prompt:
            prompt = PromptBuilder.build_simple_prompt(scene)
        else:
            prompt = PromptBuilder.build_planning_prompt(scene)
        
        # Generate plan
        response, inference_time = self.llm.generate_plan(prompt)
        
        if response is None:
            return {
                'trial': trial_num,
                'status': 'ERROR',
                'error': 'Generation failed',
                'inference_time': 0
            }
        
        print(f"LLM Response:\n{response}")
        
        # Parse plan
        actions = PlanParser.parse_plan(response)
        print(f"Parsed actions: {actions}")
        
        # Validate
        validation = PlanParser.validate_plan(actions, scene)
        comparison = PlanParser.compare_to_ground_truth(actions)
        
        result = {
            'trial': trial_num,
            'status': 'SUCCESS' if validation['valid'] and validation['achieves_goal'] else 'FAILURE',
            'prompt_type': 'simple' if use_simple_prompt else 'detailed',
            'raw_response': response,
            'parsed_actions': [list(a) for a in actions],
            'action_count': len(actions),
            'inference_time': inference_time,
            'validation': validation,
            'ground_truth_comparison': comparison,
            'valid_plan': validation['valid'],
            'achieves_goal': validation['achieves_goal'],
            'correct_ordering': comparison['correct_ordering'],
            'exact_match': comparison['exact_match']
        }
        
        self.results.append(result)
        return result
    
    def run_experiment(self, num_trials=10, use_simple_prompt=True):
        """Run the full experiment."""
        print(f"\n{'='*60}")
        print(f"EXPERIMENT 3: LLM Task Planner")
        print(f"Model: {self.llm.model_name}")
        print(f"Trials: {num_trials}")
        print(f"{'='*60}")
        
        # Load model
        if not self.llm.load_model():
            print("Failed to load model. Exiting.")
            return
        
        # Run trials
        for i in range(num_trials):
            self.run_trial(i + 1, use_simple_prompt)
        
        # Generate summary
        self.save_results()
        self.print_summary()
        
        # Cleanup
        self.llm.unload_model()
    
    def save_results(self):
        """Save results to files."""
        # Save individual trials
        for result in self.results:
            filepath = os.path.join(RESULTS_DIR, f"trial_{result['trial']:03d}.json")
            with open(filepath, 'w') as f:
                json.dump(result, f, indent=2)
        
        # Save summary
        summary = self.generate_summary()
        filepath = os.path.join(RESULTS_DIR, "summary.json")
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nResults saved to: {RESULTS_DIR}")
    
    def generate_summary(self) -> dict:
        """Generate experiment summary."""
        if not self.results:
            return {}
        
        valid_plans = sum(1 for r in self.results if r['valid_plan'])
        achieves_goal = sum(1 for r in self.results if r['achieves_goal'])
        correct_ordering = sum(1 for r in self.results if r['correct_ordering'])
        exact_matches = sum(1 for r in self.results if r['exact_match'])
        
        inference_times = [r['inference_time'] for r in self.results if r['inference_time'] > 0]
        
        return {
            'experiment': EXPERIMENT_NAME,
            'model': self.llm.model_name,
            'total_trials': len(self.results),
            'valid_plans': valid_plans,
            'valid_plan_rate': valid_plans / len(self.results),
            'achieves_goal': achieves_goal,
            'goal_achievement_rate': achieves_goal / len(self.results),
            'correct_ordering': correct_ordering,
            'correct_ordering_rate': correct_ordering / len(self.results),
            'exact_matches': exact_matches,
            'exact_match_rate': exact_matches / len(self.results),
            'avg_inference_time': np.mean(inference_times) if inference_times else 0,
            'std_inference_time': np.std(inference_times) if inference_times else 0,
            'ground_truth': GROUND_TRUTH_SIMPLE
        }
    
    def print_summary(self):
        """Print experiment summary."""
        summary = self.generate_summary()
        
        print(f"\n{'='*60}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        print(f"Model: {summary['model']}")
        print(f"Total Trials: {summary['total_trials']}")
        print(f"")
        print(f"Valid Plans:      {summary['valid_plans']}/{summary['total_trials']} ({summary['valid_plan_rate']*100:.1f}%)")
        print(f"Achieves Goal:    {summary['achieves_goal']}/{summary['total_trials']} ({summary['goal_achievement_rate']*100:.1f}%)")
        print(f"Correct Ordering: {summary['correct_ordering']}/{summary['total_trials']} ({summary['correct_ordering_rate']*100:.1f}%)")
        print(f"Exact Match:      {summary['exact_matches']}/{summary['total_trials']} ({summary['exact_match_rate']*100:.1f}%)")
        print(f"")
        print(f"Avg Inference Time: {summary['avg_inference_time']:.2f}s ± {summary['std_inference_time']:.2f}s")
        print(f"{'='*60}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Experiment 3: LLM Task Planner")
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                       help='HuggingFace model name')
    parser.add_argument('--trials', type=int, default=10,
                       help='Number of trials to run')
    parser.add_argument('--no-4bit', action='store_true',
                       help='Disable 4-bit quantization')
    parser.add_argument('--detailed-prompt', action='store_true',
                       help='Use detailed prompt instead of simple')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    args = parser.parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Run experiment
    runner = ExperimentRunner(
        model_name=args.model,
        use_4bit=not args.no_4bit
    )
    runner.run_experiment(
        num_trials=args.trials,
        use_simple_prompt=not args.detailed_prompt
    )


if __name__ == "__main__":
    main()
