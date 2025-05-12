"""
Veritas AI Scientist - Prompt Optimization Module

This module provides optimized prompts and strategies for the Veritas AI Scientist system.
It enhances the quality of generated responses across different research phases by
adapting prompts specifically for the Mistral model's capabilities.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Standard system prompts
STANDARD_PROMPTS = {
    "idea_generation": "You are an AI scientist working on coming up with novel ideas...",
    "experiments": "You are an AI scientist who needs to design and run experiments...",
    "writeup": "You are an AI scientist who needs to write up experimental results..."
}

# Enhanced prompts optimized for Mistral
OPTIMIZED_PROMPTS = {
    "idea_generation": """You are a creative AI scientist. Generate promising research ideas.
    
Focus on ideas that are:
1. Novel but not overly complicated
2. Testable with clear metrics
3. Specific, not abstract
4. Focused on solving a concrete problem

Be precise and detailed, explaining both the idea and experiment.
""",
    
    "experiments": """You are a practical AI scientist running experiments.

For each experiment:
1. Define clear metrics to measure success
2. Use step-by-step reasoning
3. Focus on practical implementation details
4. Provide simple but robust code
5. Document results thoroughly
""",
    
    "writeup": """You are a clear technical writer for scientific papers.

Your paper should:
1. Explain the key idea without jargon
2. Present methods precisely
3. Analyze results objectively
4. Use simple figures and tables
5. Be structured and organized
""",

    "evaluation": """You are a fair reviewer of scientific work.

In your evaluation:
1. Focus on the strengths first
2. Identify limitations clearly
3. Suggest specific improvements
4. Be constructive and helpful
5. Rate the work on novelty, validity, and clarity
"""
}

def get_optimized_prompt(prompt_type: str, original_prompt: str = None) -> str:
    """
    Get an optimized prompt based on the prompt type.
    
    Args:
        prompt_type: Type of prompt (idea_generation, experiments, etc.)
        original_prompt: The standard prompt to enhance
        
    Returns:
        Optimized prompt for better results
    """
    # Check if we have a specific optimization for this prompt type
    if prompt_type in OPTIMIZED_PROMPTS:
        optimized = OPTIMIZED_PROMPTS[prompt_type]
        
        # If original prompt provided, append key instructions
        if original_prompt:
            # Extract key instructions (usually after "Your task" or similar phrases)
            import re
            task_match = re.search(r"Your task is to (.*?)(\.|$)", original_prompt)
            if task_match:
                task_instruction = task_match.group(1)
                optimized += f"\n\nSpecific task: {task_instruction}."
        
        return optimized
    
    # If no optimization available, return original or default
    return original_prompt or "You are a helpful AI assistant."

def enhance_scientific_prompts():
    """
    Enhance the scientific research prompts with optimized versions.
    
    This function improves the quality of prompts used in different
    phases of the scientific research process.
    
    Returns:
        bool: True if enhancement was successful, False otherwise
    """
    try:
        # Import research modules with prompts
        from ai_scientist.generate_ideas import generate_ideas
        from ai_scientist.perform_experiments import perform_experiments
        from ai_scientist.perform_writeup import perform_writeup
        
        # Check if we can access system prompts in these modules
        if hasattr(generate_ideas, 'SYSTEM_PROMPT'):
            logger.info("Enhancing idea generation prompts")
            generate_ideas.SYSTEM_PROMPT = get_optimized_prompt("idea_generation", 
                                                              generate_ideas.SYSTEM_PROMPT)
        
        # Similar checks for other modules
        if hasattr(perform_experiments, 'SYSTEM_PROMPT'):
            logger.info("Enhancing experiment execution prompts")
            perform_experiments.SYSTEM_PROMPT = get_optimized_prompt("experiments", 
                                                                  perform_experiments.SYSTEM_PROMPT)
        
        if hasattr(perform_writeup, 'SYSTEM_PROMPT'):
            logger.info("Enhancing scientific writeup prompts")
            perform_writeup.SYSTEM_PROMPT = get_optimized_prompt("writeup", 
                                                              perform_writeup.SYSTEM_PROMPT)
        
        logger.info("Successfully enhanced scientific research prompts")
        return True
    
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        return False
    
    except Exception as e:
        logger.error(f"Failed to enhance prompts: {e}")
        return False

def format_prompt_for_mistral(prompt_text: str) -> str:
    """
    Format a prompt for optimal performance with Mistral.
    
    Args:
        prompt_text: Original prompt text
        
    Returns:
        Formatted prompt for better results
    """
    # Simplify and make more direct
    prompt_text = prompt_text.replace("Please", "")
    prompt_text = prompt_text.replace("Could you", "")
    prompt_text = prompt_text.replace("I want you to", "")
    
    # Add clear formatting
    if "\n\n" not in prompt_text:
        # Add line breaks between sentences for clarity
        import re
        prompt_text = re.sub(r'(\. )', r'.\n', prompt_text)
    
    # Add a clear instruction at the end
    if not prompt_text.endswith('.'):
        prompt_text += '.'
    
    return prompt_text 