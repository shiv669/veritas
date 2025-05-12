#!/usr/bin/env python3
"""
Veritas AI Scientist - Main Entry Point

This script provides a scientific research assistant that uses the local Mistral RAG system
to generate research ideas, design experiments, and produce scientific writeups.

Usage:
    python run_scientist.py --experiment nanoGPT_lite --num-ideas 2
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import json

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/veritas_scientist.log")
    ]
)
logger = logging.getLogger(__name__)

# Get the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

def setup_paths():
    """
    Set up Python paths to include required components.
    
    Returns:
        Tuple of paths (veritas_path, templates_path)
    """
    # Add the project root to sys.path if not already there
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    
    # Templates path (research templates)
    templates_path = os.path.join(PROJECT_ROOT, "models", "Cognition")
    if not os.path.exists(templates_path):
        logger.error(f"Scientific templates path not found: {templates_path}")
        raise FileNotFoundError(f"Scientific templates not found at {templates_path}")
    
    # Add templates path to sys.path if not already there
    if templates_path not in sys.path:
        sys.path.append(templates_path)
    
    logger.info(f"Added to Python path: {PROJECT_ROOT}, {templates_path}")
    return PROJECT_ROOT, templates_path

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Run Veritas AI Scientist")
    
    # Add arguments
    parser.add_argument(
        "--skip-idea-generation",
        action="store_true",
        help="Skip idea generation and load existing ideas",
    )
    parser.add_argument(
        "--skip-novelty-check",
        action="store_true",
        help="Skip novelty check and use existing ideas",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="nanoGPT_lite",
        help="Research template to use for the scientific inquiry",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistral-local-rag",
        help="Model to use for research (default: mistral-local-rag)",
    )
    parser.add_argument(
        "--writeup",
        type=str,
        default="latex",
        choices=["latex"],
        help="What format to use for writeup",
    )
    parser.add_argument(
        "--improvement",
        action="store_true",
        help="Improve based on reviews.",
    )
    parser.add_argument(
        "--num-ideas",
        type=int,
        default=2,
        help="Number of ideas to generate (lower is better for performance)",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="openalex",
        choices=["semanticscholar", "openalex"],
        help="Scholar engine to use (openalex doesn't require API key)",
    )
    parser.add_argument(
        "--use-optimized-prompts",
        action="store_true",
        default=True,
        help="Use optimized prompts for better results (recommended)",
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["idea", "experiment", "writeup", "all"],
        default="all",
        help="Which phase to run (default: all phases)"
    )
    parser.add_argument(
        "--use-direct-implementation",
        action="store_true",
        help="Use our simpler implementation for better memory efficiency"
    )
    return parser.parse_args()

def run_veritas_scientist():
    """
    Main function to run Veritas AI Scientist.
    """
    # Set up paths
    veritas_path, templates_path = setup_paths()
    logger.info(f"Veritas path: {veritas_path}")
    logger.info(f"Templates path: {templates_path}")
    
    # Import our modules
    from src.veritas.ai_scientist.custom_llm import initialize_llm_system, MISTRAL_MODEL_NAME
    from src.veritas.ai_scientist.memory_manager import wrap_ai_scientist_functions, clean_memory
    from src.veritas.ai_scientist.prompt_strategy import enhance_scientific_prompts
    
    # Parse arguments
    args = parse_arguments()
    
    # Set the model to our Mistral model if not specified
    if args.model != MISTRAL_MODEL_NAME:
        logger.warning(f"Model '{args.model}' specified, but system only supports {MISTRAL_MODEL_NAME}")
        args.model = MISTRAL_MODEL_NAME
    
    # Initialize the LLM module
    logger.info("Initializing the LLM system...")
    success = initialize_llm_system()
    if not success:
        logger.error("Failed to initialize LLM system")
        return 1
    
    # Apply memory management to prevent OOM errors
    logger.info("Applying memory management...")
    success = wrap_ai_scientist_functions()
    if not success:
        logger.warning("Failed to apply memory management (continuing anyway)")
    
    # Apply optimized prompts if requested
    if args.use_optimized_prompts:
        logger.info("Applying optimized prompts...")
        success = enhance_scientific_prompts()
        if not success:
            logger.warning("Failed to apply optimized prompts (continuing with defaults)")
    
    # Clear memory before starting
    clean_memory(force_gpu=True)
    
    # Import modules based on requested phase
    sys.path.insert(0, templates_path)
    
    if args.phase == "idea":
        # Only run idea generation phase
        from ai_scientist.generate_ideas import generate_ideas, check_idea_novelty
        from ai_scientist.llm import create_client
        
        logger.info("Running idea generation phase only...")
        
        # Create the results directory
        os.makedirs("results", exist_ok=True)
        output_file = f"results/ideas_{args.experiment}.json"
        
        # Create client for the model
        client = create_client(args.model)
        
        # Generate ideas
        try:
            if args.use_direct_implementation:
                # Use our direct implementation
                ideas = generate_ideas_simple(args.experiment, args.model, client, args.num_ideas)
            else:
                # Use standard implementation
                ideas = generate_ideas(args.experiment, args.model, args.num_ideas)
        except Exception as e:
            logger.warning(f"Error with generate_ideas: {e}")
            logger.warning("Falling back to direct implementation")
            ideas = generate_ideas_simple(args.experiment, args.model, client, args.num_ideas)
        
        # Check novelty if not skipped
        if not args.skip_novelty_check:
            try:
                # Fixed: Pass client parameter 
                ideas = check_idea_novelty(ideas, client, args.model)
            except Exception as e:
                logger.warning(f"Error with check_idea_novelty: {e}")
                logger.warning("Skipping novelty check")
        
        # Save ideas to file
        with open(output_file, "w") as f:
            json.dump(ideas, f, indent=2)
        
        logger.info(f"Generated {len(ideas)} ideas, saved to {output_file}")
        return 0
        
    elif args.phase == "experiment":
        # Only run experiment phase (ideas must exist)
        from ai_scientist.perform_experiments import perform_experiments
        
        logger.info("Running experiment phase only...")
        # This would need idea input - not fully implemented
        logger.error("Experiment-only phase not fully implemented yet")
        return 1
        
    elif args.phase == "writeup":
        # Only run writeup phase (experiments must exist)
        from ai_scientist.perform_writeup import perform_writeup
        
        logger.info("Running writeup phase only...")
        # This would need experiment results - not fully implemented
        logger.error("Writeup-only phase not fully implemented yet")
        return 1
    
    else:
        # Run all phases (default)
        # Import UIFramework and DeploymentMode from launch_scientist
        from launch_scientist import UIFramework, DeploymentMode, run_model
        
        # Convert arguments to match expected format
        scientist_args = argparse.Namespace(
            skip_idea_generation=args.skip_idea_generation,
            skip_novelty_check=args.skip_novelty_check,
            experiment=args.experiment,
            model=args.model,
            writeup=args.writeup,
            parallel=0,  # No parallelism for memory reasons
            improvement=args.improvement,
            gpus=None,  # Auto-detect
            num_ideas=args.num_ideas,
            engine=args.engine,
            host="0.0.0.0",
            port=None
        )
    
        # Run Veritas Scientist with our arguments
        logger.info(f"Running Veritas AI Scientist with Mistral RAG model...")
        logger.info(f"Experiment: {args.experiment}, Ideas: {args.num_ideas}")
        
        # Run the model
        run_model(
            ui_framework=UIFramework.TERMINAL,
            deployment_mode=DeploymentMode.LOCAL,
            model_config=None,  # Use defaults
            host=scientist_args.host,
            port=scientist_args.port
        )
    
    logger.info("Veritas AI Scientist completed successfully")
    return 0

# Direct implementation of idea generation with Mistral
def generate_ideas_simple(template_name, model_name, client, num_ideas=2):
    """
    A simplified implementation of idea generation.
    
    Args:
        template_name: Name of the template to use
        model_name: Name of the model to use
        client: LLM client
        num_ideas: Number of ideas to generate
        
    Returns:
        List of generated ideas
    """
    logger.info(f"Using optimized implementation to generate ideas for {template_name}")
    
    # Load seed ideas from template
    template_path = os.path.join(PROJECT_ROOT, "models", "Cognition", "templates", template_name)
    seed_ideas_path = os.path.join(template_path, "seed_ideas.json")
    
    logger.info(f"Loading seed ideas from {seed_ideas_path}")
    with open(seed_ideas_path, "r") as f:
        seed_ideas = json.load(f)
    
    # Load prompt template if available
    prompt_path = os.path.join(template_path, "prompt.json")
    try:
        with open(prompt_path, "r") as f:
            prompt_template = json.load(f)
            logger.info(f"Loaded prompt template from {prompt_path}")
    except:
        prompt_template = None
        logger.warning(f"Failed to load prompt template from {prompt_path}")
    
    # Use a fixed system message for Mistral
    system_message = f"""You are a creative AI scientist helping to generate novel research ideas for {template_name}.
    Your task is to generate unique research ideas that are:
    1. Novel and interesting
    2. Feasible to implement
    3. Have clear evaluation metrics
    4. Build on existing research
    
    Format each idea as a JSON object with Name, Title, and Experiment fields."""
    
    # Generate ideas
    ideas = []
    for i in range(num_ideas):
        logger.info(f"Generating idea {i+1}/{num_ideas}")
        
        # Create a prompt from seed ideas
        seed_examples = json.dumps(seed_ideas[:2], indent=2)
        prompt = f"""Here are some example research ideas:
        
        {seed_examples}
        
        Please generate a new, unique research idea following the same format.
        Make sure it's different from the examples and has a clear experiment design.
        Return only the JSON for the new idea.
        """
        
        # Send to model
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Generate a response
            response = client.chat_completions_create(
                model=model_name,
                messages=messages,
                temperature=0.8,
                max_tokens=500
            )
            
            # Extract content
            idea_text = response.choices[0].message.content
            
            # Try to parse JSON
            try:
                # Extract JSON if it's surrounded by ```json and ```
                if "```json" in idea_text and "```" in idea_text.split("```json")[1]:
                    idea_text = idea_text.split("```json")[1].split("```")[0].strip()
                elif "```" in idea_text and "```" in idea_text.split("```")[1]:
                    idea_text = idea_text.split("```")[1].split("```")[0].strip()
                
                # Parse the idea
                idea = json.loads(idea_text)
                
                # Validate idea format
                if all(key in idea for key in ["Name", "Title", "Experiment"]):
                    ideas.append(idea)
                    logger.info(f"Successfully generated idea: {idea['Name']}")
                else:
                    logger.warning(f"Idea missing required fields: {idea}")
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse idea as JSON: {idea_text}")
                # Print the raw text for debugging
                print(f"Raw response: {idea_text}")
        except Exception as e:
            logger.error(f"Error generating idea: {e}")
    
    return ideas

if __name__ == "__main__":
    try:
        sys.exit(run_veritas_scientist())
    except Exception as e:
        logger.error(f"Error running Veritas AI Scientist: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1) 