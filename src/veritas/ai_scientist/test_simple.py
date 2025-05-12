#!/usr/bin/env python3
"""
Simple test script for Veritas AI Scientist

This script directly tests the idea generation functionality using
the Mistral model with RAG capabilities.
"""

import os
import sys
import logging
import json
from pathlib import Path

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/simple_test.log")
    ]
)
logger = logging.getLogger(__name__)

# Get the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

def setup_paths():
    """
    Set up Python paths to include required modules.
    
    Returns:
        Tuple of paths (veritas_path, templates_path)
    """
    veritas_path = PROJECT_ROOT
    templates_path = os.path.join(PROJECT_ROOT, "models", "Cognition")
    
    # Add to Python path if not already there
    if veritas_path not in sys.path:
        sys.path.insert(0, veritas_path)
    if templates_path not in sys.path:
        sys.path.insert(0, templates_path)
    
    logger.info(f"Added to Python path: {veritas_path}, {templates_path}")
    return veritas_path, templates_path

def test_simple_idea_generation():
    """Test simple idea generation with Mistral model"""
    # Create our adapter
    from src.veritas.ai_scientist.adapter import MistralAdapter
    
    # Create a client
    client = MistralAdapter()
    
    # Load seed ideas from template
    template_name = "nanoGPT_lite"
    template_path = os.path.join(PROJECT_ROOT, "models", "Cognition", "templates", template_name)
    seed_ideas_path = os.path.join(template_path, "seed_ideas.json")
    
    logger.info(f"Loading seed ideas from {seed_ideas_path}")
    with open(seed_ideas_path, "r") as f:
        seed_ideas = json.load(f)
    
    # System message
    system_message = f"""You are a creative AI scientist helping to generate novel research ideas for {template_name}.
    Your task is to generate unique research ideas that are:
    1. Novel and interesting
    2. Feasible to implement
    3. Have clear evaluation metrics
    4. Build on existing research
    
    Format the idea as a JSON object with Name, Title, and Experiment fields."""
    
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
    
    logger.info("Sending request to Mistral...")
    try:
        # Generate a response
        response = client.chat_completions_create(
            model="mistral-local-rag",
            messages=messages,
            temperature=0.8,
            max_tokens=500
        )
        
        # Extract content
        idea_text = response.choices[0].message.content
        logger.info(f"Response received: {idea_text}")
        
        # Try to parse JSON
        try:
            # Extract JSON if it's surrounded by ```json and ```
            if "```json" in idea_text and "```" in idea_text.split("```json")[1]:
                idea_text = idea_text.split("```json")[1].split("```")[0].strip()
            elif "```" in idea_text and "```" in idea_text.split("```")[1]:
                idea_text = idea_text.split("```")[1].split("```")[0].strip()
            
            # Parse the idea
            idea = json.loads(idea_text)
            
            # Save the idea to file
            output_file = "results/simple_idea.json"
            os.makedirs("results", exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(idea, f, indent=2)
            
            logger.info(f"Successfully generated and saved idea to {output_file}")
            return True
        except json.JSONDecodeError:
            logger.error(f"Failed to parse idea as JSON: {idea_text}")
            return False
    except Exception as e:
        logger.error(f"Error generating idea: {e}")
        return False

if __name__ == "__main__":
    veritas_path, templates_path = setup_paths()
    logger.info(f"Veritas path: {veritas_path}")
    logger.info(f"Templates path: {templates_path}")
    
    # Run the test
    success = test_simple_idea_generation()
    
    # Print result
    if success:
        print("✅ Successfully generated research idea with Veritas AI Scientist")
        print("See results/simple_idea.json for the generated idea")
    else:
        print("❌ Failed to generate research idea")
        print("See logs for details") 