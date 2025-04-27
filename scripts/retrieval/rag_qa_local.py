#!/usr/bin/env python3
"""
rag_qa_local.py

Local RAG QA system with support for different prompt templates and generation parameters.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import torch
from dataclasses import dataclass
from enum import Enum
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from veritas.config import (
    MODELS_DIR, LOGS_DIR,
    DEFAULT_GEN_MODEL, MAX_NEW_TOKENS,
    TEMPERATURE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "rag_qa.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PromptTemplate(Enum):
    """Available prompt templates."""
    DEFAULT = "default"
    DETAILED = "detailed"
    CONCISE = "concise"
    ANALYTICAL = "analytical"

@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    model_name: str = DEFAULT_GEN_MODEL
    max_new_tokens: int = MAX_NEW_TOKENS
    temperature: float = TEMPERATURE
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True

class PromptManager:
    """Manager for different prompt templates."""
    
    TEMPLATES = {
        PromptTemplate.DEFAULT: """
Context: {context}

Question: {question}

Answer the question based on the context above. If the context doesn't contain relevant information, say "I don't have enough information to answer this question."
""",
        PromptTemplate.DETAILED: """
Context: {context}

Question: {question}

Please provide a detailed answer to the question based on the context above. Include relevant facts and explanations. If the context doesn't contain relevant information, say "I don't have enough information to answer this question."
""",
        PromptTemplate.CONCISE: """
Context: {context}

Question: {question}

Provide a brief, direct answer to the question based on the context above. If the context doesn't contain relevant information, say "I don't have enough information to answer this question."
""",
        PromptTemplate.ANALYTICAL: """
Context: {context}

Question: {question}

Analyze the question and context carefully. Provide a well-reasoned answer that:
1. Addresses the key points in the question
2. Uses specific information from the context
3. Explains the reasoning behind the answer

If the context doesn't contain relevant information, say "I don't have enough information to answer this question."
"""
    }
    
    @classmethod
    def get_prompt(cls, template: PromptTemplate, context: str, question: str) -> str:
        """Get formatted prompt from template."""
        return cls.TEMPLATES[template].format(
            context=context,
            question=question
        )

class OutputFormatter:
    """Formatter for different output formats."""
    
    @staticmethod
    def format_json(answer: str, sources: List[Dict]) -> str:
        """Format output as JSON."""
        return json.dumps({
            "answer": answer,
            "sources": sources
        }, indent=2)
    
    @staticmethod
    def format_text(answer: str, sources: List[Dict]) -> str:
        """Format output as plain text."""
        output = [f"Answer: {answer}\n"]
        if sources:
            output.append("Sources:")
            for i, source in enumerate(sources, 1):
                output.append(f"{i}. {source.get('text', '')[:200]}...")
                if source.get('metadata'):
                    output.append(f"   Source: {source.get('metadata', {}).get('source', 'Unknown')}")
        return "\n".join(output)
    
    @staticmethod
    def format_markdown(answer: str, sources: List[Dict]) -> str:
        """Format output as markdown."""
        output = [f"## Answer\n\n{answer}\n"]
        if sources:
            output.append("## Sources\n")
            for i, source in enumerate(sources, 1):
                output.append(f"### Source {i}\n")
                output.append(source.get('text', ''))
                if source.get('metadata'):
                    output.append("\n**Metadata:**")
                    for key, value in source['metadata'].items():
                        output.append(f"- **{key}**: {value}")
                output.append("")
        return "\n".join(output)

class RAGQA:
    """RAG-based Question Answering system."""
    
    def __init__(
        self,
        search_results: List[Dict],
        model: Any,
        generation_config: Optional[GenerationConfig] = None,
        prompt_template: PromptTemplate = PromptTemplate.DEFAULT
    ):
        self.search_results = search_results
        self.model = model
        self.generation_config = generation_config or GenerationConfig()
        self.prompt_template = prompt_template
    
    def _prepare_context(self) -> str:
        """Prepare context from search results."""
        return "\n\n".join(result['text'] for result in self.search_results)
    
    def _generate_answer(self, prompt: str) -> str:
        """Generate answer using the model."""
        try:
            inputs = self.model.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.generation_config.max_new_tokens,
                temperature=self.generation_config.temperature,
                top_p=self.generation_config.top_p,
                top_k=self.generation_config.top_k,
                repetition_penalty=self.generation_config.repetition_penalty,
                do_sample=self.generation_config.do_sample
            )
            return self.model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I encountered an error while generating the answer."

def answer_question(
    question: str,
    search_results: List[Dict],
    model: Any,
    generation_config: Optional[GenerationConfig] = None,
    prompt_template: PromptTemplate = PromptTemplate.DEFAULT,
    output_format: str = "json"
) -> str:
    """
    Answer a question using RAG.
    
    Args:
        question: Question to answer
        search_results: List of search results
        model: Language model instance
        generation_config: Generation configuration
        prompt_template: Prompt template to use
        output_format: Output format (json, text, markdown)
    
    Returns:
        Formatted answer with sources
    """
    # Initialize RAG QA system
    rag = RAGQA(
        search_results,
        model,
        generation_config,
        prompt_template
    )
    
    # Prepare context and prompt
    context = rag._prepare_context()
    prompt = PromptManager.get_prompt(prompt_template, context, question)
    
    # Generate answer
    answer = rag._generate_answer(prompt)
    
    # Format output
    if output_format == "json":
        return OutputFormatter.format_json(answer, search_results)
    elif output_format == "text":
        return OutputFormatter.format_text(answer, search_results)
    elif output_format == "markdown":
        return OutputFormatter.format_markdown(answer, search_results)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

if __name__ == "__main__":
    import argparse
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    parser = argparse.ArgumentParser(description="RAG-based Question Answering")
    parser.add_argument("question", type=str, help="Question to answer")
    parser.add_argument("--search-results", type=str, required=True,
                      help="Path to JSON file containing search results")
    parser.add_argument("--model-name", type=str, default=DEFAULT_GEN_MODEL,
                      help="Name of the language model to use")
    parser.add_argument("--prompt-template", type=str,
                      choices=[t.value for t in PromptTemplate],
                      default=PromptTemplate.DEFAULT.value,
                      help="Prompt template to use")
    parser.add_argument("--output-format", type=str,
                      choices=["json", "text", "markdown"],
                      default="json", help="Output format")
    
    args = parser.parse_args()
    
    # Load search results
    with open(args.search_results, 'r') as f:
        search_results = json.load(f)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model.tokenizer = tokenizer
    model.device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(model.device)
    
    # Create generation config
    generation_config = GenerationConfig(model_name=args.model_name)
    
    # Generate answer
    result = answer_question(
        args.question,
        search_results,
        model,
        generation_config,
        PromptTemplate(args.prompt_template),
        args.output_format
    )
    print(result)
