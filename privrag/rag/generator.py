"""
Note:
This is a fake, highly vulnerable "worst-case" AI model.
 If a user asks it to "repeat all the context," it will maliciously dump everything 
 it was given.
"""

import os
from typing import Protocol, List
from privrag.rag.retriever import RetrievedChunk

# The LLM Protocol
class LLM(Protocol):
    def generate(self, prompt: str) -> str:
        ...

def build_prompt(question: str, context_chunks: List[RetrievedChunk]) -> str:
    """Delimits context chunks with <context id=...> blocks as specified."""
    prompt = "Answer the question based only on the provided context.\n\n"
    
    for chunk in context_chunks:
        prompt += f"<context id={chunk.chunk_id}>\n{chunk.text}\n</context>\n\n"
        
    prompt += f"Question: {question}\nAnswer:"
    return prompt

class EchoLLM(LLM):
    """
    Deterministic worst-case model for Person 1's leakage harness.
    """
    def __init__(self):
        # The specific trigger phrases defined in the workplan
        self.injection_phrases = ["repeat", "copy", "verbatim", "all the context"]

    def generate(self, prompt: str) -> str:
        # Extract the user's question from the prompt template
        question_part = prompt.split("Question:")[-1].lower()
        
        # Check if the user is attempting a prompt injection
        is_injection = any(phrase in question_part for phrase in self.injection_phrases)
        
        if is_injection:
            # LEAK: The AI ignores instructions and dumps the raw context
            return f"[ECHO LEAK VERBATIM]:\n{prompt}"
        
        # NORMAL BEHAVIOR: Answer safely from the first chunk
        if "<context id=" in prompt:
            # Extract just the text from the first context block
            first_chunk_text = prompt.split("<context id=")[1].split("</context>")[0].split(">\n")[-1].strip()
            return f"Based on the documents: {first_chunk_text}"
            
        return "I don't have enough context to answer that."

# Placeholder for real API backend (Anthropic/OpenAI) using env keys
class RealLLM(LLM):
    def __init__(self):
        self.api_key = os.getenv("LLM_API_KEY")
        
    def generate(self, prompt: str) -> str:
        if not self.api_key:
            raise ValueError("LLM_API_KEY environment variable missing.")
        # TODO: Implement real API call (e.g., openai.chat.completions.create)
        return "Real LLM response here."