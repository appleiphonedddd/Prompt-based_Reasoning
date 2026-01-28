"""
Llama LLM client implementation.

This module provides a concrete implementation of the BaseLLM interface
using the Llama API (OpenAI-compatible endpoint). It handles API key
resolution, request execution, response parsing, and error handling.

Classes:
- LlamaClient: An LLM client that communicates with Llama models via a
  compatible OpenAI-style API and returns standardized LLMResponse objects.

Dependencies:
- Requires a valid Llama API key, provided either directly or via the
  API_KEY environment variable.

Author: Egor Morozov
"""

import os
from openai import OpenAI
from .base import BaseLLM, LLMResponse

class LlamaClient(BaseLLM):

    def __init__(self, api_key: str = None, model: str ="llama3:8b"):
        key = api_key or os.getenv("API_KEY")
        if not key:
            raise ValueError("API Key is required.")
        super().__init__(key, model)
        self.client = OpenAI(api_key=self.api_key, base_url="http://192.168.50.132:11434/v1")
    
    def generate(self, prompt: str, temperature: float = 0) -> LLMResponse:

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )

            message_content = response.choices[0].message.content
            usage = response.usage

            return LLMResponse(
                content=message_content,
                model_name=self.model,
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                raw_response=response.model_dump()
            )
        except Exception as e:
            raise RuntimeError(f"Llama API Error: {e}")