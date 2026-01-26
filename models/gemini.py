"""
Gemini LLM client implementation.

This module provides a concrete implementation of the BaseLLM interface
using Google's Gemini API. It manages API key resolution, request
configuration, response handling, and error propagation.

Classes:
- GeminiClient: An LLM client that sends prompts to the Gemini API and
  returns standardized LLMResponse objects.

Dependencies:
- Requires a valid Gemini API key, provided either directly or via the
  GEMINI_API_KEY environment variable.

Author: Egor Morozov
"""

import os
from google import genai
from google.genai import types
from .base import BaseLLM, LLMResponse

class GeminiClient(BaseLLM):
    def __init__(self, api_key: str = None, model: str = "gemini-2.0-flash-lite"):
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("Gemini API Key is required.")
        super().__init__(key, model)

        self.client = genai.Client(api_key=self.api_key)

    def generate(self, prompt: str, temperature: float = 0) -> LLMResponse:
        try:
            config = types.GenerateContentConfig(
                temperature=temperature
            )
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config
            )

            text_content = response.text
            usage = response.usage_metadata

            return LLMResponse(
                content=text_content,
                model_name=self.model,
                input_tokens=usage.prompt_token_count if usage else 0,
                output_tokens=usage.candidates_token_count if usage else 0,
                raw_response={"usage": str(usage)}
            )
        except Exception as e:
            raise RuntimeError(f"Gemini API Error: {e}")