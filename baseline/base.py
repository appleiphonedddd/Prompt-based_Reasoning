"""
Base definitions for Prompt Engineering Baselines.

This module defines:
- BaselineResponse: a standardized dataclass for baseline execution outputs
- BaseBaseline: an abstract base class that enforces a common interface
  for all prompt engineering methods

The goal of this module is to provide a consistent, extensible
foundation for integrating different prompt engineering techniques.

Author: Egor Morozov
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

@dataclass
class BaselineResponse:
    
    """Container for a standardized baseline execution response."""

    final_answer = str
    reasoning_trace: str = ""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    num_llm_calls: int = 0
    baseline_type: str = ""