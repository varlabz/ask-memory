# Python Code Instructions

## Overview
This document provides comprehensive guidelines for writing high-quality Python code in the pydantic-ai agent project. These instructions ensure consistency, maintainability, and adherence to Python best practices.

## 1. Code Style and Formatting

### PEP 8 Compliance
- Follow [PEP 8](https://peps.python.org/pep-0008/) style guidelines
- Use 4 spaces for indentation (no tabs)
- Maximum line length: 88 characters (Black formatter standard)
- Use meaningful variable and function names
- Don't use trailing whitespace
- Don't use cast function from typing module, use type annotations instead

### Import Organization
```python
# Standard library imports first
import os
import sys
import asyncio
from typing import Dict, List, Optional, Union

# Third-party imports second
import yaml
from pydantic import BaseModel, Field
from pydantic_ai import Agent

# Local application imports last
from config import AgentConfig, ModelConfig
```

### Naming Conventions
- **Variables and functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private attributes**: prefix with single underscore `_private_var`
- **Magic methods**: double underscores `__init__`

## 2. Type Hints and Documentation

### Type Annotations
Always use type hints for function parameters and return values:

```python
from typing import Dict, List, Optional, Union, Any, Tuple

def load_config(config_path: str, agent_name: str = "default") -> Tuple[AgentConfig, ModelConfig]:
    """Load configuration from YAML file."""
    # Implementation here
    pass

async def process_request(prompt: str, config: AgentConfig) -> Optional[str]:
    """Process user request asynchronously."""
    # Implementation here
    pass
```

### Docstrings
Use Google-style docstrings for all functions, classes, and modules:

```python
def model_factory(model_cfg: ModelConfig) -> str | OpenAIModel:
    """
    Factory function to create model instances based on configuration.
    
    Args:
        model_cfg: Configuration object containing model settings
        
    Returns:
        Model instance or model string for built-in support
        
    Raises:
        ValueError: If provider is not supported
        
    Example:
        config = ModelConfig(model="openai:gpt-4")
        model = model_factory(config)
    """
    # Implementation here
    pass
```

## 3. Error Handling and Validation

### Exception Handling
Use specific exception types and provide meaningful error messages:

```python
try:
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"Configuration file not found: {config_path}")
except yaml.YAMLError as e:
    raise ValueError(f"Invalid YAML configuration: {e}")
```

### Input Validation
Use Pydantic models for data validation:

```python
from pydantic import BaseModel, Field, field_validator

class ModelConfig(BaseModel):
    model: str
    temperature: float
    max_tokens: int
    
    @field_validator("model", mode="before")
    def validate_model(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Model name cannot be empty")
        return v.strip()
```

## 7. Testing Guidelines

### Unit Tests
Write comprehensive unit tests using pytest:

```python
import pytest
from unittest.mock import patch, MagicMock
from src.config import ModelConfig
from src.agent import model_factory

class TestModelFactory:
   
    def test_invalid_provider(self):
        """Test handling of invalid provider."""
        config = ModelConfig(model="invalid/model")
        # Should return the model string for built-in support
        result = model_factory(config)
        assert result == "invalid/model"
    
    @pytest.mark.asyncio
    async def test_async_agent_run(self):
        """Test asynchronous agent execution."""
        # Mock async test
        pass
```

## 9. Security Best Practices

### API Key Management
- Never hardcode API keys

Follow these instructions consistently to maintain high code quality across the project.