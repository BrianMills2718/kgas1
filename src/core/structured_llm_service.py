#!/usr/bin/env python3
"""
Structured LLM Service - Universal LLM Kit Integration

Provides structured output with Pydantic schema validation using the Universal LLM Kit.
Replaces manual JSON parsing with proper schema-guided output generation.
"""

import logging
import sys
import os
from typing import TypeVar, Type, Dict, Any, Optional
from pydantic import BaseModel, ValidationError
from datetime import datetime

# Add universal_llm_kit to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'universal_llm_kit'))

try:
    import litellm
    from dotenv import load_dotenv
    load_dotenv()  # Load API keys from .env
    litellm_available = True
except ImportError:
    logging.warning("LiteLLM not available - structured output will be simulated")
    litellm_available = False

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

class StructuredLLMService:
    """
    Service for structured LLM output using Universal LLM Kit with Pydantic validation.
    
    Provides fail-fast structured output generation with comprehensive error logging.
    No fallback patterns - fails immediately on validation errors.
    """
    
    def __init__(self, default_model: str = "smart"):
        """
        Initialize structured LLM service.
        
        Args:
            default_model: Default model type to use ("smart", "fast", "code", etc.)
        """
        self.default_model = default_model
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Check LiteLLM availability
        self.available = litellm_available
        
        if self.available:
            try:
                # Test API availability with a simple check
                import os
                has_keys = bool(os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))
                if not has_keys:
                    self.logger.warning("No API keys found in environment")
                    self.available = False
                else:
                    self.logger.info("Structured LLM service initialized with LiteLLM direct")
            except Exception as e:
                self.logger.error(f"Failed to initialize LiteLLM: {e}")
                self.available = False
        else:
            self.logger.warning("LiteLLM not available - service will fail fast")
        
        # Performance tracking
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "validation_failures": 0,
            "llm_failures": 0
        }
    
    def structured_completion(
        self, 
        prompt: str, 
        schema: Type[T], 
        model: Optional[str] = None,
        temperature: float = 0.05,
        max_tokens: int = 32000
    ) -> T:
        """
        Generate structured output with Pydantic schema validation.
        
        Args:
            prompt: The prompt for the LLM
            schema: Pydantic model class for validation
            model: Model type to use (defaults to service default)
            temperature: Temperature for generation
            max_tokens: Maximum tokens for output
            
        Returns:
            Validated instance of the schema
            
        Raises:
            RuntimeError: If Universal LLM Kit is not available
            ValidationError: If response doesn't match schema
            Exception: For other LLM or parsing errors
        """
        self.stats["total_requests"] += 1
        start_time = datetime.now()
        
        # Import monitoring
        try:
            from ..monitoring.structured_output_monitor import get_monitor
            monitor = get_monitor()
        except ImportError:
            monitor = None
        
        # Determine model for tracking
        model_to_use = model or self.default_model
        model_mapping = {
            "smart": "gemini/gemini-2.5-flash",
            "fast": "gemini/gemini-2.5-flash-lite",  
            "code": "gemini/gemini-2.5-flash",
            "reasoning": "gpt-4o",
        }
        litellm_model = model_mapping.get(model_to_use, model_to_use) or "gemini/gemini-2.5-flash"
        
        # Start monitoring if available
        monitor_context = None
        if monitor:
            monitor_context = monitor.track_operation(
                component="structured_llm_service",
                schema_name=schema.__name__,
                model=litellm_model,
                temperature=temperature,
                max_tokens=max_tokens,
                input_text=prompt
            )
        
        try:
            if monitor_context:
                tracker = monitor_context.__enter__()
            
            # Fail fast if service not available
            if not self.available:
                self.stats["llm_failures"] += 1
                error_msg = ("LiteLLM not available or no API keys found. Cannot generate structured output. "
                           "Ensure API keys are set in .env file.")
                if monitor_context and tracker:
                    tracker.set_llm_error(error_msg)
                raise RuntimeError(error_msg)
            
            self.logger.debug(f"Generating structured output with {schema.__name__} schema using {litellm_model}")
            
            # Build schema-guided prompt
            schema_json = schema.model_json_schema()
            structured_prompt = f"""{prompt}

IMPORTANT: Respond with valid JSON that matches this exact schema:

{schema_json}

Your response must be valid JSON only, no markdown formatting."""
            
            # Use LiteLLM directly with structured output
            response = litellm.completion(
                model=litellm_model,
                messages=[{"role": "user", "content": structured_prompt}],
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            response_json = response.choices[0].message.content
            
            self.logger.debug(f"Received response: {response_json[:200]}...")
            
            # Validate with Pydantic
            validated_response = schema.model_validate_json(response_json)
            
            # Success metrics
            self.stats["successful_requests"] += 1
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Update monitoring
            if monitor_context and tracker:
                tracker.set_success(True, validated_response)
            
            self.logger.info(
                f"Structured output successful: {schema.__name__} validated in {execution_time:.3f}s"
            )
            
            return validated_response
            
        except ValidationError as e:
            self.stats["validation_failures"] += 1
            execution_time = (datetime.now() - start_time).total_seconds()
            
            error_msg = f"Schema validation failed for {schema.__name__}: {e}"
            self.logger.error(error_msg)
            self.logger.debug(f"Raw response that failed validation: {response_json}")
            self.logger.debug(f"Expected schema: {schema.model_json_schema()}")
            self.logger.debug(f"Failed after {execution_time:.3f}s")
            
            # Update monitoring
            if monitor_context and tracker:
                tracker.set_validation_error(str(e))
            
            raise ValidationError(
                f"LLM response does not match {schema.__name__} schema: {e}"
            )
            
        except Exception as e:
            self.stats["llm_failures"] += 1
            execution_time = (datetime.now() - start_time).total_seconds()
            
            error_msg = f"LLM generation failed for {schema.__name__}: {e}"
            self.logger.error(error_msg)
            self.logger.debug(f"Model: {litellm_model}, Temperature: {temperature}, Max tokens: {max_tokens}")
            self.logger.debug(f"Failed after {execution_time:.3f}s")
            
            # Update monitoring
            if monitor_context and tracker:
                tracker.set_llm_error(str(e))
            
            raise Exception(f"Structured LLM generation failed: {e}")
            
        finally:
            if monitor_context:
                try:
                    monitor_context.__exit__(None, None, None)
                except Exception as monitor_error:
                    self.logger.warning(f"Monitoring context cleanup failed: {monitor_error}")
    
    async def async_structured_completion(
        self, 
        prompt: str, 
        schema: Type[T], 
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 32000
    ) -> T:
        """
        Async version of structured completion.
        
        Currently wraps the sync version. Could be enhanced for true async if needed.
        """
        return self.structured_completion(
            prompt=prompt,
            schema=schema,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service performance statistics."""
        total = self.stats["total_requests"]
        
        return {
            "service_available": self.available,
            "total_requests": total,
            "successful_requests": self.stats["successful_requests"],
            "validation_failures": self.stats["validation_failures"],
            "llm_failures": self.stats["llm_failures"],
            "success_rate": self.stats["successful_requests"] / total if total > 0 else 0.0,
            "validation_failure_rate": self.stats["validation_failures"] / total if total > 0 else 0.0
        }
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "validation_failures": 0,
            "llm_failures": 0
        }
        self.logger.info("Performance statistics reset")


# Global service instance
_structured_llm_service: Optional[StructuredLLMService] = None

def get_structured_llm_service() -> StructuredLLMService:
    """Get global structured LLM service instance."""
    global _structured_llm_service
    
    if _structured_llm_service is None:
        _structured_llm_service = StructuredLLMService()
    
    return _structured_llm_service


# Convenience functions
def structured_completion(prompt: str, schema: Type[T], **kwargs) -> T:
    """Convenience function for structured completion."""
    service = get_structured_llm_service()
    return service.structured_completion(prompt, schema, **kwargs)


async def async_structured_completion(prompt: str, schema: Type[T], **kwargs) -> T:
    """Convenience function for async structured completion."""
    service = get_structured_llm_service()
    return await service.async_structured_completion(prompt, schema, **kwargs)


if __name__ == "__main__":
    # Simple test
    from pydantic import BaseModel, Field
    from typing import List
    
    class TestResponse(BaseModel):
        message: str = Field(description="A simple test message")
        confidence: float = Field(description="Confidence score", ge=0.0, le=1.0)
        items: List[str] = Field(description="List of items")
    
    try:
        service = StructuredLLMService()
        print(f"Service available: {service.available}")
        print(f"Stats: {service.get_stats()}")
        
        if service.available:
            # Test structured completion
            result = service.structured_completion(
                prompt="Generate a test response with message 'Hello World', confidence 0.9, and items ['test1', 'test2']",
                schema=TestResponse
            )
            
            print(f"Test result: {result}")
            print(f"Updated stats: {service.get_stats()}")
        
    except Exception as e:
        print(f"Test failed: {e}")