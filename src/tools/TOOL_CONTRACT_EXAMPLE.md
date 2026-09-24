# Historical unified tool contract example

This proposal was moved from the former tool instruction file. Check the current implementation before using it.

1. **Create Unified Tool Interface Contract**
```python
# src/tools/base_classes/tool_protocol.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

class ToolStatus(Enum):
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    MAINTENANCE = "maintenance"

@dataclass(frozen=True)
class ToolRequest:
    """Standardized tool input format"""
    tool_id: str
    operation: str
    input_data: Any
    parameters: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None
    validation_mode: bool = False

@dataclass(frozen=True)
class ToolResult:
    """Standardized tool output format"""
    tool_id: str
    status: str  # "success" or "error"
    data: Any
    metadata: Dict[str, Any]
    execution_time: float
    memory_used: int
    error_code: Optional[str] = None
    error_message: Optional[str] = None

@dataclass(frozen=True)
class ToolContract:
    """Tool capability and requirement specification"""
    tool_id: str
    name: str
    description: str
    category: str  # "graph", "table", "vector", "cross_modal"
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    dependencies: List[str]
    performance_requirements: Dict[str, Any]
    error_conditions: List[str]

class UnifiedTool(ABC):
    """Contract all tools MUST implement"""

    @abstractmethod
    def get_contract(self) -> ToolContract:
        """Return tool contract specification"""
        pass

    @abstractmethod
    def execute(self, request: ToolRequest) -> ToolResult:
        """Execute tool operation with standardized input/output"""
        pass

    @abstractmethod
    def validate_input(self, input_data: Any) -> bool:
        """Validate input against tool contract"""
        pass

    @abstractmethod
    def health_check(self) -> ToolResult:
        """Check tool health and readiness"""
        pass

    @abstractmethod
    def get_status(self) -> ToolStatus:
        """Get current tool status"""
        pass

    @abstractmethod
    def cleanup(self) -> bool:
        """Clean up tool resources"""
        pass
```
