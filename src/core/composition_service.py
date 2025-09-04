#!/usr/bin/env python3
"""
Composition Service - Bridge between framework and production tools
CRITICAL: This is the convergence point for all tool systems
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Add framework to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tool_compatability" / "poc"))

from framework import ToolFramework, ExtensibleTool, ToolResult
from data_types import DataType
from src.core.tool_contract import get_tool_registry
from src.core.service_manager import ServiceManager
from src.analytics.cross_modal_orchestrator import CrossModalOrchestrator
from src.core.adapter_factory import UniversalAdapterFactory
from src.core.service_bridge import ServiceBridge


class CompositionService:
    """Single source of truth for tool composition"""
    
    def __init__(self, service_manager: ServiceManager = None, config_path: str = None):
        """
        Initialize with both framework and production systems
        
        Args:
            service_manager: Service manager instance
            config_path: Path to configuration file
        """
        from src.core.config_loader import load_service_config
        
        self.service_manager = service_manager or ServiceManager()
        self.framework = ToolFramework()
        self.production_registry = get_tool_registry()
        self.orchestrator = CrossModalOrchestrator(service_manager)
        
        # Load configuration
        config = load_service_config(config_path) if config_path else {}
        
        # Create service bridge with configuration
        self.service_bridge = ServiceBridge(self.service_manager, config)
        
        # Create adapter factory with service bridge
        self.adapter_factory = UniversalAdapterFactory(self.service_bridge)
        
        # Metrics for thesis evidence
        self.composition_metrics = {
            'chains_discovered': 0,
            'tools_adapted': 0,
            'execution_time': [],
            'overhead_percentage': []
        }
        
    def register_any_tool(self, tool: Any) -> bool:
        """
        Register ANY tool regardless of interface
        Returns True if successful
        """
        try:
            # Will implement with adapter factory
            if not self.adapter_factory:
                raise NotImplementedError("Adapter factory not yet created")
                
            adapted = self.adapter_factory.wrap(tool)
            self.framework.register_tool(adapted)
            
            self.composition_metrics['tools_adapted'] += 1
            return True
            
        except Exception as e:
            print(f"❌ Failed to register {tool}: {e}")
            return False
            
    def discover_chains(self, input_type: str, output_type: str) -> List[List[str]]:
        """
        Discover all possible chains from both systems
        """
        # Get chains from framework
        framework_chains = self.framework.find_chains(input_type, output_type)
        
        # TODO: Also query production registry
        
        self.composition_metrics['chains_discovered'] += len(framework_chains)
        return framework_chains
    
    def find_chains(self, input_type: Union[str, DataType], output_type: Union[str, DataType], 
                   domain=None) -> List[List[str]]:
        """
        Find all possible chains (alias for discover_chains with DataType support)
        """
        # Convert string to DataType if needed
        if isinstance(input_type, str):
            input_type = DataType(input_type.lower())
        if isinstance(output_type, str):
            output_type = DataType(output_type.lower())
            
        # Use framework's find_chains with domain if provided
        if domain:
            return self.framework.find_chains(input_type, output_type, domain=domain)
        return self.framework.find_chains(input_type, output_type)
        
    def execute_chain(self, chain: List[str], input_data: Any) -> ToolResult:
        """
        Execute a discovered chain with performance tracking
        """
        start_time = time.time()
        
        # Use framework to execute the chain
        result = self.framework.execute_chain(chain, input_data)
        
        execution_time = time.time() - start_time
        self.composition_metrics['execution_time'].append(execution_time)
        
        return result
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get composition metrics for thesis evidence"""
        return self.composition_metrics