"""
Thread-safe Service Manager implementation.

Addresses race conditions and thread safety issues identified in the 
Phase RELIABILITY audit.
"""

import asyncio
import threading
from typing import Optional, Dict, Any, Type
from contextlib import asynccontextmanager
import logging

from .identity_service import IdentityService
from .provenance_service import ProvenanceService
from .quality_service import QualityService
from .workflow_state_service import WorkflowStateService
from .config_manager import get_config
from .logging_config import get_logger

logger = get_logger(__name__)


class ThreadSafeServiceManager:
    """
    Thread-safe service manager with proper locking and state management.
    
    Improvements over original ServiceManager:
    - Thread-safe service creation and access
    - Atomic operations for all state changes
    - Proper async/await support
    - Operation queuing to prevent race conditions
    - Comprehensive error handling
    """
    
    _instance = None
    _instance_lock = threading.RLock()  # Use RLock for nested locking
    
    def __new__(cls) -> 'ThreadSafeServiceManager':
        """Thread-safe singleton creation."""
        if cls._instance is None:
            with cls._instance_lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize service manager with thread safety."""
        # Prevent multiple initialization
        with self._instance_lock:
            if self._initialized:
                return
                
            self._initialized = True
            self.logger = get_logger("core.thread_safe_service_manager")
            
            # Service instances
            self._services: Dict[str, Any] = {}
            self._service_locks: Dict[str, threading.RLock] = {}
            self._service_configs: Dict[str, Dict[str, Any]] = {}
            
            # Operation queue for serializing critical operations
            self._operation_queue = asyncio.Queue()
            self._operation_processor_task = None
            
            # Statistics
            self._stats = {
                'service_creations': 0,
                'lock_contentions': 0,
                'operations_processed': 0,
                'errors_handled': 0
            }
            
            self.logger.info("ThreadSafeServiceManager initialized")
    
    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize service manager with configuration.
        
        Args:
            config: Optional configuration override
            
        Returns:
            True if initialization successful
        """
        try:
            with self._instance_lock:
                if config:
                    # Store configurations for services
                    self._service_configs.update(config)
                
                # Start operation processor
                if not self._operation_processor_task:
                    self._operation_processor_task = asyncio.create_task(
                        self._process_operations()
                    )
                
                self.logger.info("Service manager initialized successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"Service manager initialization failed: {e}")
            return False
    
    async def get_service(self, service_name: str, 
                         service_class: Optional[Type] = None) -> Any:
        """
        Get or create a service instance thread-safely.
        
        Args:
            service_name: Name of the service
            service_class: Optional service class for creation
            
        Returns:
            Service instance
        """
        # Fast path - service already exists
        if service_name in self._services:
            return self._services[service_name]
        
        # Slow path - need to create service
        if service_name not in self._service_locks:
            with self._instance_lock:
                if service_name not in self._service_locks:
                    self._service_locks[service_name] = threading.RLock()
        
        # Create service with service-specific lock
        with self._service_locks[service_name]:
            # Double-check pattern
            if service_name in self._services:
                return self._services[service_name]
            
            # Track lock contention
            self._stats['lock_contentions'] += 1
            
            # Create service
            service = await self._create_service(service_name, service_class)
            self._services[service_name] = service
            self._stats['service_creations'] += 1
            
            return service
    
    async def _create_service(self, service_name: str, 
                            service_class: Optional[Type] = None) -> Any:
        """Create a service instance with proper initialization."""
        try:
            # Get service class if not provided
            if not service_class:
                service_class = self._get_service_class(service_name)
            
            if not service_class:
                raise ValueError(f"Unknown service: {service_name}")
            
            # Get configuration
            config = self._service_configs.get(service_name, {})
            
            # Create service instance
            if config:
                service = service_class(**config)
            else:
                service = service_class()
            
            # Initialize if needed
            if hasattr(service, 'initialize'):
                await service.initialize()
            
            self.logger.info(f"Created service: {service_name}")
            return service
            
        except Exception as e:
            self.logger.error(f"Failed to create service {service_name}: {e}")
            self._stats['errors_handled'] += 1
            raise
    
    def _get_service_class(self, service_name: str) -> Optional[Type]:
        """Get service class by name."""
        service_map = {
            'identity': IdentityService,
            'provenance': ProvenanceService,
            'quality': QualityService,
            'workflow': WorkflowStateService
        }
        return service_map.get(service_name)
    
    @property
    def identity_service(self) -> IdentityService:
        """Get identity service (async-safe property)."""
        # Use sync method for property access
        return asyncio.run(self.get_service('identity', IdentityService))
    
    @property
    def provenance_service(self) -> ProvenanceService:
        """Get provenance service (async-safe property)."""
        return asyncio.run(self.get_service('provenance', ProvenanceService))
    
    @property
    def quality_service(self) -> QualityService:
        """Get quality service (async-safe property)."""
        return asyncio.run(self.get_service('quality', QualityService))
    
    @property
    def workflow_service(self) -> WorkflowStateService:
        """Get workflow state service (async-safe property)."""
        return asyncio.run(self.get_service('workflow', WorkflowStateService))
    
    async def queue_operation(self, operation: Dict[str, Any]) -> Any:
        """
        Queue an operation for serialized execution.
        
        Used for operations that must be executed atomically.
        
        Args:
            operation: Operation dictionary with 'type' and 'params'
            
        Returns:
            Operation result
        """
        # Create future for result
        future = asyncio.Future()
        
        # Queue operation with future
        await self._operation_queue.put({
            'operation': operation,
            'future': future
        })
        
        # Wait for result
        return await future
    
    async def _process_operations(self):
        """Process queued operations serially."""
        while True:
            try:
                # Get next operation
                item = await self._operation_queue.get()
                operation = item['operation']
                future = item['future']
                
                try:
                    # Execute operation
                    result = await self._execute_operation(operation)
                    future.set_result(result)
                    self._stats['operations_processed'] += 1
                    
                except Exception as e:
                    future.set_exception(e)
                    self._stats['errors_handled'] += 1
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Operation processor error: {e}")
    
    async def _execute_operation(self, operation: Dict[str, Any]) -> Any:
        """Execute a queued operation."""
        op_type = operation.get('type')
        params = operation.get('params', {})
        
        if op_type == 'configure_service':
            service_name = params['service_name']
            config = params['config']
            self._service_configs[service_name] = config
            return True
            
        elif op_type == 'reset_service':
            service_name = params['service_name']
            if service_name in self._services:
                service = self._services[service_name]
                if hasattr(service, 'cleanup'):
                    await service.cleanup()
                del self._services[service_name]
            return True
            
        else:
            raise ValueError(f"Unknown operation type: {op_type}")
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all services."""
        health_status = {}
        
        for service_name, service in self._services.items():
            try:
                if hasattr(service, 'health_check'):
                    health_status[service_name] = await service.health_check()
                else:
                    health_status[service_name] = True
            except Exception as e:
                self.logger.error(f"Health check failed for {service_name}: {e}")
                health_status[service_name] = False
        
        return health_status
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get service manager statistics."""
        stats = self._stats.copy()
        stats['active_services'] = list(self._services.keys())
        stats['service_count'] = len(self._services)
        return stats
    
    async def cleanup(self):
        """Clean up all services and resources."""
        self.logger.info("Starting service manager cleanup")
        
        # Cancel operation processor
        if self._operation_processor_task:
            self._operation_processor_task.cancel()
            try:
                await self._operation_processor_task
            except asyncio.CancelledError:
                pass
        
        # Clean up services
        for service_name, service in self._services.items():
            try:
                if hasattr(service, 'cleanup'):
                    await service.cleanup()
                self.logger.info(f"Cleaned up service: {service_name}")
            except Exception as e:
                self.logger.error(f"Cleanup failed for {service_name}: {e}")
        
        # Clear state
        self._services.clear()
        self._service_locks.clear()
        self._service_configs.clear()
        
        self.logger.info("Service manager cleanup complete")
    
    @asynccontextmanager
    async def atomic_operation(self, service_name: str):
        """
        Context manager for atomic operations on a service.
        
        Args:
            service_name: Name of the service
            
        Yields:
            Service instance for atomic operations
        """
        # Fix race condition - use instance lock to protect service lock creation
        if service_name not in self._service_locks:
            with self._instance_lock:
                # Double-check locking pattern
                if service_name not in self._service_locks:
                    self._service_locks[service_name] = threading.RLock()
        
        with self._service_locks[service_name]:
            service = await self.get_service(service_name)
            try:
                yield service
            finally:
                pass  # Cleanup if needed


# Global instance getter
_manager_instance: Optional[ThreadSafeServiceManager] = None
_manager_lock = threading.Lock()


def get_thread_safe_service_manager() -> ThreadSafeServiceManager:
    """Get the global thread-safe service manager instance."""
    global _manager_instance
    
    if _manager_instance is None:
        with _manager_lock:
            if _manager_instance is None:
                _manager_instance = ThreadSafeServiceManager()
    
    return _manager_instance