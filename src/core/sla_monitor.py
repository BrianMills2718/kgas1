"""
SLA (Service Level Agreement) monitoring for KGAS.

Defines and enforces performance thresholds for all operations.
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
from pathlib import Path
import aiofiles

from ..monitoring.performance_tracker import get_performance_tracker, PerformanceTracker

logger = logging.getLogger(__name__)


class SLASeverity(Enum):
    """SLA violation severity levels."""
    WARNING = "warning"      # Approaching threshold
    VIOLATION = "violation"  # Exceeded threshold
    CRITICAL = "critical"    # Severely exceeded


@dataclass
class SLAThreshold:
    """Performance threshold for an operation."""
    operation: str
    max_duration: float           # Maximum acceptable duration (seconds)
    warning_duration: float       # Warning threshold (seconds)
    critical_duration: float      # Critical threshold (seconds)
    max_error_rate: float        # Maximum acceptable error rate (0-1)
    min_success_rate: float      # Minimum success rate (0-1)
    evaluation_window: int       # Window size for rate calculations
    
    def check_duration(self, duration: float) -> Optional[SLASeverity]:
        """Check if duration violates SLA."""
        if duration >= self.critical_duration:
            return SLASeverity.CRITICAL
        elif duration >= self.max_duration:
            return SLASeverity.VIOLATION
        elif duration >= self.warning_duration:
            return SLASeverity.WARNING
        return None
    
    def check_error_rate(self, error_rate: float) -> Optional[SLASeverity]:
        """Check if error rate violates SLA."""
        if error_rate > self.max_error_rate:
            return SLASeverity.VIOLATION
        elif error_rate > self.max_error_rate * 0.8:  # 80% of threshold
            return SLASeverity.WARNING
        return None


@dataclass
class SLAViolation:
    """Record of an SLA violation."""
    operation: str
    severity: SLASeverity
    violation_type: str  # "duration" or "error_rate"
    actual_value: float
    threshold_value: float
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class SLAMonitor:
    """
    Monitors operations against defined SLA thresholds.
    
    Features:
    - Configurable thresholds per operation
    - Real-time violation detection
    - Alert callbacks
    - Historical violation tracking
    - Automatic threshold recommendations
    """
    
    # Default SLA thresholds for common operations
    DEFAULT_SLAS = {
        "tool_execution": SLAThreshold(
            operation="tool_execution",
            max_duration=5.0,
            warning_duration=4.0,
            critical_duration=10.0,
            max_error_rate=0.05,
            min_success_rate=0.95,
            evaluation_window=100
        ),
        "database_query": SLAThreshold(
            operation="database_query",
            max_duration=1.0,
            warning_duration=0.8,
            critical_duration=3.0,
            max_error_rate=0.01,
            min_success_rate=0.99,
            evaluation_window=1000
        ),
        "api_request": SLAThreshold(
            operation="api_request",
            max_duration=2.0,
            warning_duration=1.5,
            critical_duration=5.0,
            max_error_rate=0.02,
            min_success_rate=0.98,
            evaluation_window=500
        ),
        "document_processing": SLAThreshold(
            operation="document_processing",
            max_duration=30.0,
            warning_duration=25.0,
            critical_duration=60.0,
            max_error_rate=0.1,
            min_success_rate=0.9,
            evaluation_window=50
        ),
        "pipeline_execution": SLAThreshold(
            operation="pipeline_execution",
            max_duration=120.0,
            warning_duration=100.0,
            critical_duration=300.0,
            max_error_rate=0.15,
            min_success_rate=0.85,
            evaluation_window=20
        )
    }
    
    def __init__(self,
                 performance_tracker: Optional[PerformanceTracker] = None,
                 config_path: Optional[Path] = None):
        """
        Initialize SLA monitor.
        
        Args:
            performance_tracker: Performance tracker instance
            config_path: Path to SLA configuration file
        """
        self.tracker = performance_tracker or get_performance_tracker()
        self.config_path = config_path or Path("sla_config.json")
        
        # SLA thresholds
        self._thresholds: Dict[str, SLAThreshold] = self.DEFAULT_SLAS.copy()
        
        # Violation history
        self._violations: List[SLAViolation] = []
        self._violation_counts: Dict[str, int] = {}
        
        # Alert callbacks
        self._alert_handlers: List[Callable] = []
        
        # Statistics
        self._stats = {
            "total_checks": 0,
            "total_violations": 0,
            "critical_violations": 0,
            "operations_monitored": set()
        }
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        # Flag to track if config is loaded
        self._config_loaded = False
        
        # Monitoring task (will be started when needed)
        self._monitoring_task = None
        
    async def _load_config(self):
        """Load SLA configuration from file."""
        try:
            if self.config_path.exists():
                async with aiofiles.open(self.config_path, 'r') as f:
                    config = json.loads(await f.read())
                    
                    for op, threshold_data in config.get("thresholds", {}).items():
                        self._thresholds[op] = SLAThreshold(**threshold_data)
                        
                logger.info(f"Loaded SLA config for {len(self._thresholds)} operations")
        except Exception as e:
            logger.error(f"Failed to load SLA config: {e}")
    
    async def _ensure_config_loaded(self):
        """Ensure configuration is loaded."""
        if not self._config_loaded:
            await self._load_config()
            self._config_loaded = True
    
    async def _ensure_monitoring_started(self):
        """Ensure monitoring task is started."""
        if self._monitoring_task is None:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def _save_config(self):
        """Save current SLA configuration."""
        try:
            config = {
                "thresholds": {
                    op: {
                        "operation": t.operation,
                        "max_duration": t.max_duration,
                        "warning_duration": t.warning_duration,
                        "critical_duration": t.critical_duration,
                        "max_error_rate": t.max_error_rate,
                        "min_success_rate": t.min_success_rate,
                        "evaluation_window": t.evaluation_window
                    }
                    for op, t in self._thresholds.items()
                },
                "stats": {
                    "total_checks": self._stats["total_checks"],
                    "total_violations": self._stats["total_violations"],
                    "critical_violations": self._stats["critical_violations"]
                }
            }
            
            async with aiofiles.open(self.config_path, 'w') as f:
                await f.write(json.dumps(config, indent=2))
                
        except Exception as e:
            logger.error(f"Failed to save SLA config: {e}")
    
    async def set_sla(self, operation: str, threshold: SLAThreshold):
        """Set SLA threshold for an operation."""
        async with self._lock:
            self._thresholds[operation] = threshold
            await self._save_config()
            logger.info(f"Set SLA for {operation}: max={threshold.max_duration}s")
    
    async def check_operation(self, 
                            operation: str, 
                            duration: float,
                            success: bool,
                            metadata: Optional[Dict] = None) -> Optional[SLAViolation]:
        """
        Check if an operation violates SLA.
        
        Args:
            operation: Operation name
            duration: Operation duration in seconds
            success: Whether operation succeeded
            metadata: Optional metadata
            
        Returns:
            SLAViolation if threshold exceeded, None otherwise
        """
        await self._ensure_config_loaded()
        await self._ensure_monitoring_started()
        
        async with self._lock:
            self._stats["total_checks"] += 1
            self._stats["operations_monitored"].add(operation)
            
            # Get threshold
            threshold = self._thresholds.get(operation)
            if not threshold:
                # No SLA defined for this operation
                return None
            
            violations = []
            
            # Check duration
            if success:  # Only check duration for successful operations
                severity = threshold.check_duration(duration)
                if severity:
                    violation = SLAViolation(
                        operation=operation,
                        severity=severity,
                        violation_type="duration",
                        actual_value=duration,
                        threshold_value=threshold.max_duration,
                        timestamp=datetime.now().isoformat(),
                        metadata=metadata or {}
                    )
                    violations.append(violation)
            
            # Check error rate (need historical data)
            stats = await self.tracker.get_operation_stats(operation)
            if "error" not in stats:
                error_rate = 1.0 - stats.get("success_rate", 1.0)
                severity = threshold.check_error_rate(error_rate)
                if severity:
                    violation = SLAViolation(
                        operation=operation,
                        severity=severity,
                        violation_type="error_rate",
                        actual_value=error_rate,
                        threshold_value=threshold.max_error_rate,
                        timestamp=datetime.now().isoformat(),
                        metadata=metadata or {}
                    )
                    violations.append(violation)
            
            # Record violations
            for violation in violations:
                await self._record_violation(violation)
            
            # Return most severe violation
            if violations:
                return max(violations, key=lambda v: 
                    [SLASeverity.WARNING, SLASeverity.VIOLATION, SLASeverity.CRITICAL].index(v.severity)
                )
            
            return None
    
    async def _record_violation(self, violation: SLAViolation):
        """Record an SLA violation."""
        self._violations.append(violation)
        self._violation_counts[violation.operation] = \
            self._violation_counts.get(violation.operation, 0) + 1
        
        self._stats["total_violations"] += 1
        if violation.severity == SLASeverity.CRITICAL:
            self._stats["critical_violations"] += 1
        
        # Trigger alerts
        for handler in self._alert_handlers:
            try:
                await handler(violation)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
        
        # Log violation
        logger.warning(
            f"SLA {violation.severity.value} for {violation.operation}: "
            f"{violation.violation_type}={violation.actual_value:.3f} "
            f"(threshold={violation.threshold_value:.3f})"
        )
    
    async def _monitoring_loop(self):
        """Continuous monitoring loop."""
        while True:
            try:
                # Check recent operations every 10 seconds
                await asyncio.sleep(10)
                
                # Get performance summary
                summary = await self.tracker.get_system_summary()
                
                # Check each operation
                for op, stats in summary.get("operations", {}).items():
                    if op in self._thresholds:
                        # Check recent performance
                        recent_mean = stats.get("recent_mean", 0)
                        success_rate = stats.get("success_rate", 1)
                        
                        await self.check_operation(
                            operation=op,
                            duration=recent_mean,
                            success=success_rate > 0.5,
                            metadata={"source": "monitoring_loop"}
                        )
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
    
    def register_alert_handler(self, handler: Callable):
        """
        Register an alert handler for SLA violations.
        
        Handler signature: async def handler(violation: SLAViolation)
        """
        self._alert_handlers.append(handler)
        logger.info(f"Registered SLA alert handler: {handler.__name__}")
    
    async def get_violation_history(self, 
                                  operation: Optional[str] = None,
                                  severity: Optional[SLASeverity] = None,
                                  hours: int = 24) -> List[SLAViolation]:
        """Get recent SLA violations."""
        async with self._lock:
            cutoff = datetime.now() - timedelta(hours=hours)
            
            violations = []
            for v in self._violations:
                # Filter by time
                v_time = datetime.fromisoformat(v.timestamp)
                if v_time < cutoff:
                    continue
                    
                # Filter by operation
                if operation and v.operation != operation:
                    continue
                    
                # Filter by severity
                if severity and v.severity != severity:
                    continue
                    
                violations.append(v)
            
            return violations
    
    async def get_sla_report(self) -> Dict[str, Any]:
        """Generate SLA compliance report."""
        async with self._lock:
            report = {
                "summary": {
                    "total_operations": len(self._stats["operations_monitored"]),
                    "total_checks": self._stats["total_checks"],
                    "total_violations": self._stats["total_violations"],
                    "critical_violations": self._stats["critical_violations"],
                    "violation_rate": (
                        self._stats["total_violations"] / 
                        max(1, self._stats["total_checks"])
                    )
                },
                "operations": {}
            }
            
            # Per-operation report
            for op in self._stats["operations_monitored"]:
                violations = self._violation_counts.get(op, 0)
                threshold = self._thresholds.get(op)
                
                op_report = {
                    "violations": violations,
                    "has_sla": threshold is not None
                }
                
                if threshold:
                    op_report["sla"] = {
                        "max_duration": threshold.max_duration,
                        "max_error_rate": threshold.max_error_rate
                    }
                
                report["operations"][op] = op_report
            
            return report
    
    async def recommend_sla(self, operation: str) -> Optional[SLAThreshold]:
        """
        Recommend SLA thresholds based on historical performance.
        
        Args:
            operation: Operation to analyze
            
        Returns:
            Recommended SLA threshold or None if insufficient data
        """
        stats = await self.tracker.get_operation_stats(operation)
        if "error" in stats or stats.get("sample_count", 0) < 100:
            return None
        
        # Base recommendations on current performance + buffer
        baseline = stats.get("baseline", {})
        if not baseline:
            return None
        
        # Add 20% buffer to baseline
        recommended = SLAThreshold(
            operation=operation,
            max_duration=baseline["p95"] * 1.2,
            warning_duration=baseline["p95"],
            critical_duration=baseline["p95"] * 2.0,
            max_error_rate=min(0.1, (1 - stats["success_rate"]) * 2),
            min_success_rate=max(0.9, stats["success_rate"] * 0.95),
            evaluation_window=100
        )
        
        return recommended
    
    async def cleanup(self):
        """Clean up resources."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        await self._save_config()


# Global monitor instance
_global_monitor: Optional[SLAMonitor] = None


def get_sla_monitor() -> SLAMonitor:
    """Get or create global SLA monitor."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = SLAMonitor()
    return _global_monitor