from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


@dataclass
class ExecutionMetrics:
    """Track execution metrics for tool observability."""
    
    start_time_ms: float = field(default_factory=lambda: time.time() * 1000)
    end_time_ms: Optional[float] = None
    execution_path: List[str] = field(default_factory=list)
    cache_hit: bool = False
    error_type: Optional[str] = None
    
    def add_step(self, step: str) -> None:
        """Add execution step to path."""
        self.execution_path.append(step)
    
    def set_cache_hit(self, hit: bool = True) -> None:
        """Mark if result came from cache."""
        self.cache_hit = hit
    
    def set_error(self, error_type: str) -> None:
        """Mark error type if execution failed."""
        self.error_type = error_type
    
    def finalize(self) -> Dict[str, Any]:
        """Convert metrics to output dict."""
        self.end_time_ms = time.time() * 1000
        latency = int(self.end_time_ms - self.start_time_ms)
        
        result: Dict[str, Any] = {
            "latency_ms": latency,
            "cache_hit": self.cache_hit,
            "execution_path": " → ".join(self.execution_path) if self.execution_path else "unknown",
            "timestamp_iso": datetime.utcnow().isoformat() + "Z"
        }
        
        if self.error_type:
            result["error_type"] = self.error_type
        
        return result