"""
Retry utility with exponential backoff for resilient network operations.

Handles disconnects, timeouts, and transient failures with configurable
retry attempts and exponential backoff delays.
"""

import time
import functools
from typing import Callable, Any, TypeVar, Optional, Type, Tuple
from collections.abc import Iterable

F = TypeVar('F', bound=Callable[..., Any])


def retry(
    max_attempts: int = 3,
    initial_delay: int = 5,
    backoff_multiplier: float = 2.0,
    max_delay: int = 60,
    on_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    verbose: bool = True,
) -> Callable[[F], F]:
    """
    Decorator for automatic retry with exponential backoff.
    
    Args:
        max_attempts: Number of retry attempts (default: 3)
        initial_delay: Initial delay in seconds (default: 5s)
        backoff_multiplier: Multiplier for exponential backoff (default: 2.0)
        max_delay: Maximum delay between retries (default: 60s)
        on_exceptions: Tuple of exception types to catch (default: all)
        verbose: Print progress messages (default: True)
    
    Returns:
        Decorated function with retry logic
    
    Example:
        @retry(max_attempts=3, initial_delay=5, on_exceptions=(ConnectionError,))
        def download_from_api():
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    if verbose and attempt > 1:
                        print(f"🔄 Attempt {attempt}/{max_attempts}: {func.__name__}")
                    
                    result = func(*args, **kwargs)
                    
                    if verbose and attempt > 1:
                        print(f"✅ Success on attempt {attempt}")
                    
                    return result
                
                except on_exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts:
                        if verbose:
                            print(f"❌ Attempt {attempt} failed: {type(e).__name__}: {str(e)[:80]}")
                            print(f"⏱️  Waiting {delay}s before retry...")
                        
                        time.sleep(delay)
                        delay = min(int(delay * backoff_multiplier), max_delay)
                    else:
                        if verbose:
                            print(f"❌ All {max_attempts} attempts failed")
            
            # All retries exhausted
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator


class RetryContext:
    """
    Context manager for retry logic with exponential backoff.
    
    Example:
        with RetryContext(max_attempts=3) as retry:
            for _ in retry:
                try:
                    api.download_something()
                    break
                except ConnectionError:
                    pass  # Retry will handle it
    """
    
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: int = 5,
        backoff_multiplier: float = 2.0,
        max_delay: int = 60,
        verbose: bool = True,
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.backoff_multiplier = backoff_multiplier
        self.max_delay = max_delay
        self.verbose = verbose
        self.attempt = 0
        self.delay = initial_delay
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
    
    def __iter__(self):
        return self
    
    def __next__(self):
        self.attempt += 1
        
        if self.attempt > self.max_attempts:
            raise StopIteration
        
        if self.verbose and self.attempt > 1:
            print(f"🔄 Attempt {self.attempt}/{self.max_attempts}")
            print(f"⏱️  Waiting {self.delay}s before retry...")
            time.sleep(self.delay)
            self.delay = min(int(self.delay * self.backoff_multiplier), self.max_delay)
        
        return self.attempt


def retry_operation(
    operation: Callable[[], Any],
    operation_name: str = "Operation",
    max_attempts: int = 3,
    initial_delay: int = 5,
    on_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    verbose: bool = True,
) -> Any:
    """
    Execute an operation with automatic retry and exponential backoff.
    
    Args:
        operation: Callable that performs the operation
        operation_name: Name for progress messages
        max_attempts: Number of retry attempts
        initial_delay: Initial delay in seconds
        on_exceptions: Exception types to catch and retry
        verbose: Print progress messages
    
    Returns:
        Result of the operation
    
    Raises:
        Last exception if all retries fail
    
    Example:
        result = retry_operation(
            lambda: api.kernels_output(...),
            operation_name="Download from Kaggle",
            max_attempts=3
        )
    """
    delay = initial_delay
    last_exception = None
    
    for attempt in range(1, max_attempts + 1):
        try:
            if verbose:
                status = "Retrying" if attempt > 1 else "Starting"
                print(f"🔄 {status} {operation_name} (Attempt {attempt}/{max_attempts})")
            
            result = operation()
            
            if verbose and attempt > 1:
                print(f"✅ {operation_name} succeeded on attempt {attempt}")
            
            return result
        
        except on_exceptions as e:
            last_exception = e
            
            if attempt < max_attempts:
                if verbose:
                    print(f"❌ {operation_name} failed: {type(e).__name__}")
                    print(f"⏱️  Waiting {delay}s before retry...")
                
                time.sleep(delay)
                delay = int(delay * 2.0)  # 5s -> 10s -> 20s
            else:
                if verbose:
                    print(f"❌ {operation_name} failed after {max_attempts} attempts")
    
    if last_exception:
        raise last_exception


# Preset configurations for common scenarios
RETRY_PRESETS = {
    'aggressive': {
        'max_attempts': 5,
        'initial_delay': 2,
        'backoff_multiplier': 2.0,
        'max_delay': 30,
    },
    'moderate': {
        'max_attempts': 3,
        'initial_delay': 5,
        'backoff_multiplier': 2.0,
        'max_delay': 60,
    },
    'conservative': {
        'max_attempts': 2,
        'initial_delay': 10,
        'backoff_multiplier': 1.5,
        'max_delay': 30,
    },
}
