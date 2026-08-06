"""
Examples: How to use retry_utils for resilient operations
"""

# Example 1: Using @retry decorator with Kaggle API
from src.retry_utils import retry
import kaggle.api as api
import truststore

truststore.inject_into_ssl()

@retry(max_attempts=3, initial_delay=5, on_exceptions=(ConnectionError,))
def download_from_kaggle():
    """Download Kaggle kernel output with automatic retry"""
    api.kernels_output_cli(
        'msbasanth/sharp-llm-icmlde-five-seed-runner',
        path=r'd:\Repositories\sharp-llm\_check_now',
        file_pattern=r'graphcodebert-base.*?metrics\.json',
        force=True,
        quiet=True,
    )

# Just call it—retry logic is transparent
# download_from_kaggle()


# Example 2: Using retry_operation for one-off calls
from src.retry_utils import retry_operation

def check_kernel_status():
    """Check v29 kernel status with retry"""
    result = retry_operation(
        operation=lambda: api.kernels_status('msbasanth/sharp-llm-icmlde-five-seed-runner'),
        operation_name="Kaggle Kernel Status Check",
        max_attempts=3,
        initial_delay=5,
    )
    return result

# status = check_kernel_status()


# Example 3: Using RetryContext for complex logic
from src.retry_utils import RetryContext

def download_with_retry_context():
    """Download with manual control over retry logic"""
    with RetryContext(max_attempts=3, initial_delay=5) as retry:
        for attempt in retry:
            try:
                print(f"Downloading... (attempt {attempt})")
                api.kernels_output_cli(
                    'msbasanth/sharp-llm-icmlde-five-seed-runner',
                    path=r'd:\Repositories\sharp-llm\_check_now',
                    force=True,
                )
                print("✅ Download succeeded!")
                break
            except ConnectionError as e:
                print(f"Connection lost: {e}")
                if attempt >= 3:
                    raise


# Example 4: Using preset configurations
from src.retry_utils import retry, RETRY_PRESETS

@retry(**RETRY_PRESETS['moderate'])
def moderate_retry_download():
    """Use preset 'moderate' configuration (3 attempts, 5s initial)"""
    pass

@retry(**RETRY_PRESETS['aggressive'])
def aggressive_retry_download():
    """Use preset 'aggressive' configuration (5 attempts, 2s initial)"""
    pass


# Example 5: Handle multiple exception types
@retry(
    max_attempts=3,
    initial_delay=5,
    on_exceptions=(ConnectionError, TimeoutError, OSError)
)
def robust_network_call():
    """Retry on connection, timeout, and OS errors"""
    pass


# Integration with existing code
# ==============================================================================
# Replace this (from _fetch_metrics2.py):
# 
#     api.kernels_output_cli(
#         'msbasanth/sharp-llm-icmlde-five-seed-runner',
#         path=r'd:\Repositories\sharp-llm\_check_now',
#         file_pattern=r'metrics\.json|classification_report\.txt|...',
#         force=True,
#         quiet=False,
#     )
#
# With this:
#
#     from src.retry_utils import retry_operation
#     
#     retry_operation(
#         operation=lambda: api.kernels_output_cli(
#             'msbasanth/sharp-llm-icmlde-five-seed-runner',
#             path=r'd:\Repositories\sharp-llm\_check_now',
#             file_pattern=r'metrics\.json|classification_report\.txt|...',
#             force=True,
#             quiet=False,
#         ),
#         operation_name="Download GraphCodeBERT results from Kaggle",
#         max_attempts=3,
#     )
