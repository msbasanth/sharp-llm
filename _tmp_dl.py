import truststore
truststore.inject_into_ssl()
import kaggle.api as api
import time
import os
import json

# Try 3x with backoff
for attempt in range(3):
    try:
        print(f"Attempt {attempt+1}: Downloading graphcodebert-base...")
        api.kernels_output_cli(
            'msbasanth/sharp-llm-icmlde-five-seed-runner',
            path=r'd:\Repositories\sharp-llm\_check_now',
            file_pattern=r'graphcodebert-base.*?(metrics|report)',
            force=True,
            quiet=True,
        )
        print("Success!")
        break
    except Exception as e:
        print(f"Failed: {type(e).__name__}: {str(e)[:100]}")
        if attempt < 2:
            time.sleep(5 * (2 ** attempt))

# Check what we have
print("\nDownloaded graphcodebert-base seeds:")
for seed in range(42, 47):
    path = f'd:\\Repositories\\sharp-llm\\_check_now\\icmlde2026\\juliet118\\juliet118\\seed_{seed}\\graphcodebert-base\\evaluation\\metrics.json'
    if os.path.exists(path):
        data = json.load(open(path))
        print(f"  seed_{seed}: macro_f1={data.get('macro_f1')}")
    else:
        print(f"  seed_{seed}: MISSING")
