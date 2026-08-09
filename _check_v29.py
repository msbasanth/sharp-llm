import truststore
truststore.inject_into_ssl()
import kaggle.api as api
import json
import os
import time

print('Checking v29 (GraphCodeBERT-Base) status...')
try:
    status = api.kernels_status('msbasanth/sharp-llm-icmlde-five-seed-runner')
    print(f'Kernel status: {status}')
except Exception as e:
    print(f'Status check failed: {e}')
    
# Check what's locally vs what we expect
expected_seeds = [42, 43, 44, 45, 46]
variants = ['graphcodebert-base']
base_path = r'd:\Repositories\sharp-llm\outputs\icmlde2026\juliet118'

print('\nLocal files check:')
for seed in expected_seeds:
    for variant in variants:
        metrics_file = f'{base_path}\\seed_{seed}\\{variant}\\evaluation\\metrics.json'
        if os.path.exists(metrics_file):
            data = json.load(open(metrics_file))
            f1 = data.get('macro_f1', 'N/A')
            print(f'  seed_{seed}/{variant}: ✓ (F1={f1:.4f})')
        else:
            print(f'  seed_{seed}/{variant}: ✗ MISSING - will retry download')
