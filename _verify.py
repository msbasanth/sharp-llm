import json
data = json.load(open(r'd:\Repositories\sharp-llm\outputs\icmlde2026\juliet118\summary\aggregate_metrics.json'))
print('Aggregate Metrics Validation:')
print(f'  Total models: {len(data["models"])}')
for m in data['models']:
    status = 'COMPLETE' if m['completed_seed_count'] == 5 else f'INCOMPLETE ({m["completed_seed_count"]}/5)'
    f1_std = m['macro_f1_mean_std']
    print(f'  - {m["model_name"]:<24} {f1_std:<18} [{status}]')

all_complete = all(m['completed_seed_count'] == 5 for m in data['models'])
print(f'\nReady for publication: {all_complete}')
