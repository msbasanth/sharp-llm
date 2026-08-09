import os, json, statistics

base    = r'd:\Repositories\sharp-llm\outputs\icmlde2026\juliet118'
summary = r'd:\Repositories\sharp-llm\outputs\icmlde2026\juliet118\summary\aggregate_metrics.json'

variants = ['codet5-small', 'codet5-base', 'codebert-base', 'graphcodebert-base']
seeds    = [42, 43, 44, 45, 46]

print('=' * 75)
print('JULIET-118 RESULTS VERIFICATION')
print('=' * 75)

# 1. File integrity
print('\n[1] File integrity check')
errors = []
all_metrics = {v: {} for v in variants}
for v in variants:
    for seed in seeds:
        p = os.path.join(base, 'seed_{}'.format(seed), v, 'evaluation', 'metrics.json')
        if not os.path.exists(p):
            errors.append('MISSING: seed_{}/{}'.format(seed, v))
        else:
            try:
                all_metrics[v][seed] = json.load(open(p))
            except Exception as e:
                errors.append('INVALID JSON: seed_{}/{}: {}'.format(seed, v, e))
if errors:
    for e in errors: print('  x ' + e)
else:
    print('  OK  All 20 metrics.json valid (4 models x 5 seeds)')

# 2. Required fields
print('\n[2] Required fields check')
required_keys = ['macro_f1', 'accuracy', 'mcc']
field_errors = []
for v in variants:
    for seed in seeds:
        data = all_metrics[v].get(seed)
        if data is None: continue
        missing = [k for k in required_keys if k not in data]
        if missing:
            field_errors.append('seed_{}/{} missing: {}'.format(seed, v, missing))
if field_errors:
    for e in field_errors: print('  x ' + e)
else:
    print('  OK  All required fields present: {}'.format(required_keys))

# 3. Value range
print('\n[3] Value range check (all in [0,1])')
range_errors = []
for v in variants:
    for seed in seeds:
        data = all_metrics[v].get(seed)
        if data is None: continue
        for key in ['macro_f1', 'accuracy', 'mcc']:
            val = data.get(key)
            if val is not None and not (0.0 <= val <= 1.0):
                range_errors.append('seed_{}/{} {}={} OUT OF RANGE'.format(seed, v, key, val))
if range_errors:
    for e in range_errors: print('  x ' + e)
else:
    print('  OK  All values within [0, 1]')

# 4. Cross-validate aggregate
print('\n[4] Aggregate cross-validation (recompute vs saved)')
agg = json.load(open(summary))
all_match = True
for m in agg['models']:
    v       = m['variant']
    sv_mean = m['macro_f1_mean']
    sv_std  = m['macro_f1_std']
    f1s     = [all_metrics[v][s]['macro_f1'] for s in seeds if s in all_metrics[v]]
    c_mean  = statistics.mean(f1s)
    c_std   = statistics.stdev(f1s)
    ok = abs(c_mean - sv_mean) < 0.0001 and abs(c_std - sv_std) < 0.0001
    if not ok: all_match = False
    tag = 'OK' if ok else 'MISMATCH'
    name = m['model_name']
    print('  {} {:<24}  calc_mean={:.4f} saved={:.4f}  calc_std={:.4f} saved={:.4f}'.format(
        tag, name, c_mean, sv_mean, c_std, sv_std))
if all_match:
    print('  All aggregated values match recomputed values')

# 5. Per-seed F1 table
print('\n[5] Per-seed macro-F1 table')
print('  {:<24}  {}  mean    std'.format('Model', '  '.join('s{}'.format(s) for s in seeds)))
print('  ' + '-' * 70)
for v in variants:
    f1s = [all_metrics[v][s]['macro_f1'] for s in seeds]
    cols = '  '.join('{:.4f}'.format(f) for f in f1s)
    print('  {:<24}  {}  {:.4f}  {:.4f}'.format(v, cols, statistics.mean(f1s), statistics.stdev(f1s)))

# 6. Accuracy + MCC
print('\n[6] Accuracy and MCC')
print('  {:<24}  acc_mean  acc_std   mcc_mean  mcc_std'.format('Model'))
print('  ' + '-' * 60)
for v in variants:
    accs = [all_metrics[v][s]['accuracy'] for s in seeds]
    mccs = [all_metrics[v][s]['mcc'] for s in seeds]
    print('  {:<24}  {:.4f}    {:.4f}    {:.4f}    {:.4f}'.format(
        v, statistics.mean(accs), statistics.stdev(accs),
           statistics.mean(mccs), statistics.stdev(mccs)))
