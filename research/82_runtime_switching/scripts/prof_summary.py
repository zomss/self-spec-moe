import json, sys
for f in sys.argv[1:]:
    try:
        d = json.load(open(f))
        print(f)
        items = d.items() if isinstance(d, dict) else []
        for k, v in sorted(items):
            if isinstance(v, dict) and "mean_s" in v:
                print(f"  {k}: {v['mean_s']*1e3:.3f} ms x{v.get('count', 0)}")
            elif isinstance(v, (int, float)):
                print(f"  {k}: {v}")
            elif isinstance(v, dict):
                print(f"  {k}: {json.dumps(v)[:120]}")
    except Exception as e:
        print(f, "ERR", e)
