"""Parse torch-profiler traces: execution mode of each model forward.

For every `gpu_model_runner: forward` span, count cudaGraphLaunch vs
cudaLaunchKernel runtime calls inside its window, plus wall duration.
FULL cudagraph ~ 1 graph launch / ~0 kernel launches; PIECEWISE ~ n_layers
graph launches + eager kernels; eager ~ thousands of kernel launches.
"""

import glob
import gzip
import json
import sys


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)


def main(trace_dir):
    paths = sorted(glob.glob(f"{trace_dir}/*.json*"))
    if not paths:
        print(f"no trace files in {trace_dir}")
        return 1
    path = max(paths, key=lambda p: len(p))
    print(f"# {path}")
    d = load(path)
    ev = d["traceEvents"]

    fwd = [
        e for e in ev
        if e.get("ph") == "X" and e.get("name") == "gpu_model_runner: forward"
    ]
    fwd.sort(key=lambda e: e["ts"])
    runtime = [
        e for e in ev
        if e.get("ph") == "X"
        and e.get("cat") in ("cuda_runtime", "cuda_driver")
        and e.get("name") in ("cudaGraphLaunch", "cudaLaunchKernel",
                              "cuLaunchKernel", "cudaMemcpyAsync")
    ]
    print(f"forwards: {len(fwd)}, runtime-launch events: {len(runtime)}")
    print(f"{'#':>3} {'dur_ms':>9} {'graphLaunch':>11} {'launchKernel':>12} "
          f"{'memcpy':>7}")
    for i, f in enumerate(fwd):
        t0, t1 = f["ts"], f["ts"] + f["dur"]
        ing = ink = inm = 0
        for r in runtime:
            if t0 <= r["ts"] < t1:
                if r["name"] == "cudaGraphLaunch":
                    ing += 1
                elif r["name"] in ("cudaLaunchKernel", "cuLaunchKernel"):
                    ink += 1
                else:
                    inm += 1
        print(f"{i:>3} {f['dur'] / 1e3:>9.2f} {ing:>11} {ink:>12} {inm:>7}")

    # Longest CPU ops overall, to catch unexpected hotspots.
    ops = [
        e for e in ev
        if e.get("ph") == "X" and e.get("cat") == "cpu_op" and e.get("dur", 0) > 0
    ]
    ops.sort(key=lambda e: -e["dur"])
    print("\n# longest cpu ops:")
    for e in ops[:10]:
        print(f"  {e['dur'] / 1e3:9.2f} ms  {e['name'][:90]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
