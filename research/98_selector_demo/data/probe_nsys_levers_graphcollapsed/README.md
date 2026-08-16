# Superseded: nsys traces taken at GRAPH granularity

These runs used the nsys default `--cuda-graph-trace=graph`, which reports a
whole CUDA-graph replay as a single entry. Under piecewise CUDA graphs that
collapses a 36-layer forward into one row: the traces show 200 GEMM instances
for 200 forwards, and the w4a16 draft appears to run no quantized kernel at
all. Both are artifacts of the granularity, not findings.

Kept because the NVTX range tree in them is unaffected -- ranges are host-side
and were correctly attributed -- and because the artifact is worth being able
to point at. Kernel-class numbers here must not be used.

Superseded by `../probe_nsys_levers/`, taken with `--cuda-graph-trace=node`.
