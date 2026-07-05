"""Phase 56 Stage A — overlap PHYSICS microbench on the real 2-node fabric.

The decisive question the ahead-chain composition exists to exploit: on this
fabric, does GPU compute on one stream OVERLAP with inter-node NCCL collectives
on another stream, or does NCCL serialize them? This is a property of the
fabric + NCCL, measured directly here (no fragile spec machinery).

Layout: world = 16 ranks (2 nodes x 8 GPUs). The collective loop is an
all_gather over all 16 ranks at a decode-ish payload, looped to stand in for
the verify's ~2*num_layers inter-node collectives. The compute loop is a
stack of matmuls standing in for the draft forward. We time each solo, then
both concurrent on separate streams.

Reading:
  overlap_ratio = T_concurrent / max(T_comm, T_compute)
  ~1.0  => full overlap (compute hides behind comm)  -> overlap thesis GO
  ~ (T_comm+T_compute)/max(...)  => full serialization -> overlap NO-GO
"""

import os

import torch
import torch.distributed as dist


def _time(fn, iters, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.synchronize()
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters  # ms/iter


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)

    # --- comm: all_gather over all 16 ranks (inter-node), decode-ish payload.
    # ~256 KB/rank matches the Phase-48 knee; loop count ~ verify collective
    # count per step (~2 x 60 layers = 120) folded into iters below.
    payload = 256 * 1024 // 2  # bf16 elems
    send = torch.randn(payload, dtype=torch.bfloat16, device=dev)
    recv = [torch.empty_like(send) for _ in range(world)]

    def comm_once():
        for _ in range(int(os.environ.get('NC',30))):
            dist.all_gather(recv, send)

    # --- compute: matmul stack (draft-forward stand-in), no comm.
    n = 4096
    a = torch.randn(n, n, dtype=torch.bfloat16, device=dev)
    bmat = torch.randn(n, n, dtype=torch.bfloat16, device=dev)

    def compute_once():
        x = a
        for _ in range(int(os.environ.get('NM',40))):
            x = torch.mm(x, bmat)
        return x

    s_comm = torch.cuda.Stream()
    s_comp = torch.cuda.Stream()

    def concurrent_once():
        torch.cuda.current_stream().synchronize()
        with torch.cuda.stream(s_comm):
            for _ in range(int(os.environ.get('NC',30))):
                dist.all_gather(recv, send)
        with torch.cuda.stream(s_comp):
            x = a
            for _ in range(int(os.environ.get('NM',40))):
                x = torch.mm(x, bmat)
        s_comm.synchronize()
        s_comp.synchronize()

    iters = 20
    t_comm = _time(comm_once, iters)
    t_comp = _time(compute_once, iters)
    t_conc = _time(concurrent_once, iters)

    # reduce max across ranks for a fabric-wide read
    t = torch.tensor([t_comm, t_comp, t_conc], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    t_comm, t_comp, t_conc = t.tolist()

    if rank == 0:
        mx = max(t_comm, t_comp)
        sm = t_comm + t_comp
        print(f"# world={world} ranks, {payload*2//1024}KB/rank all_gather")
        print(f"T_comm      = {t_comm:7.2f} ms")
        print(f"T_compute   = {t_comp:7.2f} ms")
        print(f"T_concurrent= {t_conc:7.2f} ms")
        print(f"max(comm,comp)= {mx:7.2f}   sum= {sm:7.2f}")
        print(f"overlap_ratio (conc/max) = {t_conc/mx:.3f}  "
              f"[1.0=full overlap, {sm/mx:.2f}=full serial]")
        hidden = (sm - t_conc) / min(t_comm, t_comp)
        print(f"fraction of the smaller op HIDDEN = {hidden:.2f}  "
              f"[1.0=fully hidden, 0=not hidden]")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
