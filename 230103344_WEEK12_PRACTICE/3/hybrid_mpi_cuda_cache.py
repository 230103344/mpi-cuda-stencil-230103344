import os
import sys
import json
import math
import time
import hashlib
import platform
import statistics
import subprocess
import warnings
import numpy as np
from mpi4py import MPI
import numba
from numba import cuda, njit, float64, uint64

try:  # silence "grid size too small" warnings for the 1-cell ring launches
    numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0
except Exception:
    pass
warnings.filterwarnings("ignore", category=numba.NumbaPerformanceWarning)

TILE_DIM = 16
SHARED_DIM = TILE_DIM + 2  # 18x18 shared tile
# CPU fallback: no GPU in the runtime -> kernels run in Numba's CUDA simulator
# (NUMBA_ENABLE_CUDASIM=1, set by the launcher cell). The simulator executes every
# CUDA thread in Python, so the grid is reduced to keep the run within minutes.
CPU_FALLBACK = os.environ.get("LAB_CPU_FALLBACK", "0") == "1"
GLOBAL_NX = 32 if CPU_FALLBACK else 512
GLOBAL_NY = 32 if CPU_FALLBACK else 512
EXECUTION_MODE = "cpu_fallback_cuda_simulator" if CPU_FALLBACK else "gpu"
STENCIL_STEPS = 300
SHM_ITERS = 40_000_000
LAMBDA_DAMP = 0.15

PROC_DIMS = [2, 2]
NY_LOC = GLOBAL_NY // PROC_DIMS[0]   # 256
NX_LOC = GLOBAL_NX // PROC_DIMS[1]   # 256
DX = 1.0 / (GLOBAL_NX + 1)           # 1/513
DY = 1.0 / (GLOBAL_NY + 1)           # 1/513
KAPPA_MIN, KAPPA_MAX = 0.08, 0.24
CFL_SAFETY = 0.8                     # dt = 0.8 * dt_max
DIAG_EVERY = 100

SHM_SLOTS = 64                       # 64 uint64 slots = 512 B node-shared window
SHM_STRIDES = (1, 8, 16)
SHM_REPEATS = 3
SHM_ALIGN = 128                      # 128 B alignment (=> 64 B aligned, 128 B pair aligned)

# Halo slot / message-tag convention (tag = direction the message travels)
TAG_UP, TAG_DOWN, TAG_LEFT, TAG_RIGHT = 0, 1, 2, 3
S_TOP, S_BOTTOM, S_LEFT, S_RIGHT = 0, 1, 2, 3


def get_gpu_uuid() -> str:
    if CPU_FALLBACK:
        return "NO-GPU::NUMBA-CUDASIM::" + platform.machine()
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        encoding="utf-8",
    )
    return out.strip().splitlines()[0].strip()


def detect_cpu_line_size() -> int:
    path = "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    return 64


def detect_cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def log(ctx, msg, all_ranks=False):
    if all_ranks or ctx["rank"] == 0:
        print(f"[rank {ctx['rank']} {tuple(ctx['coords'])}] {msg}", flush=True)


# ==============================================================================
# TASK 1: Hybrid MPI Cartesian Grid + Node-Local MPI-3 Shared-Memory Window
# ==============================================================================
def init_hybrid_context(github_user: str) -> dict:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size != 4:
        raise RuntimeError(f"Requires exactly 4 MPI ranks (2x2 grid), got {size}.")

    cuda.select_device(0)
    gpu_uuid = comm.bcast(get_gpu_uuid() if rank == 0 else None, root=0)
    cpu_sig = f"{platform.machine()}::{os.cpu_count()}"
    line_bytes = detect_cpu_line_size()

    digest = hashlib.sha256(f"{github_user.strip()}::{gpu_uuid}::{cpu_sig}".encode("utf-8")).hexdigest()
    instance_seed = int(digest[:16], 16)

    # --- Tier 1: non-periodic 2x2 Cartesian communicator -------------------------
    cart_comm = comm.Create_cart(dims=PROC_DIMS, periods=[False, False], reorder=False)
    cart_rank = cart_comm.Get_rank()
    coords = cart_comm.Get_coords(cart_rank)
    nbr_up, nbr_down = cart_comm.Shift(0, 1)
    nbr_left, nbr_right = cart_comm.Shift(1, 1)

    # --- Tier 4: node-local MPI-3 shared-memory window ---------------------------
    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    node_rank = node_comm.Get_rank()
    nbytes = SHM_SLOTS * 8 + SHM_ALIGN if node_rank == 0 else 0
    win = MPI.Win.Allocate_shared(nbytes, 8, comm=node_comm)
    buf, _itemsize = win.Shared_query(0)
    raw = np.ndarray(buffer=buf, dtype=np.uint8, shape=(SHM_SLOTS * 8 + SHM_ALIGN,))
    off = (-raw.ctypes.data) % SHM_ALIGN
    shm_view = raw[off: off + SHM_SLOTS * 8].view(np.uint64)
    assert shm_view.ctypes.data % 64 == 0
    if node_rank == 0:
        shm_view[:] = 0
    win.Sync()
    node_comm.Barrier()

    # --- CPU pinning (one core per rank when the host has >= 4 cores) ------------
    ncpu = os.cpu_count() or 1
    pinned_core = None
    if hasattr(os, "sched_setaffinity"):
        avail = sorted(os.sched_getaffinity(0))
        if len(avail) >= size:
            pinned_core = avail[node_rank % len(avail)]
            os.sched_setaffinity(0, {pinned_core})

    ctx = dict(
        comm=comm, cart_comm=cart_comm, node_comm=node_comm, win=win, shm_view=shm_view,
        rank=cart_rank, node_rank=node_rank, size=size, coords=list(coords),
        nbr_up=nbr_up, nbr_down=nbr_down, nbr_left=nbr_left, nbr_right=nbr_right,
        gpu_uuid=gpu_uuid, cpu_sig=cpu_sig, cpu_model=detect_cpu_model(), logical_cores=ncpu,
        cpu_line_bytes=line_bytes, instance_seed=instance_seed, github_user=github_user.strip(),
        shm_base_addr=int(shm_view.ctypes.data), pinned_core=pinned_core,
    )

    def nname(n):
        return "PROC_NULL" if n == MPI.PROC_NULL else str(n)

    log(ctx, f"GitHub user = {ctx['github_user']} | GPU UUID = {gpu_uuid}")
    log(ctx, f"CPU = {ctx['cpu_model']} | cpu_sig = {cpu_sig} | CPU line = {line_bytes} B")
    log(ctx, f"INSTANCE_SEED = {instance_seed} (0x{instance_seed:016x})")
    log(ctx, f"node_comm size = {node_comm.Get_size()} | shared window base = "
             f"0x{ctx['shm_base_addr']:x} (mod 64 = {ctx['shm_base_addr'] % 64}, "
             f"mod 128 = {ctx['shm_base_addr'] % 128}) | {SHM_SLOTS} uint64 slots")
    cart_comm.Barrier()
    for r in range(size):
        if r == cart_rank:
            log(ctx, f"up={nname(nbr_up)} down={nname(nbr_down)} left={nname(nbr_left)} "
                     f"right={nname(nbr_right)} | pinned CPU core = {pinned_core}", all_ranks=True)
        cart_comm.Barrier()
    return ctx


# ==============================================================================
# TASK 2: Rank-Local GPU Subdomain & Halo Pack/Unpack Kernels
# ==============================================================================
def initialize_fields(ctx):
    seed = ctx["instance_seed"]
    row_rank, col_rank = ctx["coords"]
    ny, nx = NY_LOC, NX_LOC

    # Every rank builds the SAME RNG from INSTANCE_SEED -> identical global parameters
    rng = np.random.default_rng(seed)
    # kappa(x, y) = 0.16 + 0.08 * sin(...) * cos(...)  in [0.08, 0.24]
    kx, ky = rng.uniform(1.0, 3.0, size=2)
    phx, phy = rng.uniform(0.0, 2.0 * np.pi, size=2)
    # Gaussian thermal pulses
    n_pulses = 3 + int(seed % 3)
    centers = rng.uniform(0.15, 0.85, size=(n_pulses, 2))
    amps = rng.uniform(0.8, 1.6, size=n_pulses)
    sigmas = rng.uniform(0.04, 0.08, size=n_pulses)
    # Localized source term (a cold sink, S <= 0, keeps the run dissipative)
    src_center = rng.uniform(0.3, 0.7, size=2)
    src_amp = rng.uniform(0.3, 0.6)
    src_sigma = 0.05

    # Local index 0..ny+1 -> global padded index -> physical coordinate
    gi = row_rank * ny + np.arange(ny + 2)          # rows  -> y
    gj = col_rank * nx + np.arange(nx + 2)          # cols  -> x
    y = gi * DY
    x = gj * DX
    X, Y = np.meshgrid(x, y)                        # shape (258, 258), local only

    kappa = 0.16 + 0.08 * np.sin(2 * np.pi * kx * X + phx) * np.cos(2 * np.pi * ky * Y + phy)
    kappa = np.clip(kappa, KAPPA_MIN, KAPPA_MAX)

    u = np.zeros((ny + 2, nx + 2), dtype=np.float64)
    for (cx, cy), a, s in zip(centers, amps, sigmas):
        u += a * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * s * s))
    S = -src_amp * np.exp(-((X - src_center[0]) ** 2 + (Y - src_center[1]) ** 2) / (2.0 * src_sigma ** 2))

    # Ghost apron: zero everywhere initially (real halos get filled by the exchange,
    # PROC_NULL sides stay 0.0 forever = homogeneous Dirichlet)
    u[0, :] = 0.0
    u[-1, :] = 0.0
    u[:, 0] = 0.0
    u[:, -1] = 0.0
    S[0, :] = S[-1, :] = 0.0
    S[:, 0] = S[:, -1] = 0.0

    local_kmax = float(kappa[1:ny + 1, 1:nx + 1].max())
    kappa_max = ctx["cart_comm"].allreduce(local_kmax, op=MPI.MAX)
    dt_max = DX * DX / (4.0 * kappa_max)
    dt = CFL_SAFETY * dt_max

    params = dict(
        kx=float(kx), ky=float(ky), n_pulses=n_pulses,
        pulse_centers=centers.tolist(), pulse_amps=amps.tolist(), pulse_sigmas=sigmas.tolist(),
        src_center=src_center.tolist(), src_amp=float(src_amp),
        kappa_max=kappa_max, dt_max=dt_max, dt=dt,
    )
    return u, kappa, S, params



@cuda.jit
def pack_halos_kernel(u, send_buf, ny, nx):
    """Strided gather of the 4 owned border strips into a contiguous (4, n) buffer."""
    k = cuda.grid(1)
    if k < nx:
        send_buf[0, k] = u[1, k + 1]        # top owned row    -> goes UP
        send_buf[1, k] = u[ny, k + 1]       # bottom owned row -> goes DOWN
    if k < ny:
        send_buf[2, k] = u[k + 1, 1]        # left owned column  (stride = 258 doubles)
        send_buf[3, k] = u[k + 1, nx]       # right owned column (stride = 258 doubles)


@cuda.jit
def unpack_halos_kernel(u, recv_buf, ny, nx, has_up, has_down, has_left, has_right):
    """Strided scatter of received vectors into ghost rows/cols 0 and n+1."""
    k = cuda.grid(1)
    if k < nx:
        if has_up:
            u[0, k + 1] = recv_buf[0, k]
        if has_down:
            u[ny + 1, k + 1] = recv_buf[1, k]
    if k < ny:
        if has_left:
            u[k + 1, 0] = recv_buf[2, k]
        if has_right:
            u[k + 1, nx + 1] = recv_buf[3, k]


def post_halo_mpi(ctx, h_send, h_recv):
    """Post tag-matched non-blocking Irecv/Isend for all 4 orthogonal neighbors."""
    cc = ctx["cart_comm"]
    up, down, left, right = ctx["nbr_up"], ctx["nbr_down"], ctx["nbr_left"], ctx["nbr_right"]
    reqs = [
        # receives first (a message travelling DOWN from my up-neighbour fills my top ghost, ...)
        cc.Irecv([h_recv[S_TOP], MPI.DOUBLE], source=up, tag=TAG_DOWN),
        cc.Irecv([h_recv[S_BOTTOM], MPI.DOUBLE], source=down, tag=TAG_UP),
        cc.Irecv([h_recv[S_LEFT], MPI.DOUBLE], source=left, tag=TAG_RIGHT),
        cc.Irecv([h_recv[S_RIGHT], MPI.DOUBLE], source=right, tag=TAG_LEFT),
        # sends
        cc.Isend([h_send[S_TOP], MPI.DOUBLE], dest=up, tag=TAG_UP),
        cc.Isend([h_send[S_BOTTOM], MPI.DOUBLE], dest=down, tag=TAG_DOWN),
        cc.Isend([h_send[S_LEFT], MPI.DOUBLE], dest=left, tag=TAG_LEFT),
        cc.Isend([h_send[S_RIGHT], MPI.DOUBLE], dest=right, tag=TAG_RIGHT),
    ]
    return reqs   # PROC_NULL requests complete immediately (no-ops)


def exchange_halos_async(ctx, st, d_u, stream):
    """Full host-staged halo exchange on one stream (used by the serialized path)."""
    n = st["n"]
    pack_halos_kernel[st["halo_blocks"], st["halo_threads"], stream](d_u, st["d_send"], n, n)
    st["d_send"].copy_to_host(st["h_send"], stream=stream)        # async D2H into pinned memory
    stream.synchronize()
    reqs = post_halo_mpi(ctx, st["h_send"], st["h_recv"])
    MPI.Request.Waitall(reqs)
    st["d_recv"].copy_to_device(st["h_recv"], stream=stream)      # async H2D from pinned memory
    unpack_halos_kernel[st["halo_blocks"], st["halo_threads"], stream](
        d_u, st["d_recv"], n, n, *st["has_nbr"])



# ==============================================================================
# TASK 3: 2D Shared-Memory Tiled Stencil & Dual-Stream Pipeline
# ==============================================================================
@cuda.jit
def tiled_stencil_kernel(u, u_new, kappa, src, dt, inv_dx2, inv_dy2, lam,
                         r_start, r_end, c_start, c_end):
    smem = cuda.shared.array(shape=(SHARED_DIM, SHARED_DIM), dtype=float64)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    i = r_start + cuda.blockIdx.y * TILE_DIM + ty     # row (y) index in padded array
    j = c_start + cuda.blockIdx.x * TILE_DIM + tx     # col (x) index in padded array
    H = u.shape[0]
    W = u.shape[1]

    # ---- load centre + 1-cell apron into shared memory -------------------------------
    if i < H and j < W:
        smem[ty + 1, tx + 1] = u[i, j]
        if ty == 0:
            smem[0, tx + 1] = u[i - 1, j]
        if ty == TILE_DIM - 1 and i + 1 < H:
            smem[TILE_DIM + 1, tx + 1] = u[i + 1, j]
        if tx == 0:
            smem[ty + 1, 0] = u[i, j - 1]
        if tx == TILE_DIM - 1 and j + 1 < W:
            smem[ty + 1, TILE_DIM + 1] = u[i, j + 1]

    cuda.syncthreads()   # every apron write is visible to every warp before any read

    # ---- stencil arithmetic (only inside the requested bounding box) ------------------
    if i <= r_end and j <= c_end:
        c = smem[ty + 1, tx + 1]
        uN = smem[ty, tx + 1]
        uS = smem[ty + 2, tx + 1]
        uW = smem[ty + 1, tx]
        uE = smem[ty + 1, tx + 2]

        kc = kappa[i, j]
        kE = 0.5 * (kappa[i, j + 1] + kc)
        kW = 0.5 * (kappa[i, j - 1] + kc)
        kS = 0.5 * (kappa[i + 1, j] + kc)
        kN = 0.5 * (kappa[i - 1, j] + kc)

        lap_x = (kE * (uE - c) - kW * (c - uW)) * inv_dx2
        lap_y = (kS * (uS - c) - kN * (c - uN)) * inv_dy2
        u_new[i, j] = c + dt * (lap_x + lap_y - lam * c * c * c + src[i, j])


def launch_stencil(st, d_u, d_unew, r0, r1, c0, c1, stream):
    rows = r1 - r0 + 1
    cols = c1 - c0 + 1
    grid = ((cols + TILE_DIM - 1) // TILE_DIM, (rows + TILE_DIM - 1) // TILE_DIM)
    tiled_stencil_kernel[grid, (TILE_DIM, TILE_DIM), stream](
        d_u, d_unew, st["d_kappa"], st["d_src"], st["dt"], st["inv_dx2"], st["inv_dy2"],
        LAMBDA_DAMP, r0, r1, c0, c1)


def step_serialized(ctx, st, d_u, d_unew):
    """Reference path: exchange first, then one full-box stencil, all on one stream."""
    s = st["stream_boundary"]
    n = st["n"]
    exchange_halos_async(ctx, st, d_u, s)
    launch_stencil(st, d_u, d_unew, 1, n, 1, n, s)
    s.synchronize()


def step_overlapped(ctx, st, d_u, d_unew):
    """Dual-stream pipeline: Stage A -> Stage B (MPI || interior) -> Stage C (ring)."""
    sb = st["stream_boundary"]
    si = st["stream_interior"]
    n = st["n"]

    # Stage A: pack border strips + async D2H into pinned buffers (stream_boundary)
    pack_halos_kernel[st["halo_blocks"], st["halo_threads"], sb](d_u, st["d_send"], n, n)
    st["d_send"].copy_to_host(st["h_send"], stream=sb)
    sb.synchronize()

    # Stage B: post non-blocking MPI, immediately launch deep interior on stream_interior
    reqs = post_halo_mpi(ctx, st["h_send"], st["h_recv"])
    launch_stencil(st, d_u, d_unew, 2, n - 1, 2, n - 1, si)

    # Stage C: wait, async H2D, unpack ghosts, update the 1-cell perimeter ring
    MPI.Request.Waitall(reqs)
    st["d_recv"].copy_to_device(st["h_recv"], stream=sb)
    unpack_halos_kernel[st["halo_blocks"], st["halo_threads"], sb](
        d_u, st["d_recv"], n, n, *st["has_nbr"])
    launch_stencil(st, d_u, d_unew, 1, 1, 1, n, sb)          # top row
    launch_stencil(st, d_u, d_unew, n, n, 1, n, sb)          # bottom row
    launch_stencil(st, d_u, d_unew, 2, n - 1, 1, 1, sb)      # left column
    launch_stencil(st, d_u, d_unew, 2, n - 1, n, n, sb)      # right column

    si.synchronize()
    sb.synchronize()



# ==============================================================================
# TASK 4: Multi-Rank Host Cache-Line False Sharing Sweep (MPI-3 shared window)
# ==============================================================================
@njit(nogil=True, cache=False)
def host_shm_accumulate(shm_view, slot, iters, seed):
    """iters in-place RMW updates on shm_view[slot] with NO register hoisting.

    The store index is re-derived each iteration from a value loaded from the
    window (idx = slot + (shm_view[slot] >> 63)). At run time the shift is 0
    (counters never reach 2**63) but LLVM cannot prove it, so the address is
    not loop-invariant and every iteration issues a real load + store.
    """
    s = uint64(seed)
    for i in range(iters):
        v = shm_view[slot]
        idx = slot + (v >> uint64(63))
        shm_view[idx] = shm_view[idx] + (uint64(1) + ((s + uint64(i)) & uint64(3)))
    return shm_view[slot]


@njit(nogil=True, cache=False)
def host_naive_hoistable(shm_view, slot, iters, seed):
    """Reference ONLY for the LLVM audit: LLVM promotes shm_view[slot] to a register."""
    s = uint64(seed)
    for i in range(iters):
        shm_view[slot] = shm_view[slot] + (uint64(1) + ((s + uint64(i)) & uint64(3)))
    return shm_view[slot]


def rank_shm_seed(ctx, r):
    return (ctx["instance_seed"] + 0x9E3779B97F4A7C15 * (r + 1)) & 0xFFFFFFFFFFFFFFFF


def expected_counter(seed: int, iters: int) -> int:
    full, rem = divmod(iters, 4)
    total = iters + full * 6
    for i in range(rem):
        total += (seed + full * 4 + i) & 3
    return total & 0xFFFFFFFFFFFFFFFF


def benchmark_shm_strides(ctx) -> dict:
    nc = ctx["node_comm"]
    view = ctx["shm_view"]
    r = ctx["node_rank"]
    line = ctx["cpu_line_bytes"]
    seed = np.uint64(rank_shm_seed(ctx, r))
    host_shm_accumulate(np.zeros(16, np.uint64), 0, 1000, np.uint64(1))     # JIT warm-up
    host_naive_hoistable(np.zeros(16, np.uint64), 0, 1000, np.uint64(1))
    expected = expected_counter(int(seed), SHM_ITERS)

    results = {}
    if ctx["rank"] == 0:
        print(f"\n[SHM] {nc.Get_size()} ranks x {SHM_ITERS:,} RMW updates in MPI-3 shared window")
        print(f"{'stride':>6s} {'byte offsets':>20s} {'CPU lines':>12s} {'GPU 32B sectors':>16s} "
              f"{'median ns':>16s} {'ns/update':>10s}")
    for stride in SHM_STRIDES:
        slot = r * stride
        runs, ok = [], True
        for _ in range(SHM_REPEATS):
            if r == 0:
                view[:] = 0
            ctx["win"].Sync()
            nc.Barrier()
            t0 = time.perf_counter_ns()
            host_shm_accumulate(view, slot, SHM_ITERS, seed)
            t_local = time.perf_counter_ns() - t0
            nc.Barrier()
            ctx["win"].Sync()
            ok = ok and int(view[slot]) == expected
            runs.append(nc.allreduce(t_local, op=MPI.MAX))     # slowest rank = wall time
        ok_all = nc.allreduce(int(ok), op=MPI.MIN) == 1
        offs = [k * stride * 8 for k in range(nc.Get_size())]
        base = ctx["shm_base_addr"]
        lines = [(base + o) // line - base // line for o in offs]
        sectors = [(base + o) // 32 - base // 32 for o in offs]
        med = int(statistics.median(runs))
        results[f"stride_{stride}"] = dict(
            stride=stride, byte_offsets=offs, cpu_line_index=lines, gpu_sector_index=sectors,
            runs_ns=runs, median_ns=med, min_ns=int(min(runs)),
            ns_per_update=med / SHM_ITERS, counters_exact=ok_all,
        )
        if ctx["rank"] == 0:
            print(f"{stride:6d} {str(offs):>20s} {str(lines):>12s} {str(sectors):>16s} "
                  f"{med:16,d} {med / SHM_ITERS:10.3f}  parity={'OK' if ok_all else 'FAIL'}", flush=True)
    ref = results["stride_8"]["median_ns"]
    for k in results:
        results[k]["slowdown_vs_stride_8"] = results[k]["median_ns"] / ref
    if ctx["rank"] == 0:
        print("[SHM] slowdown vs stride 8: " + ", ".join(
            f"{k}={v['slowdown_vs_stride_8']:.2f}x" for k, v in results.items()), flush=True)
    return results


def _native_llvm_function(llvm: str) -> str:
    out, keep = [], False
    for line in llvm.splitlines():
        if line.startswith("define "):
            name = line.split("@", 1)[1].split("(", 1)[0]
            keep = name.startswith("_ZN") and "cpython" not in name and "NRT" not in name
        if keep:
            out.append(line)
            if line.startswith("}"):
                keep = False
    return "\n".join(out)


def _native_asm_function(asm: str) -> str:
    out, keep = [], False
    for line in asm.splitlines():
        st = line.strip()
        if not line.startswith((" ", "\t")) and st.endswith(":") and st.startswith("_ZN"):
            keep = "cpython" not in st
        if keep:
            if st.startswith((".loc", ".cfi", ".file")):
                continue
            out.append(line)
            if st.startswith((".Lfunc_end", "Lfunc_end")):
                keep = False
    return "\n".join(out)


def _store_in_loop(asm_text: str) -> bool:
    import re
    lines = asm_text.splitlines()
    labels = {l.strip()[:-1]: i for i, l in enumerate(lines) if re.match(r"^\s*\.?L\w+:$", l)}
    for i, l in enumerate(lines):
        m = re.search(r"\b(j\w+|b\.\w+|b|cbnz|cbz|tbnz)\s+(\.?L\w+)", l)
        if m and m.group(2) in labels and labels[m.group(2)] < i:
            for b in lines[labels[m.group(2)]: i + 1]:
                if re.search(r"(mov\w*|add\w*)\s+%?\w+,\s*-?\d*\(", b) or re.search(r"\bstp?\s", b) \
                        or re.search(r"\bstr\s", b):
                    return True
    return False


def llvm_audit(ctx) -> dict:
    def stats(fn):
        llvm_all = "\n".join(fn.inspect_llvm().values())
        asm_all = "\n".join(fn.inspect_asm().values())
        ir = _native_llvm_function(llvm_all)
        asm = _native_asm_function(asm_all)
        return dict(
            store_i64=ir.count("store i64"),
            load_i64=ir.count("load i64"),
            pyobject_refs_in_native=sum(ir.count(k) for k in ("PyObject", "@Py", "pyobject")),
            releases_gil_in_wrapper=("PyEval_SaveThread" in llvm_all and "PyEval_RestoreThread" in llvm_all),
            store_inside_loop_asm=_store_in_loop(asm),
            nopython_signatures=[str(s) for s in fn.nopython_signatures],
        ), ir, asm
    s, ir, asm = stats(host_shm_accumulate)
    n, _, _ = stats(host_naive_hoistable)
    passed = bool(s["store_i64"] > 0 and s["pyobject_refs_in_native"] == 0 and
                  s["store_inside_loop_asm"] and s["releases_gil_in_wrapper"])
    if ctx["rank"] == 0:
        open("host_shm_accumulate.ll", "w").write(ir)
        open("host_shm_accumulate.s", "w").write(asm)
        log(ctx, f"LLVM audit host_shm_accumulate: store i64 = {s['store_i64']}, load i64 = {s['load_i64']}, "
                 f"PyObject refs = {s['pyobject_refs_in_native']}, store inside loop = "
                 f"{s['store_inside_loop_asm']}, GIL released = {s['releases_gil_in_wrapper']} -> "
                 f"{'PASS' if passed else 'FAIL'}")
        log(ctx, f"LLVM audit host_naive_hoistable (control): store i64 = {n['store_i64']}, "
                 f"store inside loop = {n['store_inside_loop_asm']} (register-promoted)")
    return dict(host_shm_accumulate=s, host_naive_hoistable=n, llvm_audit_passed=passed)


# TASK 5: Cross-Tier Verification & Manifest Export
# ==============================================================================
def build_state(ctx, u0, kappa, S, params):
    n = NY_LOC
    st = dict(n=n)
    st["d_kappa"] = cuda.to_device(kappa)
    st["d_src"] = cuda.to_device(S)
    st["dt"] = params["dt"]
    st["inv_dx2"] = 1.0 / (DX * DX)
    st["inv_dy2"] = 1.0 / (DY * DY)
    st["d_send"] = cuda.device_array((4, n), dtype=np.float64)
    st["d_recv"] = cuda.device_array((4, n), dtype=np.float64)
    st["h_send"] = cuda.pinned_array((4, n), dtype=np.float64)
    st["h_recv"] = cuda.pinned_array((4, n), dtype=np.float64)
    st["h_send"][:] = 0.0
    st["h_recv"][:] = 0.0
    st["halo_threads"] = 128
    st["halo_blocks"] = (n + 127) // 128
    st["has_nbr"] = tuple(int(x != MPI.PROC_NULL) for x in
                          (ctx["nbr_up"], ctx["nbr_down"], ctx["nbr_left"], ctx["nbr_right"]))
    st["stream_boundary"] = cuda.stream()
    st["stream_interior"] = cuda.stream()
    return st


def global_diagnostics(ctx, u_host, u_prev_host):
    n = NY_LOC
    core = u_host[1:n + 1, 1:n + 1]
    loc = np.array([core.sum() * DX * DY,
                    0.0 if u_prev_host is None else float(((core - u_prev_host[1:n + 1, 1:n + 1]) ** 2).sum()),
                    float(np.abs(core).max())], dtype=np.float64)
    glob = np.zeros(2, dtype=np.float64)
    ctx["cart_comm"].Allreduce(loc[:2], glob, op=MPI.SUM)
    gmax = np.zeros(1, dtype=np.float64)
    ctx["cart_comm"].Allreduce(loc[2:3], gmax, op=MPI.MAX)
    return float(glob[0]), float(math.sqrt(glob[1])), float(gmax[0])


def event_ms(ev_a, ev_b):
    """CUDA Event elapsed time in ms (0.0 under the CUDA simulator)."""
    try:
        return float(cuda.event_elapsed_time(ev_a, ev_b))
    except AttributeError:
        return 0.0


def run_mode(ctx, mode, u0, st):
    step_fn = step_serialized if mode == "serialized" else step_overlapped
    d_u = cuda.to_device(u0)
    d_unew = cuda.to_device(u0)          # ghosts of both buffers start at 0.0
    trace = []

    E, R, M = global_diagnostics(ctx, u0, None)
    trace.append(dict(step=0, energy=E, l2_residual=R, max_abs_u=M))
    log(ctx, f"[{mode:>10}] step {0:4d} | E = {E:.15e} | L2 res = {R:.6e} | max|u| = {M:.6f}")

    ctx["cart_comm"].Barrier()
    cuda.synchronize()
    ev_start, ev_end = cuda.event(), cuda.event()   # timing=True by default
    ev_start.record()
    t0 = MPI.Wtime()
    gpu_ms = 0.0
    for step in range(1, STENCIL_STEPS + 1):
        step_fn(ctx, st, d_u, d_unew)
        d_u, d_unew = d_unew, d_u
        if step % DIAG_EVERY == 0:
            ev_end.record()
            ev_end.synchronize()
            gpu_ms += event_ms(ev_start, ev_end)
            u_h = d_u.copy_to_host()
            u_prev = d_unew.copy_to_host()
            E, R, M = global_diagnostics(ctx, u_h, u_prev)
            trace.append(dict(step=step, energy=E, l2_residual=R, max_abs_u=M))
            log(ctx, f"[{mode:>10}] step {step:4d} | E = {E:.15e} | L2 res = {R:.6e} | max|u| = {M:.6f}")
            ev_start.record()
    cuda.synchronize()
    wall_ms = (MPI.Wtime() - t0) * 1e3
    wall_ms = ctx["cart_comm"].allreduce(wall_ms, op=MPI.MAX)
    gpu_ms = ctx["cart_comm"].allreduce(gpu_ms, op=MPI.MAX)
    return d_u.copy_to_host(), trace, wall_ms, gpu_ms



_PTX_HELPER = r"""
import sys, json
import numpy as _np
if not hasattr(_np, "row_stack"):
    _np.row_stack = _np.vstack
from numba import cuda, float64, types
TILE_DIM = 16
SHARED_DIM = TILE_DIM + 2
src = open(sys.argv[1]).read()
exec(compile(src, "tiled_stencil_kernel", "exec"))
A = types.Array(types.float64, 2, "C")
sig = (A, A, A, A, types.float64, types.float64, types.float64, types.float64,
       types.int64, types.int64, types.int64, types.int64)
ptx, _ = cuda.compile_ptx(tiled_stencil_kernel.py_func, sig, cc=(7, 5))
open(sys.argv[2], "w").write(ptx)
"""


def offline_ptx_text():
    """Compile tiled_stencil_kernel to PTX for sm_75 (Tesla T4) without a GPU (needs libNVVM)."""
    import inspect, tempfile
    src = inspect.getsource(tiled_stencil_kernel.py_func if hasattr(tiled_stencil_kernel, "py_func")
                            else tiled_stencil_kernel.fn)
    src = "from numba import cuda, float64\n" + src
    with tempfile.TemporaryDirectory() as d:
        ks, hs, out = os.path.join(d, "k.py"), os.path.join(d, "h.py"), os.path.join(d, "k.ptx")
        open(ks, "w").write(src)
        open(hs, "w").write(_PTX_HELPER)
        env = {k: v for k, v in os.environ.items()
               if k != "NUMBA_ENABLE_CUDASIM" and not k.startswith(("OMPI_", "PMIX_", "OPAL_"))}
        subprocess.run([sys.executable, hs, ks, out], env=env, check=True,
                       capture_output=True, text=True, timeout=600)
        return open(out).read()


def ptx_audit(ctx):
    source = "inspect_asm"
    try:
        asm = tiled_stencil_kernel.inspect_asm()
        text = "\n".join(asm.values()) if isinstance(asm, dict) else str(asm)
    except Exception as e:
        try:
            text = offline_ptx_text()        # same kernel compiled for sm_75 via libNVVM
            source = "numba.cuda.compile_ptx(cc=7.5) [no GPU in runtime]"
        except Exception as e2:
            log(ctx, f"PTX audit unavailable: {e} | offline: {e2}")
            return dict(st_shared_f64=0, ld_shared_f64=0, bar_sync=0, ptx_audit_passed=False,
                        ptx_source="unavailable")
    counts = dict(
        st_shared_f64=text.count("st.shared.f64"),
        ld_shared_f64=text.count("ld.shared.f64"),
        bar_sync=text.count("bar.sync") + text.count("barrier.sync"),
        st_shared_total=text.count("st.shared"),
        ld_shared_total=text.count("ld.shared"),
    )
    counts["ptx_source"] = source
    counts["ptx_audit_passed"] = bool(counts["st_shared_total"] > 0 and
                                      counts["ld_shared_total"] > 0 and
                                      counts["bar_sync"] > 0)
    if ctx["rank"] == 0:
        with open("tiled_stencil_kernel.ptx", "w") as f:
            f.write(text)
    return counts


def warmup(ctx, u0, st):
    """JIT-compile all kernels once so compilation is excluded from timings."""
    d_a = cuda.to_device(u0)
    d_b = cuda.to_device(u0)
    step_serialized(ctx, st, d_a, d_b)
    step_overlapped(ctx, st, d_a, d_b)
    cuda.synchronize()
    ctx["cart_comm"].Barrier()



def main(github_user: str):
    ctx = init_hybrid_context(github_user)
    cc = ctx["cart_comm"]

    # ---------------- Tier 1-3: GPU stencil -------------------------------------
    u0, kappa, S, params = initialize_fields(ctx)
    log(ctx, f"local padded array shape = {u0.shape} (owned {NY_LOC}x{NX_LOC}), "
             f"global grid {GLOBAL_NY}x{GLOBAL_NX}, dx = dy = 1/{GLOBAL_NX + 1}")
    log(ctx, f"kappa_max = {params['kappa_max']:.10f} | dt_max = dx^2/(4 kappa_max) = "
             f"{params['dt_max']:.6e} | dt = {CFL_SAFETY} * dt_max = {params['dt']:.6e} | "
             f"CFL satisfied = {params['dt'] <= params['dt_max']}")

    st = build_state(ctx, u0, kappa, S, params)
    warmup(ctx, u0, st)
    u_ser, trace_ser, ser_ms, ser_gpu = run_mode(ctx, "serialized", u0, st)
    u_ovl, trace_ovl, ovl_ms, ovl_gpu = run_mode(ctx, "overlapped", u0, st)

    loc_diff = float(np.abs(u_ser - u_ovl).max())
    parity = cc.allreduce(loc_diff, op=MPI.MAX)
    parity_ok = parity < 1e-12
    energies = [t["energy"] for t in trace_ovl]
    energy_monotone = all(b <= a + 1e-15 for a, b in zip(energies, energies[1:]))
    energy_mode_diff = max(abs(a["energy"] - b["energy"]) for a, b in zip(trace_ser, trace_ovl))
    log(ctx, f"gpu_serialized_ms = {ser_ms:.3f} | gpu_overlapped_ms = {ovl_ms:.3f} | "
             f"speedup = {ser_ms / ovl_ms:.3f}x (CUDA-event ms: {ser_gpu:.3f} vs {ovl_gpu:.3f})")
    log(ctx, f"parity_max_diff = {parity:.3e} -> {'PASS' if parity_ok else 'FAIL'} (< 1e-12) | "
             f"energy monotone non-increasing = {energy_monotone} | "
             f"|E_ser - E_ovl| max = {energy_mode_diff:.3e}")

    ptx = ptx_audit(ctx)
    log(ctx, f"PTX audit: st.shared.f64 = {ptx.get('st_shared_f64')} | ld.shared.f64 = "
             f"{ptx.get('ld_shared_f64')} | bar.sync = {ptx.get('bar_sync')} -> "
             f"{'PASS' if ptx['ptx_audit_passed'] else 'FAIL'}")

    # ---------------- Tier 4: MPI-3 shared-memory false sharing ------------------
    cuda.synchronize()
    cc.Barrier()
    shm = benchmark_shm_strides(ctx)
    llvm = llvm_audit(ctx)
    shm_ok = all(v["counters_exact"] for v in shm.values())

    verification_passed = bool(parity_ok and ptx["ptx_audit_passed"] and llvm["llvm_audit_passed"]
                               and shm_ok and params["dt"] <= params["dt_max"])

    if ctx["rank"] == 0:
        topo = []
        for r in range(ctx["size"]):
            crd = cc.Get_coords(r)
            nb = {}
            for name, axis, disp in (("up", 0, -1), ("down", 0, 1), ("left", 1, -1), ("right", 1, 1)):
                c2 = list(crd)
                c2[axis] += disp
                nb[name] = cc.Get_cart_rank(c2) if 0 <= c2[axis] < PROC_DIMS[axis] else "PROC_NULL"
            topo.append(dict(rank=r, coords=list(crd), neighbors=nb))
        try:
            gpu_name = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
                encoding="utf-8").strip()
        except Exception:
            gpu_name = "unknown"

        manifest = dict(
            GITHUB_USERNAME=ctx["github_user"],
            gpu_uuid=ctx["gpu_uuid"],
            gpu=gpu_name,
            cpu_model=ctx["cpu_model"],
            cpu_signature=ctx["cpu_sig"],
            logical_cores=ctx["logical_cores"],
            cpu_line_bytes=ctx["cpu_line_bytes"],
            instance_seed=str(ctx["instance_seed"]),
            instance_seed_hex=f"0x{ctx['instance_seed']:016x}",
            mpi_ranks=ctx["size"],
            cart_dims=PROC_DIMS,
            periods=[False, False],
            topology=topo,
            global_grid=[GLOBAL_NY, GLOBAL_NX],
            local_padded_shape=[NY_LOC + 2, NX_LOC + 2],
            tile=dict(block=[TILE_DIM, TILE_DIM], shared_tile=[SHARED_DIM, SHARED_DIM]),
            stencil_steps=STENCIL_STEPS,
            lambda_damp=LAMBDA_DAMP,
            dx=DX, dy=DY,
            kappa_max=params["kappa_max"],
            dt_max=params["dt_max"],
            dt=params["dt"],
            cfl_satisfied=params["dt"] <= params["dt_max"],
            init_params=params,
            energy_trace_serialized=trace_ser,
            energy_trace_overlapped=trace_ovl,
            energy_monotone_non_increasing=energy_monotone,
            gpu_serialized_ms=ser_ms,
            gpu_overlapped_ms=ovl_ms,
            gpu_serialized_cuda_event_ms=ser_gpu,
            gpu_overlapped_cuda_event_ms=ovl_gpu,
            overlap_speedup=ser_ms / ovl_ms,
            parity_max_diff=parity,
            parity_passed=bool(parity_ok),
            ptx=ptx,
            ptx_audit_passed=ptx["ptx_audit_passed"],
            shm_window=dict(slots=SHM_SLOTS, base_addr_mod_64=ctx["shm_base_addr"] % 64,
                            base_addr_mod_128=ctx["shm_base_addr"] % 128,
                            node_comm_size=ctx["node_comm"].Get_size(), iters_per_rank=SHM_ITERS,
                            repeats=SHM_REPEATS),
            shm_results=shm,
            shm_false_sharing_slowdown=shm["stride_1"]["slowdown_vs_stride_8"],
            shm_stride16_vs_stride8=shm["stride_16"]["slowdown_vs_stride_8"],
            shm_parity_passed=shm_ok,
            llvm_audit=llvm,
            llvm_audit_passed=llvm["llvm_audit_passed"],
            forbidden_collectives_on_fields=False,
            versions=dict(python=platform.python_version(), numpy=np.__version__,
                          numba=numba.__version__, mpi=MPI.Get_library_version().splitlines()[0]),
            timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            execution_mode=EXECUTION_MODE,
            tasks_completed=[1, 2, 3, 4, 5],
            verification_passed=verification_passed,
        )
        with open("hybrid_lab_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        log(ctx, f"hybrid_lab_manifest.json written | verification_passed = {verification_passed}")
    cc.Barrier()
    ctx["win"].Free()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: mpirun -np 4 python3 hybrid_mpi_cuda_cache.py <github_username>")
    main(sys.argv[1])
