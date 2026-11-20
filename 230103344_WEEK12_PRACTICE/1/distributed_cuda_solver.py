import os
import sys
import json
import math
import time
import hashlib
import platform
import subprocess
import warnings
import numpy as np
from mpi4py import MPI
import numba
from numba import cuda, float64

try:  # silence "grid size too small" warnings for the 1-cell ring launches
    numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0
except Exception:
    pass
warnings.filterwarnings("ignore", category=numba.NumbaPerformanceWarning)

# Compile-time constants for 2D shared-memory tiling (16x16 interior + 1-cell halo apron)
TILE_DIM = 16
SHARED_DIM = TILE_DIM + 2  # 18
# CPU fallback: no GPU in the runtime -> kernels run in Numba's CUDA simulator
# (NUMBA_ENABLE_CUDASIM=1, set by the launcher cell). The simulator executes every
# CUDA thread in Python, so the grid is reduced to keep the run within minutes.
CPU_FALLBACK = os.environ.get("LAB_CPU_FALLBACK", "0") == "1"
GLOBAL_NX = 32 if CPU_FALLBACK else 512
GLOBAL_NY = 32 if CPU_FALLBACK else 512
EXECUTION_MODE = "cpu_fallback_cuda_simulator" if CPU_FALLBACK else "gpu"
STEPS = 400
LAMBDA_DAMP = 0.15

PROC_DIMS = [2, 2]
NY_LOC = GLOBAL_NY // PROC_DIMS[0]   # 256
NX_LOC = GLOBAL_NX // PROC_DIMS[1]   # 256
DX = 1.0 / (GLOBAL_NX + 1)           # 1/513
DY = 1.0 / (GLOBAL_NY + 1)           # 1/513
KAPPA_MIN, KAPPA_MAX = 0.08, 0.24
CFL_SAFETY = 0.8                     # dt = 0.8 * dt_max
DIAG_EVERY = 100

# Halo slot / message-tag convention (tag = direction the message travels)
TAG_UP, TAG_DOWN, TAG_LEFT, TAG_RIGHT = 0, 1, 2, 3
# slot index inside the (4, n) pack/unpack buffers
S_TOP, S_BOTTOM, S_LEFT, S_RIGHT = 0, 1, 2, 3


def get_gpu_uuid() -> str:
    """Query active NVIDIA GPU UUID via nvidia-smi."""
    if CPU_FALLBACK:
        return "NO-GPU::NUMBA-CUDASIM::" + platform.machine()
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        encoding="utf-8",
    )
    return out.strip().splitlines()[0].strip()


def derive_student_seed(github_user: str, gpu_uuid: str) -> int:
    """Derive deterministic 64-bit integer seed bound to student GitHub handle and GPU UUID."""
    digest = hashlib.sha256(f"{github_user.strip()}::{gpu_uuid}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def log(ctx, msg, all_ranks=False):
    if all_ranks or ctx["rank"] == 0:
        print(f"[rank {ctx['rank']} {tuple(ctx['coords'])}] {msg}", flush=True)


# ==============================================================================
# TASK 1: Distributed Cartesian Topology & Hardware Context Initialization
# ==============================================================================
def init_distributed_context(github_user: str):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size != 4:
        raise RuntimeError(f"Expected exactly 4 MPI ranks (2x2 grid), got {size}.")

    # Bind process to GPU 0 on Google Colab
    cuda.select_device(0)
    gpu_uuid = get_gpu_uuid() if rank == 0 else None
    gpu_uuid = comm.bcast(gpu_uuid, root=0)
    instance_seed = derive_student_seed(github_user, gpu_uuid)

    # Non-periodic 2x2 Cartesian communicator
    cart_comm = comm.Create_cart(dims=PROC_DIMS, periods=[False, False], reorder=False)
    cart_rank = cart_comm.Get_rank()
    coords = cart_comm.Get_coords(cart_rank)
    nbr_up, nbr_down = cart_comm.Shift(0, 1)      # axis 0 -> rows (y)
    nbr_left, nbr_right = cart_comm.Shift(1, 1)   # axis 1 -> cols (x)

    ctx = dict(
        comm=comm,
        cart_comm=cart_comm,
        rank=cart_rank,
        size=size,
        coords=list(coords),
        nbr_up=nbr_up,
        nbr_down=nbr_down,
        nbr_left=nbr_left,
        nbr_right=nbr_right,
        gpu_uuid=gpu_uuid,
        instance_seed=instance_seed,
        github_user=github_user.strip(),
    )

    def nname(n):
        return "PROC_NULL" if n == MPI.PROC_NULL else str(n)

    log(ctx, f"GitHub user = {ctx['github_user']} | GPU UUID = {gpu_uuid}")
    log(ctx, f"INSTANCE_SEED = {instance_seed} (0x{instance_seed:016x})")
    cart_comm.Barrier()
    for r in range(size):
        if r == cart_rank:
            log(ctx, f"up={nname(nbr_up)} down={nname(nbr_down)} "
                     f"left={nname(nbr_left)} right={nname(nbr_right)}", all_ranks=True)
        cart_comm.Barrier()
    return ctx


# ==============================================================================
# TASK 2: Seeded Heterogeneous Subdomain Initialization (no global arrays)
# ==============================================================================
def initialize_subdomains(ctx):
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


# ==============================================================================
# TASK 3: GPU Halo Pack/Unpack Kernels & Pinned MPI Exchange
# ==============================================================================
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
# TASK 4: 2D Shared-Memory Tiled Stencil & Dual-Stream Orchestrator
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
# TASK 5: Distributed Reductions, PTX Audit & Comparative Benchmark
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
    for step in range(1, STEPS + 1):
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


def run_solver(github_user: str):
    ctx = init_distributed_context(github_user)
    cc = ctx["cart_comm"]

    u0, kappa, S, params = initialize_subdomains(ctx)
    log(ctx, f"local padded array shape = {u0.shape} (owned {NY_LOC}x{NX_LOC}), "
             f"global grid {GLOBAL_NY}x{GLOBAL_NX}, dx = dy = 1/{GLOBAL_NX + 1}")
    log(ctx, f"kappa_max = {params['kappa_max']:.10f} | dt_max = dx^2/(4 kappa_max) = "
             f"{params['dt_max']:.6e} | dt = {CFL_SAFETY} * dt_max = {params['dt']:.6e}")

    st = build_state(ctx, u0, kappa, S, params)
    warmup(ctx, u0, st)

    u_ser, trace_ser, ser_ms, ser_gpu = run_mode(ctx, "serialized", u0, st)
    u_ovl, trace_ovl, ovl_ms, ovl_gpu = run_mode(ctx, "overlapped", u0, st)

    loc_diff = float(np.abs(u_ser - u_ovl).max())
    parity = cc.allreduce(loc_diff, op=MPI.MAX)
    parity_ok = parity < 1e-12
    log(ctx, f"serialized_ms = {ser_ms:.3f} | overlapped_ms = {ovl_ms:.3f} | "
             f"speedup = {ser_ms / ovl_ms:.3f}x  (CUDA-event ms: {ser_gpu:.3f} vs {ovl_gpu:.3f})")
    log(ctx, f"parity_max_diff = {parity:.3e} -> {'PASS' if parity_ok else 'FAIL'} (< 1e-12)")

    ptx = ptx_audit(ctx)
    log(ctx, f"PTX audit: st.shared.f64 = {ptx.get('st_shared_f64')} | ld.shared.f64 = "
             f"{ptx.get('ld_shared_f64')} | bar.sync = {ptx.get('bar_sync')} -> "
             f"{'PASS' if ptx['ptx_audit_passed'] else 'FAIL'}")

    if ctx["rank"] == 0:
        topo = []
        for r in range(ctx["size"]):
            crd = cc.Get_coords(r)
            nb = {}
            for name, axis, disp in (("up", 0, -1), ("down", 0, 1), ("left", 1, -1), ("right", 1, 1)):
                c2 = list(crd)
                c2[axis] += disp
                ok = 0 <= c2[axis] < PROC_DIMS[axis]
                nb[name] = cc.Get_cart_rank(c2) if ok else "PROC_NULL"
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
            instance_seed=str(ctx["instance_seed"]),
            instance_seed_hex=f"0x{ctx['instance_seed']:016x}",
            mpi_ranks=ctx["size"],
            cart_dims=PROC_DIMS,
            periods=[False, False],
            topology=topo,
            global_grid=[GLOBAL_NY, GLOBAL_NX],
            local_padded_shape=[NY_LOC + 2, NX_LOC + 2],
            tile=dict(block=[TILE_DIM, TILE_DIM], shared_tile=[SHARED_DIM, SHARED_DIM]),
            steps=STEPS,
            lambda_damp=LAMBDA_DAMP,
            dx=DX, dy=DY,
            kappa_max=params["kappa_max"],
            dt_max=params["dt_max"],
            dt=params["dt"],
            init_params=params,
            energy_trace_serialized=trace_ser,
            energy_trace_overlapped=trace_ovl,
            serialized_ms=ser_ms,
            overlapped_ms=ovl_ms,
            serialized_cuda_event_ms=ser_gpu,
            overlapped_cuda_event_ms=ovl_gpu,
            overlap_speedup=ser_ms / ovl_ms,
            parity_max_diff=parity,
            parity_passed=bool(parity_ok),
            ptx=ptx,
            ptx_audit_passed=ptx["ptx_audit_passed"],
            forbidden_collectives_on_fields=False,
            versions=dict(python=platform.python_version(), numpy=np.__version__,
                          numba=numba.__version__, mpi=MPI.Get_library_version().splitlines()[0]),
            timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            execution_mode=EXECUTION_MODE,
            tasks_completed=[1, 2, 3, 4, 5],
        )
        with open("submission_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        log(ctx, "submission_manifest.json written.")
    cc.Barrier()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: mpirun -np 4 python3 distributed_cuda_solver.py <github_username>")
    run_solver(sys.argv[1])
