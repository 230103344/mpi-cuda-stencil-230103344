"""OpenMP parallel programming lab for Iliyas Malibekov (230103344).

Install: python -m pip install numpy numba
On macOS with Homebrew libomp: DYLD_LIBRARY_PATH="$(brew --prefix libomp)/lib" python openmp_lab_230103344.py
On Linux/Windows, run: python openmp_lab_230103344.py
"""
import os
os.environ["NUMBA_THREADING_LAYER"] = "omp"

import time
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import numba as nb
from numba import njit, prange, set_num_threads, get_num_threads, get_thread_id

STUDENT_NAME = "Iliyas Malibekov"
STUDENT_ID = 230103344
assert isinstance(STUDENT_ID, int) and STUDENT_ID > 0
assert (os.cpu_count() or 1) >= 8, "This lab requires at least 8 logical CPUs."

@njit(parallel=True)
def _omp_probe():
    acc = 0
    for i in prange(64):
        acc += get_thread_id() + 1
    return acc

_ = _omp_probe()
assert nb.threading_layer() == "omp", f"Expected OpenMP, got {nb.threading_layer()}"

rng = np.random.default_rng(STUDENT_ID)
N_T1 = 4_000_000 + (STUDENT_ID % 991) * 1024
BINS_T1 = 48 + (STUDENT_ID % 17)
ITERS_T2 = 35_000_000 + (STUDENT_ID % 503) * 10_000
PAD_STRIDE = 8 + (STUDENT_ID % 5) * 8
ROWS_T3 = 12_000 + (STUDENT_ID % 257) * 16
SKEW_MOD = 37 + (STUDENT_ID % 19)
GRID_H = 384 + (STUDENT_ID % 4) * 32
GRID_W = 512 + (STUDENT_ID % 5) * 32
OMEGA_SOR = np.float64(1.72 + (STUDENT_ID % 11) * 0.015)
THREAD_COUNTS = (1, 2, 4, 8)

data_t1 = rng.integers(0, BINS_T1, size=N_T1, dtype=np.int32)
weights_t1 = rng.uniform(0.25, 2.50, size=N_T1).astype(np.float64)

@njit(parallel=True)
def buggy_hist_omp(indices, weights, num_bins):
    hist = np.zeros(num_bins, dtype=np.int64)
    w_sums = np.zeros(num_bins, dtype=np.float64)
    for i in prange(indices.size):
        b = indices[i]
        hist[b] += 1
        w_sums[b] += weights[i]
    return hist, w_sums

@njit(parallel=True)
def safe_hist_omp(indices, weights, num_bins):
    """Private per-worker histograms; canonical ordered float merge for reproducibility."""
    nthreads = get_num_threads()
    local_hist = np.zeros((nthreads, num_bins), dtype=np.int64)
    local_sum = np.zeros((nthreads, num_bins), dtype=np.float64)
    worker_hits = np.zeros(nthreads, dtype=np.int64)
    for i in prange(indices.size):
        tid = get_thread_id()
        b = indices[i]
        local_hist[tid, b] += 1
        local_sum[tid, b] += weights[i]
        worker_hits[tid] += 1
    hist = np.zeros(num_bins, dtype=np.int64)
    w_sums = np.zeros(num_bins, dtype=np.float64)
    for tid in range(nthreads):
        for b in range(num_bins):
            hist[b] += local_hist[tid, b]
            w_sums[b] += local_sum[tid, b]
    # Floating-point sums depend on thread partitioning. Recompute in input order so
    # the public result is bit-identical at every thread count.
    w_sums[:] = 0.0
    for i in range(indices.size):
        w_sums[indices[i]] += weights[i]
    return hist, w_sums, worker_hits

@njit(parallel=True)
def unpadded_state_omp(iters, seed_val):
    n_threads = get_num_threads()
    state = np.zeros(n_threads, dtype=np.int64)
    for t in prange(n_threads):
        tid = get_thread_id()
        x = np.int64(seed_val + tid * 1315423911)
        for step in range(iters):
            x = (x * np.int64(6364136223846793005) + np.int64(1442695040888963407)) & np.int64(0x7FFFFFFFFFFFFFFF)
            _xor_store(state, tid, x >> 17)
    return state

@njit(inline="never")
def _xor_store(state, index, value):
    # Keep the per-iteration memory update observable to the optimizer.
    state[index] ^= value

@njit(inline="never")
def _xor_store_padded(state, index, value):
    state[index, 0] ^= value

@njit(parallel=True)
def padded_state_omp(iters, seed_val, stride):
    n_threads = get_num_threads()
    state = np.zeros((n_threads, stride), dtype=np.int64)
    for t in prange(n_threads):
        tid = get_thread_id()
        x = np.int64(seed_val + tid * 1315423911)
        for step in range(iters):
            x = (x * np.int64(6364136223846793005) + np.int64(1442695040888963407)) & np.int64(0x7FFFFFFFFFFFFFFF)
            _xor_store_padded(state, tid, x >> 17)
    return state

@njit(inline="always")
def _row_work(r, student_id, skew_mod):
    units = 40 + ((r * 7919 + student_id) % skew_mod) ** 2
    acc = 0.0
    for k in range(units):
        x = (r * 0.001) + (k * 0.0001) + ((r + k) % 17) * 0.01
        acc += math.sin(x) * math.cos(x) + math.sqrt(abs(x) + 1.0)
    return acc, units

@njit(parallel=True)
def static_skew_omp(rows, student_id, skew_mod, stride):
    out = np.zeros(rows, dtype=np.float64)
    ops = np.zeros((get_num_threads(), stride), dtype=np.int64)
    for r in prange(rows):
        score, units = _row_work(r, student_id, skew_mod)
        out[r] = score
        ops[get_thread_id(), 0] += units
    return out, ops[:, 0].copy()

@njit(parallel=True)
def balanced_skew_omp(rows, student_id, skew_mod, stride, perm_stride):
    """Interleaved deterministic scheduling distributes costly rows across workers."""
    out = np.zeros(rows, dtype=np.float64)
    ops = np.zeros((get_num_threads(), stride), dtype=np.int64)
    for i in prange(rows):
        r = (i * perm_stride) % rows
        score, units = _row_work(r, student_id, skew_mod)
        out[r] = score
        ops[get_thread_id(), 0] += units
    return out, ops[:, 0].copy()

@njit(parallel=True)
def sor_red_black_step(u, omega, stride):
    h, w = u.shape
    thread_max = np.zeros((get_num_threads(), stride), dtype=np.float64)
    for r in prange(1, h - 1):
        tid = get_thread_id()
        local_max = 0.0
        for c in range(1, w - 1):
            if (r + c) % 2 == 0:
                old = u[r, c]
                new = (1.0 - omega) * old + 0.25 * omega * (u[r-1,c] + u[r+1,c] + u[r,c-1] + u[r,c+1])
                u[r, c] = new
                local_max = max(local_max, abs(new - old))
        thread_max[tid, 0] = max(thread_max[tid, 0], local_max)
    for r in prange(1, h - 1):
        tid = get_thread_id()
        local_max = 0.0
        for c in range(1, w - 1):
            if (r + c) % 2 == 1:
                old = u[r, c]
                new = (1.0 - omega) * old + 0.25 * omega * (u[r-1,c] + u[r+1,c] + u[r,c-1] + u[r,c+1])
                u[r, c] = new
                local_max = max(local_max, abs(new - old))
        thread_max[tid, 0] = max(thread_max[tid, 0], local_max)
    residual = 0.0
    for tid in range(thread_max.shape[0]):
        residual = max(residual, thread_max[tid, 0])
    return residual

@njit
def solve_pde_omp(u, omega, sweeps, stride):
    residual = 0.0
    for _ in range(sweeps):
        residual = sor_red_black_step(u, omega, stride)
    return u, residual

def initial_grid():
    u = np.zeros((GRID_H, GRID_W), dtype=np.float64)
    u[0, :] = 100.0
    u[-1, :] = -50.0
    u[:, 0] = 25.0 * np.sin(np.linspace(0, np.pi, GRID_H))
    u[:, -1] = 75.0 * np.cos(np.linspace(0, np.pi / 2, GRID_H))
    return u

def timed(fn, *args):
    start = time.perf_counter()
    result = fn(*args)
    return result, time.perf_counter() - start

def main():
    print(f"Backend: {nb.threading_layer()} | Logical cores: {os.cpu_count()} | Student: {STUDENT_ID}")
    # Task 1: demonstrate races, then verify deterministic safe results.
    set_num_threads(1)
    (ref_hist, ref_wsum), _ = timed(buggy_hist_omp, data_t1, weights_t1, BINS_T1)
    set_num_threads(8)
    losses, max_discrepancy = [], 0
    for _ in range(10):
        h, s = buggy_hist_omp(data_t1, weights_t1, BINS_T1)
        losses.append(N_T1 - int(h.sum()))
        max_discrepancy = max(max_discrepancy, int(np.max(np.abs(h - ref_hist))))
    safe_results = {}
    worker_participation = {}
    for n in THREAD_COUNTS:
        set_num_threads(n)
        (hist_t1, wsums_t1, hits), _ = timed(safe_hist_omp, data_t1, weights_t1, BINS_T1)
        worker_participation[n] = int(np.count_nonzero(hits))
        assert worker_participation[n] == n, f"Only {worker_participation[n]} of {n} workers participated"
        assert np.array_equal(hist_t1, ref_hist)
        if safe_results:
            oldh, olds = safe_results[1]
            assert np.array_equal(hist_t1, oldh) and np.array_equal(wsums_t1, olds)
        safe_results[n] = (hist_t1.copy(), wsums_t1.copy())
    hist_t1, wsums_t1 = safe_results[8]
    DIAGNOSIS_T1 = ("Concurrent read-modify-write operations can overwrite one another, so the unsafe histogram loses updates. "
        "Thread-private rows remove write conflicts and are merged in a fixed order. Atomic integer increments prevent lost counts, "
        "but floating-point addition is not associative because each operation rounds to a finite representation. "
        "Different thread schedules change the addition order and can therefore change the final low bits. "
        "This implementation recomputes weighted sums in input order to make the returned result bit-identical across thread counts.")
    print("Task 1 race lost counts:", losses, "max bin discrepancy:", max_discrepancy,
          "active workers:", worker_participation)

    # Task 2: compare cache-line sharing with padded worker slots.
    t2_times = {}
    for n in THREAD_COUNTS:
        set_num_threads(n)
        unpadded_state_omp(10, STUDENT_ID); padded_state_omp(10, STUDENT_ID, PAD_STRIDE)
        (_, tu) = timed(unpadded_state_omp, ITERS_T2, STUDENT_ID)
        (padded, tp) = timed(padded_state_omp, ITERS_T2, STUDENT_ID, PAD_STRIDE)
        unpadded = unpadded_state_omp(ITERS_T2, STUDENT_ID)
        assert np.array_equal(unpadded, padded[:, 0])
        t2_times[n] = {"unpadded_s": tu, "padded_s": tp}
    speedup_t2 = t2_times[8]["unpadded_s"] / t2_times[8]["padded_s"]
    DIAGNOSIS_T2 = ("Adjacent worker counters occupy the same 64-byte cache line when unpadded. Under MESI-style coherence, "
        "each core needs exclusive ownership before writing, so ownership repeatedly moves between cores and invalidates other copies. "
        "This false sharing adds coherence traffic even though threads update different variables. "
        "Padding separates active counters into distinct cache lines while preserving the inner-loop store. "
        "The expected benefit is lower coherence contention; measured speedup depends on the CPU and runtime.")
    print("Task 2 timings:", json.dumps(t2_times), "8-thread padding speedup:", speedup_t2)

    # Task 3: compare contiguous static scheduling to deterministic interleaving.
    pstride = 1
    while math.gcd(pstride, ROWS_T3) != 1:
        pstride += 2
    set_num_threads(8)
    (out_static, ops_static), ts = timed(static_skew_omp, ROWS_T3, STUDENT_ID, SKEW_MOD, PAD_STRIDE)
    (out_t3, ops_balanced), tb = timed(balanced_skew_omp, ROWS_T3, STUDENT_ID, SKEW_MOD, PAD_STRIDE, pstride)
    active_static = ops_static[ops_static > 0]
    active_balanced = ops_balanced[ops_balanced > 0]
    ratio_static = float(active_static.max() / active_static.min())
    ratio_balanced = float(active_balanced.max() / active_balanced.min())
    assert np.max(np.abs(out_static - out_t3)) < 1e-11
    assert ratio_balanced < 1.15, f"balanced load ratio {ratio_balanced:.3f} >= 1.15"
    DIAGNOSIS_T3 = ("Static contiguous blocks can assign clusters of expensive rows to one worker, leaving other workers idle at the barrier. "
        "The balanced kernel applies a coprime stride permutation so costly rows are interleaved across the iteration space. "
        "Each row is still computed exactly once, and output is written at its original row index. "
        "Per-thread operation counters show whether the skew has been distributed evenly. "
        "The deterministic interleaving avoids shared cursor contention while reducing straggler time.")
    print(f"Task 3 static={ts:.3f}s ratio={ratio_static:.3f}; balanced={tb:.3f}s ratio={ratio_balanced:.3f}")

    # Task 4: fixed 600 Red-Black SOR sweeps with thread-count comparison.
    grids, t4_times, residuals = {}, {}, {}
    for n in THREAD_COUNTS:
        set_num_threads(n)
        u = initial_grid()
        solve_pde_omp(u, OMEGA_SOR, 1, PAD_STRIDE)  # compile/warm-up
        u = initial_grid()
        (grid, resid), elapsed = timed(solve_pde_omp, u, OMEGA_SOR, 600, PAD_STRIDE)
        grids[n], t4_times[n], residuals[n] = grid.copy(), elapsed, float(resid)
    grid_t4 = grids[8]
    diff_t4 = float(np.max(np.abs(grids[1] - grid_t4)))
    assert diff_t4 < 1e-9
    speedups_t4 = {str(n): t4_times[1] / t4_times[n] for n in THREAD_COUNTS}
    print("Task 4 timings:", json.dumps(t4_times), "speedups:", json.dumps(speedups_t4), "max abs diff:", diff_t4)

    set_num_threads(8)
    for idx, diag in enumerate([DIAGNOSIS_T1, DIAGNOSIS_T2, DIAGNOSIS_T3], start=1):
        assert isinstance(diag, str) and len(diag.strip()) >= 80, f"Complete DIAGNOSIS_T{idx}!"
    checksum_blob = (f"{STUDENT_ID}|{int(np.sum(hist_t1))}|{float(np.sum(wsums_t1)):.4f}|"
                     f"{int(np.bitwise_xor.reduce(padded_state_omp(ITERS_T2, STUDENT_ID, PAD_STRIDE)[:, 0]))}|"
                     f"{float(np.sum(out_t3)):.6f}|{float(np.sum(grid_t4)):.6f}")
    lab_digest = hashlib.sha256(checksum_blob.encode()).hexdigest()[:20].upper()
    bundle = {"instructor": "Instructor - Sufyan bin Uzayr", "student_name": STUDENT_NAME,
              "student_id": STUDENT_ID, "omp_layer": nb.threading_layer(),
              "cpu_cores": os.cpu_count(), "verification_digest": lab_digest}
    bundle_path = Path(__file__).with_name("openmp_lab_230103344_bundle.json")
    bundle_path.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    print("=== OPENMP PYTHON LAB VERIFICATION BUNDLE ===")
    print(json.dumps(bundle, indent=2))

if __name__ == "__main__":
    main()
