import os
import sys
import json
import time
import ctypes
import hashlib
import platform
import threading
import subprocess
import statistics
import numpy as np
from numba import njit, uint64

try:
    import resource
except ImportError:
    resource = None

# Replace with your exact Git handle before executing
# (can also be passed as: python3 false_sharing_lab.py <git_username>)
GIT_USERNAME = "230103344"
NUM_THREADS = 4
ITERATIONS_PER_THREAD = 60_000_000
STRIDES = (1, 4, 8, 16)
REPEATS = 3                      # each layout is timed REPEATS times, median is reported
ALIGNMENT = 128                  # 128 B alignment => also 64 B aligned, and slot 0 starts a 128 B pair
GIL_CONTROL_ITERS = 1_000_000    # pure-Python control for Q5 (kept small: bytecode is slow)


def detect_cache_line_size() -> int:
    """Query L1 data cache coherency line size in bytes from OS sysfs/sysctl."""
    linux_path = "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
    if os.path.exists(linux_path):
        with open(linux_path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    if sys.platform == "darwin":
        out = subprocess.check_output(["sysctl", "-n", "hw.cachelinesize"], encoding="utf-8")
        return int(out.strip())
    return 64


def detect_cpu_model() -> str:
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.lower().startswith(("model name", "hardware", "cpu model")):
                        return line.split(":", 1)[1].strip()
        if sys.platform == "darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"],
                                           encoding="utf-8").strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def pick_physical_cores(available, k):
    """Prefer one logical CPU per physical core (avoid SMT siblings sharing one L1)."""
    chosen, seen = [], set()
    for cpu in sorted(available):
        path = f"/sys/devices/system/cpu/cpu{cpu}/topology/core_id"
        pkg = f"/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id"
        try:
            key = (open(pkg).read().strip(), open(path).read().strip())
        except OSError:
            key = ("?", str(cpu))
        if key not in seen:
            seen.add(key)
            chosen.append(cpu)
        if len(chosen) == k:
            return chosen
    # not enough distinct physical cores -> fill with remaining logical CPUs (cycling if needed)
    for cpu in sorted(available):
        if cpu not in chosen and len(chosen) < k:
            chosen.append(cpu)
    i = 0
    avail = sorted(available)
    while len(chosen) < k:
        chosen.append(avail[i % len(avail)])
        i += 1
    return chosen


# ==============================================================================
# TASK 1: Hardware Topology & Student Identity Context
# ==============================================================================
def init_system_context(git_user: str) -> dict:
    assert git_user != "REPLACE_WITH_YOUR_GIT_USERNAME" and len(git_user.strip()) > 0, (
        "Set GIT_USERNAME to your real Git handle before running."
    )
    logical_cores = os.cpu_count() or 1
    oversubscribed = logical_cores < NUM_THREADS
    if oversubscribed:
        # Free Colab CPU runtimes expose only 2 vCPUs. We still run T = 4 threads, but they
        # time-share the cores; this is recorded in the manifest ("oversubscribed": true).
        print(f"WARNING: only {logical_cores} logical cores (< {NUM_THREADS}); running oversubscribed.")

    line_size = detect_cache_line_size()
    cpu_sig = f"{platform.machine()}::{platform.processor()}::{logical_cores}"

    # Deterministic 64-bit seed: SHA-256(git_user::cpu_sig), first 16 hex digits
    digest = hashlib.sha256(f"{git_user.strip()}::{cpu_sig}".encode("utf-8")).hexdigest()
    instance_seed = int(digest[:16], 16)

    # Core affinity (Linux only; macOS has no user-level thread pinning API)
    affinity_supported = hasattr(os, "sched_setaffinity") and hasattr(os, "sched_getaffinity")
    if affinity_supported:
        available = sorted(os.sched_getaffinity(0))
        pinned_cores = pick_physical_cores(available, NUM_THREADS)
    else:
        available = list(range(logical_cores))
        pinned_cores = [None] * NUM_THREADS

    ctx = dict(
        git_user=git_user.strip(),
        logical_cores=logical_cores,
        cpu_model=detect_cpu_model(),
        cpu_sig=cpu_sig,
        os=f"{platform.system()} {platform.release()}",
        python=platform.python_version(),
        cache_line_size=line_size,
        instance_seed=instance_seed,
        sha256=digest,
        affinity_supported=affinity_supported,
        oversubscribed=oversubscribed,
        available_cpus=available,
        pinned_cores=pinned_cores,
    )
    print("=" * 78)
    print(f"Git user          : {ctx['git_user']}")
    print(f"CPU model         : {ctx['cpu_model']}")
    print(f"CPU signature     : {cpu_sig}")
    print(f"Logical cores     : {logical_cores}")
    print(f"Cache line size   : {line_size} B")
    print(f"INSTANCE_SEED     : {instance_seed} (0x{instance_seed:016x})")
    print(f"Affinity pinning  : {'ON -> cores ' + str(pinned_cores) if affinity_supported else 'not supported by OS'}")
    print("=" * 78, flush=True)
    return ctx


# ==============================================================================
# TASK 2: GIL-Free Native Counter Kernel
# ==============================================================================
@njit(nogil=True, cache=False)
def worker_strided_accumulate(buffer, slot, iters, seed):
    """In-place RMW on buffer[slot] every iteration (no register promotion).

    The store address is re-derived each iteration from a value LOADED from the
    same buffer (idx = slot + (buffer[slot] >> 63)). At run time the shift is
    always 0 (counters never reach 2**63), but LLVM cannot prove that, so the
    address is not loop-invariant and the store may alias the next load =>
    the counter cannot be kept in a register; every iteration does a real
    load + store to L1.
    """
    s = uint64(seed)
    for i in range(iters):
        v = buffer[slot]
        idx = slot + (v >> uint64(63))
        buffer[idx] = buffer[idx] + (uint64(1) + ((s + uint64(i)) & uint64(3)))
    return buffer[slot]


@njit(nogil=True, cache=False)
def worker_naive_hoistable(buffer, slot, iters, seed):
    """Reference ONLY for the LLVM audit (Q4): LLVM promotes buffer[slot] to a register here."""
    s = uint64(seed)
    for i in range(iters):
        buffer[slot] = buffer[slot] + (uint64(1) + ((s + uint64(i)) & uint64(3)))
    return buffer[slot]


# ==============================================================================
# TASK 4: Thread-Local Register Privatization Baseline
# ==============================================================================
@njit(nogil=True, cache=False)
def worker_private_accumulate(buffer, slot, iters, seed):
    """Accumulate in a local scalar (register); single store to shared memory at the end."""
    s = uint64(seed)
    acc = buffer[slot]
    for i in range(iters):
        acc = acc + (uint64(1) + ((s + uint64(i)) & uint64(3)))
    buffer[slot] = acc
    return acc


def py_strided_accumulate(buffer, slot, iters, seed):
    """Pure CPython bytecode version (holds the GIL) — Q5 control experiment."""
    for i in range(iters):
        buffer[slot] = buffer[slot] + 1 + ((seed + i) & 3)


# ------------------------------------------------------------------------------
def aligned_uint64_buffer(n_elems: int, alignment: int = ALIGNMENT) -> np.ndarray:
    raw = np.zeros(n_elems * 8 + alignment, dtype=np.uint8)
    offset = (-raw.ctypes.data) % alignment
    buf = raw[offset: offset + n_elems * 8].view(np.uint64)
    assert buf.ctypes.data % alignment == 0 and buf.flags["C_CONTIGUOUS"]
    return buf


def thread_seed(ctx, t):
    return (ctx["instance_seed"] + 0x9E3779B97F4A7C15 * (t + 1)) & 0xFFFFFFFFFFFFFFFF


def expected_counter(seed: int, iters: int) -> int:
    """Closed form of sum_{i<iters} (1 + ((seed+i) & 3)) mod 2**64 (independent check)."""
    full, rem = divmod(iters, 4)
    total = iters + full * (0 + 1 + 2 + 3)
    for i in range(rem):
        total += (seed + full * 4 + i) & 3
    return total & 0xFFFFFFFFFFFFFFFF


def rusage_snapshot():
    if resource is None:
        return (0, 0)
    r = resource.getrusage(resource.RUSAGE_SELF)
    return (r.ru_nvcsw, r.ru_nivcsw)


def _run_threads(ctx, kernel, buffer, slots, iters, pin=True, python_mode=False):
    """Launch NUM_THREADS threads behind a barrier; return elapsed ns + ctx-switch deltas."""
    n = len(slots)
    barrier = threading.Barrier(n + 1)
    errors = []

    def body(t):
        try:
            core = ctx["pinned_cores"][t] if pin else None
            if core is not None:
                os.sched_setaffinity(0, {core})          # 0 = calling thread on Linux
            seed = thread_seed(ctx, t)
            seed = int(seed) if python_mode else np.uint64(seed)
            barrier.wait()
            kernel(buffer, slots[t], iters, seed)
        except Exception as e:                            # pragma: no cover
            errors.append(repr(e))
            barrier.abort()

    threads = [threading.Thread(target=body, args=(t,), name=f"worker-{t}") for t in range(n)]
    for th in threads:
        th.start()
    v0, iv0 = rusage_snapshot()
    try:
        barrier.wait()
    except threading.BrokenBarrierError:
        pass
    t0 = time.perf_counter_ns()
    for th in threads:
        th.join()
    elapsed = time.perf_counter_ns() - t0
    v1, iv1 = rusage_snapshot()
    if errors:
        raise RuntimeError(errors)
    return elapsed, v1 - v0, iv1 - iv0


# ==============================================================================
# TASK 3: Multi-Threaded Stride Sweep
# ==============================================================================
def run_stride_benchmark(ctx) -> dict:
    iters = ITERATIONS_PER_THREAD
    total_updates = NUM_THREADS * iters
    line = ctx["cache_line_size"]
    results = {}

    # JIT warm-up (compilation excluded from timing)
    wb = aligned_uint64_buffer(16)
    for k in (worker_strided_accumulate, worker_private_accumulate, worker_naive_hoistable):
        k(wb, 0, 1000, np.uint64(1))      # same arg types as the timed calls -> no re-JIT

    configs = [(f"stride_{s}", s, worker_strided_accumulate) for s in STRIDES]
    configs.append(("private_register", 8, worker_private_accumulate))

    print(f"\n{'layout':18s} {'stride':>6s} {'byte offsets':>22s} {'lines':>12s} "
          f"{'median ns':>16s} {'ns/update':>10s} {'vcsw':>6s} {'ivcsw':>6s}")
    for name, stride, kernel in configs:
        n_elems = stride * NUM_THREADS + 32
        slots = [t * stride for t in range(NUM_THREADS)]
        runs, vcs, ivcs, finals = [], [], [], []
        for _ in range(REPEATS):
            buf = aligned_uint64_buffer(n_elems)
            el, vc, ivc = _run_threads(ctx, kernel, buf, slots, iters)
            runs.append(el)
            vcs.append(vc)
            ivcs.append(ivc)
            finals.append([int(buf[s]) for s in slots])
        base = buf.ctypes.data
        offsets = [s * 8 for s in slots]
        line_idx = [(base + o) // line - base // line for o in offsets]
        pair_idx = [(base + o) // 128 - base // 128 for o in offsets]
        med = int(statistics.median(runs))
        results[name] = dict(
            stride=stride,
            kernel=kernel.__name__ if hasattr(kernel, "__name__") else str(kernel),
            byte_offsets=offsets,
            cache_line_index=line_idx,
            prefetch_pair_128B_index=pair_idx,
            runs_ns=runs,
            median_ns=med,
            min_ns=int(min(runs)),
            ns_per_update=med / total_updates,
            ns_per_update_per_thread=med / iters,
            voluntary_ctx_switches=vcs,
            involuntary_ctx_switches=ivcs,
            final_counters=finals[-1],
            all_repeats_identical=all(f == finals[0] for f in finals),
        )
        print(f"{name:18s} {stride:6d} {str(offsets):>22s} {str(line_idx):>12s} "
              f"{med:16,d} {med / total_updates:10.3f} {statistics.median(vcs):6.0f} "
              f"{statistics.median(ivcs):6.0f}", flush=True)

    ref = results["stride_8"]["median_ns"]
    for name in results:
        results[name]["slowdown_vs_stride_8"] = results[name]["median_ns"] / ref
    print("\nSlowdown vs stride=8: " + ", ".join(
        f"{k}={v['slowdown_vs_stride_8']:.2f}x" for k, v in results.items()))
    return results


def run_gil_control(ctx) -> dict:
    """Q5: identical stride=1 workload in pure Python threads (GIL-serialized)."""
    out = {}
    iters = GIL_CONTROL_ITERS
    for n in (1, NUM_THREADS):
        buf = [0] * (NUM_THREADS * 1 + 8)                    # python list, stride 1
        slots = list(range(n))
        sub = dict(ctx, pinned_cores=ctx["pinned_cores"][:n])
        el, vc, ivc = _run_threads(sub, py_strided_accumulate, buf, slots, iters,
                                   python_mode=True)
        out[f"python_threads_{n}"] = dict(threads=n, iters_per_thread=iters, elapsed_ns=el,
                                          ns_per_update=el / (n * iters),
                                          voluntary_ctx_switches=vc, involuntary_ctx_switches=ivc)
    r = out[f"python_threads_{NUM_THREADS}"]["elapsed_ns"] / out["python_threads_1"]["elapsed_ns"]
    out["wall_ratio_4_vs_1"] = r
    print(f"\n[GIL control] pure-Python stride=1, {iters:,} updates/thread: "
          f"1 thread = {out['python_threads_1']['elapsed_ns']:,} ns, "
          f"{NUM_THREADS} threads = {out[f'python_threads_{NUM_THREADS}']['elapsed_ns']:,} ns "
          f"({r:.2f}x wall time for {NUM_THREADS}x work -> no parallel speedup)")
    return out


# ==============================================================================
# TASK 5: Exact State Parity & LLVM IR Verification
# ==============================================================================
def _native_llvm_function(llvm: str) -> str:
    """Body of the nopython function only (skip the CPython wrapper / cfunc / NRT helpers)."""
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
            if st.startswith(".Lfunc_end") or st.startswith("Lfunc_end"):
                keep = False
    return "\n".join(out)


def _ir_stats(dispatcher):
    llvm_all = "\n".join(dispatcher.inspect_llvm().values())
    asm_all = "\n".join(dispatcher.inspect_asm().values())
    native_ir = _native_llvm_function(llvm_all)
    return dict(
        store_i64=native_ir.count("store i64"),
        load_i64=native_ir.count("load i64"),
        pyobject_refs_in_native=sum(native_ir.count(k) for k in ("PyObject", "@Py", "pyobject")),
        releases_gil_in_wrapper=("PyEval_SaveThread" in llvm_all and "PyEval_RestoreThread" in llvm_all),
        nopython_signatures=[str(s) for s in dispatcher.nopython_signatures],
        llvm_text=native_ir,
        asm_text=_native_asm_function(asm_all),
    )


def _store_in_loop(asm_text: str) -> bool:
    """x86/ARM heuristic: is there a memory-destination store inside a loop body?"""
    import re
    lines = asm_text.splitlines()
    labels = {l.strip()[:-1]: i for i, l in enumerate(lines) if re.match(r"^\s*\.?L\w+:$", l)}
    for i, l in enumerate(lines):
        m = re.search(r"\b(j\w+|b\.\w+|b|cbnz|cbz|tbnz)\s+(\.?L\w+)", l)
        if m and m.group(2) in labels and labels[m.group(2)] < i:       # backward branch = loop
            body = lines[labels[m.group(2)]: i + 1]
            for b in body:
                if re.search(r"(mov\w*|add\w*)\s+%?\w+,\s*-?\d*\(", b) or re.search(r"\bstr\s", b) \
                        or re.search(r"\bstp?\s", b) or re.search(r"\b(add|mov)q?\s+.*,\s*\(%", b):
                    return True
    return False


def audit_llvm_and_parity(ctx, results, gil) -> dict:
    iters = ITERATIONS_PER_THREAD
    expected = [expected_counter(thread_seed(ctx, t), iters) for t in range(NUM_THREADS)]
    parity = {name: (r["final_counters"] == expected and r["all_repeats_identical"])
              for name, r in results.items()}
    parity_ok = all(parity.values())

    strided = _ir_stats(worker_strided_accumulate)
    naive = _ir_stats(worker_naive_hoistable)
    private = _ir_stats(worker_private_accumulate)
    strided_store_in_loop = _store_in_loop(strided["asm_text"])
    naive_store_in_loop = _store_in_loop(naive["asm_text"])
    with open("worker_strided_accumulate.ll", "w") as f:
        f.write(strided["llvm_text"])
    with open("worker_strided_accumulate.s", "w") as f:
        f.write(strided["asm_text"])

    llvm_ok = strided["store_i64"] > 0 and strided["releases_gil_in_wrapper"] and strided["pyobject_refs_in_native"] == 0 and \
        len(strided["nopython_signatures"]) > 0

    print("\n[Parity] expected final counters per thread:", expected)
    for k, v in parity.items():
        print(f"  {k:18s}: {'BIT-EXACT' if v else 'MISMATCH'}  {results[k]['final_counters']}")
    print(f"[LLVM] worker_strided_accumulate: store i64 = {strided['store_i64']}, "
          f"load i64 = {strided['load_i64']}, PyObject refs in native fn = {strided['pyobject_refs_in_native']}, "
          f"store inside loop (asm) = {strided_store_in_loop}, "
          f"GIL released (PyEval_SaveThread) = {strided['releases_gil_in_wrapper']}")
    print(f"[LLVM] worker_naive_hoistable   : store i64 = {naive['store_i64']}, "
          f"store inside loop (asm) = {naive_store_in_loop}  (promoted to register => False expected)")
    print(f"[LLVM] worker_private_accumulate: store i64 = {private['store_i64']}")

    verification_passed = bool(parity_ok and llvm_ok)
    manifest = dict(
        git_username=ctx["git_user"],
        instance_seed=str(ctx["instance_seed"]),
        instance_seed_hex=f"0x{ctx['instance_seed']:016x}",
        cpu_model=ctx["cpu_model"],
        cpu_signature=ctx["cpu_sig"],
        os=ctx["os"],
        python=ctx["python"],
        numba=__import__("numba").__version__,
        numpy=np.__version__,
        logical_cores=ctx["logical_cores"],
        oversubscribed=ctx["oversubscribed"],
        cache_line_size=ctx["cache_line_size"],
        affinity_supported=ctx["affinity_supported"],
        pinned_cores=ctx["pinned_cores"],
        num_threads=NUM_THREADS,
        iterations_per_thread=ITERATIONS_PER_THREAD,
        repeats=REPEATS,
        buffer_alignment_bytes=ALIGNMENT,
        results=results,
        slowdown_stride1_vs_stride8=results["stride_1"]["slowdown_vs_stride_8"],
        slowdown_stride4_vs_stride8=results["stride_4"]["slowdown_vs_stride_8"],
        stride16_vs_stride8=results["stride_16"]["slowdown_vs_stride_8"],
        private_vs_stride8=results["private_register"]["slowdown_vs_stride_8"],
        gil_control=gil,
        expected_counters=[str(e) for e in expected],
        parity=parity,
        parity_passed=parity_ok,
        llvm_audit=dict(
            worker_strided_accumulate=dict(
                store_i64=strided["store_i64"], load_i64=strided["load_i64"],
                pyobject_refs_in_native=strided["pyobject_refs_in_native"],
                releases_gil_in_wrapper=strided["releases_gil_in_wrapper"],
                nopython_signatures=strided["nopython_signatures"],
                store_inside_loop_asm=strided_store_in_loop),
            worker_naive_hoistable=dict(
                store_i64=naive["store_i64"], store_inside_loop_asm=naive_store_in_loop),
            worker_private_accumulate=dict(store_i64=private["store_i64"]),
            passed=llvm_ok,
        ),
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        verification_passed=verification_passed,
    )
    with open("cache_lab_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\ncache_lab_manifest.json written | verification_passed = {verification_passed}")
    return manifest


def main():
    git_user = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GIT_USERNAME", GIT_USERNAME)
    ctx = init_system_context(git_user)
    results = run_stride_benchmark(ctx)
    gil = run_gil_control(ctx)
    manifest = audit_llvm_and_parity(ctx, results, gil)
    if not manifest["verification_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
