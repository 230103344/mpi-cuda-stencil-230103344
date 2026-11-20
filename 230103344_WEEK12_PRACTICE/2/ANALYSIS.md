# ANALYSIS — Cache Line & False Sharing Mechanics

**Student:** Iliyas Malibekov · **ID:** 230103344 · **GitHub:** 230103344 · **Repo:** cache-line-false-sharing-230103344

> Replace every `⟨…⟩` with the values from **your** `cache_lab_manifest.json` / stdout.

**Host:** ⟨cpu_model⟩ · ⟨logical_cores⟩ logical cores · cache line = ⟨cache_line_size⟩ B · pinned cores = ⟨pinned_cores⟩
**INSTANCE_SEED:** ⟨instance_seed_hex⟩ · **T = 4 threads × 60,000,000 updates** (240,000,000 updates per layout, median of 3 runs)

| Layout | stride | byte offsets | cache lines | median ns | ns / update | slowdown vs stride 8 |
|---|---|---|---|---|---|---|
| A Contiguous | 1 | 0, 8, 16, 24 | 0,0,0,0 | ⟨…⟩ | ⟨…⟩ | ⟨…⟩× |
| B Half-line | 4 | 0, 32, 64, 96 | 0,0,1,1 | ⟨…⟩ | ⟨…⟩ | ⟨…⟩× |
| C Line-padded | 8 | 0, 64, 128, 192 | 0,1,2,3 | ⟨…⟩ | ⟨…⟩ | 1.00× |
| D Prefetch-safe | 16 | 0, 128, 256, 384 | 0,2,4,6 | ⟨…⟩ | ⟨…⟩ | ⟨…⟩× |
| Private register | — | 0, 64, 128, 192 | (1 store) | ⟨…⟩ | ⟨…⟩ | ⟨…⟩× |

## Q1 — MESI Protocol State Transitions (stride = 1)
`buffer[0]` and `buffer[1]` are 8 B apart, so they sit in the **same 64-byte line L**.
1. Start: L is in no cache, i.e. **I/I** (Core 0 / Core 1).
2. Core 0 stores to `buffer[0]`. This is a write miss, so Core 0 issues an **RFO** (Read-For-Ownership = read + invalidate). No other core holds L, so Core 0 gets it exclusively, **I→E**, and the store makes it **E→M**. State: **M/I**.
3. Core 1 stores to `buffer[1]`. Its copy is Invalid, so it puts the store into its **store buffer** and sends an RFO for L. Core 0 snoops the RFO, writes back/forwards its dirty line (**M→I**), and Core 1 receives the line, **I→M**. State: **I/M**.
4. Core 0's next increment of `buffer[0]` needs L again: it sends an RFO, so Core 1 goes **M→I** and Core 0 goes **I→M**. The ping-pong repeats for every iteration on every core.

There is no `Shared` state for long. Both cores only *write*, so every request is an RFO that invalidates the other copy. Core 1's store **cannot retire** from the store buffer into L1 until it owns L in M/E state. The store buffer fills up while each RFO waits a full cross-core round trip (tens of ns through L3/the ring/mesh instead of ~1 ns L1 hits). Our kernel also *loads* `buffer[slot]` each iteration (read-modify-write), so the dependent load also stalls on the same line. With 4 threads on one line, ownership rotates through 4 cores, which is the **4-way RFO ping-pong** in my stride-1 timing (⟨…⟩ ns/update).

## Q2 — Quantitative Slowdown & Coherency Overhead
- stride 1: ⟨…⟩ ns total → ⟨…⟩ ns per update (240 M updates) → ⟨…⟩ ns per update per thread
- stride 4: ⟨…⟩ ns → ⟨…⟩ ns/update
- stride 8: ⟨…⟩ ns → ⟨…⟩ ns/update
- stride 16: ⟨…⟩ ns → ⟨…⟩ ns/update
- private register: ⟨…⟩ ns → ⟨…⟩ ns/update

**Slowdown stride 1 / stride 8 = ⟨…⟩×.** The stride-8 time is the cost of an L1-resident load→add→store chain. That chain is limited by **store-to-load forwarding** latency, because each iteration reloads the value it just stored (≈4–5 cycles ≈ ⟨…⟩ ns per update per thread). The **extra** ⟨stride1 − stride8⟩ ns per update is pure **cross-core coherence latency** (RFO + invalidation). Stride 4 (⟨…⟩×) is in between: only 2 cores fight over each line (2-way false sharing). The privatized baseline (⟨…⟩×) keeps the counter in a register, so it has neither the memory chain nor coherence traffic. Comparing private vs stride 8 isolates the store-forwarding cost; comparing stride 8 vs stride 1 isolates the coherence cost. Context switches (vcsw/ivcsw ≈ ⟨…⟩) are tiny for all layouts, so the slowdown is **not** caused by the OS scheduler. It comes from hardware.

## Q3 — L2 Spatial Prefetcher & 128-Byte Pairing
stride 8 = ⟨…⟩ ns vs stride 16 = ⟨…⟩ ns (ratio ⟨…⟩×). Intel's **L2 adjacent-cache-line (spatial) prefetcher** treats memory as 128-byte-aligned pairs of 64-byte lines (even line 2k + odd line 2k+1). When a core misses on one line, it also fetches the "buddy" line. With stride 8, threads 0/1 own lines 0/1, which are the *same 128-B pair*, and threads 2/3 own lines 2/3. A miss or RFO by thread 0 can make its core prefetch line 1, which pulls a copy of thread 1's line into the wrong core. The next store by thread 1 then has to invalidate it again. This is a **secondary invalidation** even though no two threads share a 64-B line. It happens when the prefetcher is enabled, the threads are on different cores, the lines are in the same 128-B-aligned pair, and the lines miss often (e.g. after other coherence traffic or context migration). With stride 16 every thread owns a different 128-B pair, so the prefetcher can't create cross-thread traffic. ⟨If your stride-16 is only slightly faster or equal: in steady state each line stays in M in its owner's L1, so there are few misses and the prefetcher rarely triggers; the effect is small.⟩ (On Apple Silicon `hw.cachelinesize` = 128 B, so stride 8 is real 2-way false sharing there and only stride 16 is padded.)

## Q4 — Compiler Register Promotion vs. Volatile Memory Stores
LLVM's **LICM / scalar promotion** pass moves a memory location into a register when (a) its address is loop-invariant and (b) no other access in the loop may alias it. In a naive loop `buffer[slot] += inc`, both hold. LLVM loads `buffer[slot]` once, keeps it in a register, and stores it once after the loop. Single-threaded semantics allow this because Numba arrays are not `volatile`/atomic. My audit proves it. In `worker_naive_hoistable` the asm loop has **no store inside the loop** (`store_inside_loop_asm = ⟨false⟩`): it is even vectorized into `vpaddq` and stored once after the loop. That loop would show no false sharing at all.

In `worker_strided_accumulate` the store address is computed each iteration from a **loaded value**:
```python
v = buffer[slot]; idx = slot + (v >> 63); buffer[idx] = buffer[idx] + inc
```
Because `idx` depends on memory, the address is **not loop-invariant**. The store to `buffer[idx]` *may alias* the next load of `buffer[slot]`, so LLVM must reload and store every iteration. At run time `v >> 63` is always 0 (counters stay ≪ 2⁶³), so the result is identical. The native function's IR has **⟨…⟩ `store i64`** and **⟨…⟩ `load i64`**, with **⟨0⟩ PyObject references** (pure nopython). The signature is `⟨(Array(uint64, 1, 'C', …), int64, int64, uint64) -> uint64⟩`. In the asm the loop body contains a memory-destination add (`addq %reg, (%base,%idx,8)` on x86), so **every iteration issues a real L1 store** (`store_inside_loop_asm = ⟨true⟩`).

## Q5 — CPython GIL Masking vs. `nogil=True`
Pure-Python stride-1 control (⟨…⟩ updates/thread): 1 thread = ⟨…⟩ ns, 4 threads = ⟨…⟩ ns (⟨…⟩× wall time for 4× work). This means **zero parallel speedup**. Only one thread can execute bytecode while holding the GIL, so the 4 threads never store to the cache line *at the same time*. There is no simultaneous RFO ping-pong, so false sharing cannot appear. Each `buffer[slot] = buffer[slot] + …` is dozens of bytecodes: refcounting, boxed int allocation, list item writes. That is hundreds of ns per update, so any coherence miss is hidden. More threads make it **slower**, because of GIL **lock contention**: the interpreter forces a switch every 5 ms (`sys.getswitchinterval`), threads wait on the GIL mutex/condvar (context switches: ⟨…⟩), and the GIL itself and interpreter state bounce between cores. That is *software* lock contention, not hardware cache-line contention. With `@njit(nogil=True)` the wrapper calls `PyEval_SaveThread()` before the native loop (`releases_gil_in_wrapper = ⟨true⟩` in my manifest) and `PyEval_RestoreThread()` after. The 4 threads then run machine code truly in parallel on 4 cores, and only then does the hardware MESI contention appear (stride 1 = ⟨…⟩× slower than stride 8).
