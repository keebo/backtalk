#!/usr/bin/env python3
"""Standalone repro for the MLX-thread-leak theory in TROUBLESHOOTING.md
(2026-10-08 entry): does mx.new_thread_unsafe_stream(mx.gpu) permanently
leak an OS thread every time it's called on a freshly-created Python
thread, mirroring mouth.py's _stream_kokoro_mlx per-sentence pattern?

Mirrors the exact call shape used in production (mouth.py's _produce):
    def _produce():
        import mlx.core as mx
        mx.set_default_stream(mx.new_thread_unsafe_stream(mx.gpu))
        <do a real MLX/GPU op>
    threading.Thread(target=_produce, daemon=True).start()

Run with backtalk's own venv so the MLX version matches production exactly:
    backtalk/.venv/bin/python3 backtalk/repro_mlx_stream_leak.py
"""
import gc
import os
import subprocess
import threading
import time

PID = os.getpid()
N = 80


def os_thread_count():
    """Real OS-level thread count for this process, matching what the
    crash reporter counted (407) -- NOT threading.active_count(), which
    only sees live Python Thread objects, not MLX's internal native
    scheduler threads."""
    out = subprocess.run(["ps", "-M", str(PID)], capture_output=True, text=True).stdout
    lines = [l for l in out.strip().split("\n") if l.strip()]
    return max(0, len(lines) - 1)


def produce(i, results):
    import mlx.core as mx
    mx.set_default_stream(mx.new_thread_unsafe_stream(mx.gpu))
    a = mx.random.normal((256, 256))
    b = mx.random.normal((256, 256))
    c = mx.matmul(a, b)
    mx.eval(c)
    results[i] = float(c.sum().item())


def main():
    import mlx.core as mx  # warm up on the main thread first, like warm_mlx() does
    _ = mx.array([1.0])
    mx.eval(_)

    results = {}
    baseline = os_thread_count()
    print(f"baseline OS thread count (before any per-sentence threads): {baseline}")

    counts = []
    for i in range(N):
        t = threading.Thread(target=produce, args=(i, results), daemon=True)
        t.start()
        t.join(timeout=10)
        if t.is_alive():
            print(f"iteration {i}: thread did not finish in time, skipping count")
            continue
        time.sleep(0.05)  # let any async teardown settle before counting
        c = os_thread_count()
        counts.append(c)
        if i % 10 == 0 or i == N - 1:
            print(f"after sentence {i + 1}/{N}: OS thread count = {c} (delta from baseline: {c - baseline})")

    immediately_after = counts[-1]

    # Distinguish "permanent leak" from "transiently elevated, reclaimed late":
    # force a GC pass and give MLX/the OS a few seconds to settle before the
    # final read.
    del results
    gc.collect()
    time.sleep(5)
    settled = os_thread_count()

    print()
    print(f"baseline:              {baseline}")
    print(f"immediately after {N}:  {immediately_after}  (delta: {immediately_after - baseline})")
    print(f"after 5s cooldown+gc:   {settled}  (delta: {settled - baseline})")
    print()
    if settled - baseline >= N * 0.5:
        print("LEAK CONFIRMED: thread count grew roughly 1:1 with sentences spoken and did NOT come back down after cooldown.")
    elif settled - baseline <= 2:
        print("NO LEAK: thread count returned to baseline after cooldown -- any growth during the loop was transient.")
    else:
        print("INCONCLUSIVE: partial growth survived cooldown -- worth a second, longer run before concluding either way.")


if __name__ == "__main__":
    main()
