#!/usr/bin/env python3
"""Repro #2, following up on repro_mlx_stream_leak.py's clean-path result
(no leak under strict sequential join-then-next). This tests the ONE
scenario that script deliberately didn't cover: _stream_kokoro_mlx's own
documented "known, accepted residual risk" -- a new sentence's thread
starting to generate while the PREVIOUS sentence's thread is still alive
(stalled/slow), so two new_thread_unsafe_stream() registrations are live
at once, and the first thread is never joined (same as production: daemon
threads, no explicit join when the consumer gives up and moves on).

If a leak exists, it's more likely to show up here than in the clean
sequential case already ruled out.
"""
import gc
import os
import subprocess
import threading
import time

PID = os.getpid()
N = 60


def os_thread_count():
    out = subprocess.run(["ps", "-M", str(PID)], capture_output=True, text=True).stdout
    lines = [l for l in out.strip().split("\n") if l.strip()]
    return max(0, len(lines) - 1)


def produce_slow(i, delay):
    """Mimics a stalled generate() call: registers a stream, does real
    GPU work, but takes longer than the consumer is willing to wait --
    so production code abandons it (never joins) while it's still alive."""
    import mlx.core as mx
    mx.set_default_stream(mx.new_thread_unsafe_stream(mx.gpu))
    time.sleep(delay)
    a = mx.random.normal((512, 512))
    b = mx.random.normal((512, 512))
    c = mx.matmul(a, b)
    mx.eval(c)


def main():
    import mlx.core as mx
    _ = mx.array([1.0])
    mx.eval(_)

    baseline = os_thread_count()
    print(f"baseline OS thread count: {baseline}")

    threads = []
    for i in range(N):
        # Each thread stalls a bit, but the next one fires almost
        # immediately after (0.02s) -- the orphaning/overlap pattern
        # the code comment describes, not a clean one-at-a-time flow.
        t = threading.Thread(target=produce_slow, args=(i, 0.3), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(0.02)  # next "sentence" starts before this one finishes
        if i % 10 == 0:
            print(f"spawned {i + 1}/{N} overlapping threads, OS count now = {os_thread_count()}")

    print(f"\nall {N} threads launched (overlapping, none joined yet). OS count = {os_thread_count()}")
    print("waiting for all of them to actually finish their work in the background...")
    time.sleep(N * 0.3 + 5)  # generous: let every stalled op actually complete
    after_all_done = os_thread_count()
    print(f"OS count once every thread's real work should be done: {after_all_done}")

    gc.collect()
    time.sleep(5)
    settled = os_thread_count()

    print()
    print(f"baseline:                         {baseline}")
    print(f"after all {N} overlapping threads:  {after_all_done}  (delta: {after_all_done - baseline})")
    print(f"after 5s cooldown+gc:             {settled}  (delta: {settled - baseline})")
    print()
    if settled - baseline >= N * 0.5:
        print("LEAK CONFIRMED under overlap/orphaning: thread count grew roughly 1:1 and did NOT come back down.")
    elif settled - baseline <= 2:
        print("NO LEAK even under overlap: thread count returned to baseline.")
    else:
        print("PARTIAL LEAK under overlap: some threads never got reclaimed -- this is the mechanism.")


if __name__ == "__main__":
    main()
