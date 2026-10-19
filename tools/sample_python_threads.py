"""Run a Python module while sampling every thread's stack (no dependencies).

    python -m tools.sample_python_threads <out.json> [--interval-ms 5] -- -m <module> [args...]

A daemon thread records each thread's innermost frames every interval and
writes counts per (thread, leaf function) and per (thread, stack) at exit.
Samples show where threads are, not whether they hold the GIL: a thread in a
ctypes call or a blocking read is listed at that call.
"""
import argparse
import atexit
import collections
import json
import runpy
import sys
import threading
import time


def frames(frame, depth=12):
    out = []
    while frame is not None and len(out) < depth:
        code = frame.f_code
        out.append("%s:%s:%d" % (code.co_filename.rsplit("/", 2)[-1], code.co_name, frame.f_lineno))
        frame = frame.f_back
    return out


def main():
    argv = sys.argv[1:]
    split = argv.index("--")
    p = argparse.ArgumentParser()
    p.add_argument("output")
    p.add_argument("--interval-ms", type=float, default=5)
    args = p.parse_args(argv[:split])
    target = argv[split + 1:]
    if target[:1] != ["-m"] or len(target) < 2:
        raise SystemExit("expected -- -m <module> [args...]")
    leaf, stacks, totals = collections.Counter(), collections.Counter(), collections.Counter()
    started = time.monotonic()
    me = threading.get_ident()
    me_sampler = [None]

    def sample():
        me_sampler[0] = threading.get_ident()
        interval = args.interval_ms / 1000
        while True:
            names = {t.ident: t.name for t in threading.enumerate()}
            for ident, frame in sys._current_frames().items():
                if ident == me_sampler[0]:
                    continue
                name = names.get(ident, str(ident))
                stack = frames(frame)
                totals[name] += 1
                leaf[(name, stack[0])] += 1
                stacks[(name, " < ".join(stack[:6]))] += 1
            time.sleep(interval)

    def dump():
        top = lambda c, n: [dict(thread=k[0], where=k[1], samples=v) for k, v in c.most_common(n)]
        with open(args.output, "w") as f:
            json.dump(dict(interval_ms=args.interval_ms, seconds=time.monotonic() - started,
                           main_thread=me, threads=dict(totals), leaf=top(leaf, 80),
                           stacks=top(stacks, 120)), f, indent=1)

    atexit.register(dump)
    thread = threading.Thread(target=sample, name="sampler", daemon=True)
    thread.start()
    sys.argv = [target[1]] + target[2:]
    runpy.run_module(target[1], run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
