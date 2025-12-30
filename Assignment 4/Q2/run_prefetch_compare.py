import subprocess
import re
import csv
import statistics

# Configuration
DIMX_LIST = [1024, 8192]
NSTEPS = 2000
REPEATS = 10
TARGET = "./heat"

TIME_RE = re.compile(r"Time stepping loop\.\s*Elasped\s+([0-9.]+) microseconds", re.IGNORECASE)
GFLOPS_RE = re.compile(r"throughput\s+([0-9.]+) GFLOPS", re.IGNORECASE)


def run_once(dimx: int, nsteps: int, prefetch: int):
    out = subprocess.check_output([TARGET, str(dimx), str(nsteps), str(prefetch)], text=True)
    mt = TIME_RE.search(out)
    mg = GFLOPS_RE.search(out)
    time_us = float(mt.group(1)) if mt else float("nan")
    gflops = float(mg.group(1)) if mg else float("nan")
    return time_us / 1000.0, gflops, out  # return ms, gflops, raw


def summarize(values):
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def main():
    rows = []
    for dimx in DIMX_LIST:
        for prefetch in (1, 0):
            times_ms = []
            gflops_vals = []
            print(f"Running dimX={dimx}, nsteps={NSTEPS}, prefetch={prefetch} ...")
            for i in range(REPEATS):
                t_ms, gf, _ = run_once(dimx, NSTEPS, prefetch)
                times_ms.append(t_ms)
                gflops_vals.append(gf)
                print(f"  run {i+1}: time={t_ms:.3f} ms, gflops={gf:.3f}")
            avg_t, std_t = summarize(times_ms)
            avg_g, std_g = summarize(gflops_vals)
            rows.append((dimx, NSTEPS, prefetch, REPEATS, avg_t, std_t, avg_g, std_g))
            print(f"--> avg time={avg_t:.3f} ms (std {std_t:.3f}), avg gflops={avg_g:.3f} (std {std_g:.3f})")

    with open("prefetch_compare.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dimX", "nsteps", "prefetch", "repeats", "avg_time_ms", "std_time_ms", "avg_gflops", "std_gflops"])
        w.writerows(rows)
    print("Saved prefetch_compare.csv")


if __name__ == "__main__":
    main()
