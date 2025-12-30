import subprocess
import re
import csv

import matplotlib.pyplot as plt


dimX = 1024
nsteps_list = [100, 200, 400, 800, 1600, 3200, 6400, 10000]
rows = []

for n in nsteps_list:
    out = subprocess.check_output(["./heat", str(dimX), str(n)], text=True)
    m = re.search(r"relative error .* is ([0-9.eE+-]+)", out)
    err = float(m.group(1)) if m else float("nan")
    rows.append((n, err))
    print(f"nsteps={n}, error={err}")

with open("error_sweep.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["nsteps", "relative_error"])
    w.writerows(rows)

plt.semilogx([r[0] for r in rows], [r[1] for r in rows], marker="o")
plt.xlabel("nsteps (log scale)")
plt.ylabel("relative error")
plt.grid(True, which="both")
plt.tight_layout()
plt.savefig("error_sweep.png", dpi=150)
print("Saved error_sweep.csv and error_sweep.png")
