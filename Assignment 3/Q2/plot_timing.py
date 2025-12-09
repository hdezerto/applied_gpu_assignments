"""Plot timing comparisons for streamed vs baseline (Q1) and segment-size impact (Q3).

Expected CSV columns: N, GPUElapsed_ms, MaxError, Streamed, S_seg
"""

import argparse

import pandas as pd
import matplotlib.pyplot as plt


def load_data(path: str = "timing_results.csv") -> pd.DataFrame:
	df = pd.read_csv(path)
	required = {"N", "GPUElapsed_ms", "Streamed", "S_seg"}
	missing = required.difference(df.columns)
	if missing:
		raise ValueError(f"Missing columns in CSV: {missing}")
	return df


def plot_stream_vs_baseline(df: pd.DataFrame):
	"""Compare GPU time vs N for streamed vs non-streamed (always annotated)."""
	base = df[df["Streamed"] == 0]
	stream = df[df["Streamed"] == 1]
	if base.empty or stream.empty:
		print("Need both baseline (Streamed=0) and streamed (Streamed=1) data to plot Q1; skipping.")
		return

	# Take best (min) time per N for each mode to handle repeated runs
	base_agg = base.groupby("N")["GPUElapsed_ms"].min().reset_index()
	stream_agg = stream.groupby("N")["GPUElapsed_ms"].min().reset_index()

	plt.figure(figsize=(8, 5))

	label_stream = "Streamed (4 streams)"
	uniq_s = stream["S_seg"].unique()
	if len(uniq_s) == 1:
		label_stream += f" | S_seg={uniq_s[0]}"

	plt.plot(base_agg["N"], base_agg["GPUElapsed_ms"], marker="o", label="Baseline (no streams)")
	plt.plot(stream_agg["N"], stream_agg["GPUElapsed_ms"], marker="s", label=label_stream)
	plt.xlabel("Vector length N")
	plt.ylabel("GPU time (ms)")
	plt.xscale("log", base=2)
	plt.yscale("log")
	plt.title("Streamed vs baseline GPU time")
	plt.grid(True, which="both", linestyle="--", alpha=0.4)
	plt.legend()

	if len(uniq_s) > 1:
		# Annotate streamed points with their S_seg value at that N (pick min time per N, so choose matching S_seg with min)
		# We reconstruct per-N best S_seg from original streamed data.
		best = stream.groupby(["N", "S_seg"])["GPUElapsed_ms"].min().reset_index()
		best_n = best.loc[best.groupby("N")["GPUElapsed_ms"].idxmin()]
		for _, row in best_n.iterrows():
			plt.text(row["N"], row["GPUElapsed_ms"], f"S={int(row['S_seg'])}", fontsize=8, ha="left", va="bottom")

	plt.tight_layout()
	plt.savefig("vectoradd_stream_vs_baseline.png", dpi=300)
	print("Saved vectoradd_stream_vs_baseline.png")


def plot_segment_size(df: pd.DataFrame):
	"""Impact of segment size for a fixed large N (use max N available)."""
	stream = df[df["Streamed"] == 1]
	if stream.empty:
		print("No streamed data found; cannot plot segment size impact.")
		return
	target_N = stream["N"].max()
	subset = stream[stream["N"] == target_N]
	if subset.empty:
		print(f"No streamed data at N={target_N}; skipping Q3 plot.")
		return
	agg = subset.groupby("S_seg")["GPUElapsed_ms"].min().reset_index().sort_values("S_seg")

	plt.figure(figsize=(8, 5))
	plt.bar(agg["S_seg"].astype(str), agg["GPUElapsed_ms"], color="#4C72B0")
	plt.xlabel("Segment size S_seg")
	plt.ylabel("GPU time (ms)")
	plt.title(f"Segment size impact (N={target_N})")
	plt.grid(axis="y", linestyle="--", alpha=0.4)
	plt.tight_layout()
	plt.savefig("vectoradd_segment_size.png", dpi=300)
	print("Saved vectoradd_segment_size.png")


def main():
	parser = argparse.ArgumentParser(description="Plot timing results for vectorAdd.")
	parser.add_argument("--csv", default="timing_results.csv", help="Path to timing CSV.")
	parser.add_argument("--q1", action="store_true", help="Plot streamed vs baseline (Q1).")
	parser.add_argument("--q3", action="store_true", help="Plot segment-size impact (Q3).")
	args = parser.parse_args()

	# If neither flag is set, do both
	do_q1 = args.q1 or not (args.q1 or args.q3)
	do_q3 = args.q3 or not (args.q1 or args.q3)

	df = load_data(args.csv)
	if do_q1:
		plot_stream_vs_baseline(df)
	if do_q3:
		plot_segment_size(df)


if __name__ == "__main__":
	main()