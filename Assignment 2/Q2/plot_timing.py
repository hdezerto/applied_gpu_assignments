import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV produced by `./reduction_full --sweep ... > reduction_timing.csv`
df = pd.read_csv('reduction_timing.csv')

lengths = df['length'].values
cpu = df['cpu_ms'].values
gpu = df['gpu_ms'].values

indices = np.arange(len(lengths))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(indices - width / 2, cpu, width, label='CPU')
plt.bar(indices + width / 2, gpu, width, label='GPU')

plt.xticks(indices, lengths, rotation=45)
plt.xlabel('Array Length')
plt.ylabel('Time (ms)')
plt.title('Reduction Timing: CPU vs GPU')
plt.legend()
plt.tight_layout()
plt.savefig('reduction_timing_bar_chart.png', dpi=300)
plt.show()
