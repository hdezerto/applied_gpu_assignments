import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('timing_results.csv')

N = df['N'].values
h2d = df['HostToDevice_ms'].values
kernel = df['Kernel_ms'].values
d2h = df['DeviceToHost_ms'].values

ind = np.arange(len(N))
plt.figure(figsize=(10,6))
bar1 = plt.bar(ind, h2d, label='Host to Device')
bar2 = plt.bar(ind, kernel, bottom=h2d, label='Kernel')
bar3 = plt.bar(ind, d2h, bottom=h2d+kernel, label='Device to Host')

# Add total time label above each bar
total = h2d + kernel + d2h
for i in range(len(N)):
	plt.text(ind[i], total[i]+max(total)*0.01, f'{total[i]:.2f}', ha='center', va='bottom', color='black', fontsize=10)
plt.xticks(ind, N, rotation=45)
plt.xlabel('Vector Length (N)')
plt.ylabel('Time (ms)')
plt.title('CUDA Vector Add Timing Breakdown')
plt.legend()
plt.tight_layout()
plt.savefig('vectoradd_timing_breakdown.png', dpi=300)
plt.show()