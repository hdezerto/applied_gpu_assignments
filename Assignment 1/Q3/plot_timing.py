import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('timing_results.csv')

# Extract values
NARows = df['NARows'].values
NBColumns = df['NBColumns'].values
labels = [f"{r}x{c}" for r, c in zip(NARows, NBColumns)]
h2d = df['HostToDevice_ms'].values
kernel = df['Kernel_ms'].values
d2h = df['DeviceToHost_ms'].values

ind = np.arange(len(labels))
plt.figure(figsize=(12,6))
plt.bar(ind, h2d, label='Host to Device')
plt.bar(ind, kernel, bottom=h2d, label='Kernel')
plt.bar(ind, d2h, bottom=h2d+kernel, label='Device to Host')
plt.xticks(ind, labels, rotation=45)
plt.xlabel('Matrix Size (A_rows x B_cols)')
plt.ylabel('Time (ms)')
plt.title('Matrix Multiplication Timing Breakdown')
plt.legend()
plt.tight_layout()
plt.savefig('matrixmul_timing_breakdown.png', dpi=300)
plt.show()
