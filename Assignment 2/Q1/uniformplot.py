import csv

import matplotlib.pyplot as plt

# uniform values
uniformx1024 = []
uniformx10240 = []
uniformx102400 = []
uniformx1024000 = []

uniformy1024 = []
uniformy10240 = []
uniformy102400 = []
uniformy1024000 = []

with open("data collection/uniform/1024.csv", "r") as file:
    plots = csv.reader(file, delimiter=",")

    for row in plots:
        uniformx1024.append(int(row[0]))
        uniformy1024.append(int(row[1]))

with open("data collection/uniform/10240.csv", "r") as file:
    plots = csv.reader(file, delimiter=",")

    for row in plots:
        uniformx10240.append(int(row[0]))
        uniformy10240.append(int(row[1]))

with open("data collection/uniform/102400.csv", "r") as file:
    plots = csv.reader(file, delimiter=",")

    for row in plots:
        uniformx102400.append(int(row[0]))
        uniformy102400.append(int(row[1]))

with open("data collection/uniform/1024000.csv", "r") as file:
    plots = csv.reader(file, delimiter=",")

    for row in plots:
        uniformx1024000.append(int(row[0]))
        uniformy1024000.append(int(row[1]))

plt.plot(uniformx1024, uniformy1024, label="1024")
plt.plot(uniformx10240, uniformy10240, label="10240")
plt.plot(uniformx102400, uniformy102400, label="102400")
plt.plot(uniformx1024000, uniformy1024000, label="1024000")

plt.xlabel("bins")
plt.ylabel("frequency")
plt.title("histogram")
plt.legend()
plt.savefig("uniform plot.png")
plt.show()
