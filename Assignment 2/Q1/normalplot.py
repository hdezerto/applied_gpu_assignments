import csv

import matplotlib.pyplot as plt

# normal values
normalx1024 = []
normalx10240 = []
normalx102400 = []
normalx1024000 = []

normaly1024 = []
normaly10240 = []
normaly102400 = []
normaly1024000 = []

with open("data collection/normal/1024.csv", "r") as file:
    plots = csv.reader(file, delimiter=",")

    for row in plots:
        normalx1024.append(int(row[0]))
        normaly1024.append(int(row[1]))

with open("data collection/normal/10240.csv", "r") as file:
    plots = csv.reader(file, delimiter=",")

    for row in plots:
        normalx10240.append(int(row[0]))
        normaly10240.append(int(row[1]))

with open("data collection/normal/102400.csv", "r") as file:
    plots = csv.reader(file, delimiter=",")

    for row in plots:
        normalx102400.append(int(row[0]))
        normaly102400.append(int(row[1]))

with open("data collection/normal/1024000.csv", "r") as file:
    plots = csv.reader(file, delimiter=",")

    for row in plots:
        normalx1024000.append(int(row[0]))
        normaly1024000.append(int(row[1]))

plt.plot(normalx1024, normaly1024, label="1024")
plt.plot(normalx10240, normaly10240, label="10240")
plt.plot(normalx102400, normaly102400, label="102400")
plt.plot(normalx1024000, normaly1024000, label="1024000")

plt.xlabel("bins")
plt.ylabel("frequency")
plt.title("histogram")
plt.legend()
plt.savefig("normal plot.png")
plt.show()
