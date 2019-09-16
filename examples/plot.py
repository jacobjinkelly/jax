"""
For parsing and plotting the results of neural_odes.py
"""
import numpy as np
import matplotlib.pyplot as plt

dir = "2019-09-15-22-54-04"
results_path = "%s/results.txt" % dir

file = open(results_path, "r")

lines = file.readlines()

inds = []
for i, line in enumerate(lines):
    if line.startswith("Reg:"):
        inds.append(i)

plot_points = {}
for ind, next_ind in zip(inds, inds[1:] + [len(lines)]):
    split = lines[ind][len("Reg: "):].split("Lambda ")
    reg, lam = split[0][:-1], float(split[1][:-1])
    if reg not in plot_points:
        plot_points[reg] = {}
        plot_points[reg]["lam"] = []
        plot_points[reg]["loss"] = []
        plot_points[reg]["all_forward"] = {}
        plot_points[reg]["all_backward"] = {}
        plot_points[reg]["forward"] = []
        plot_points[reg]["backward"] = []

    plot_points[reg]["lam"].append(lam)
    data = lines[ind + 1:next_ind]

    # TODO: change Error to Loss when parsing new results
    final_error = float(data[-1].split(" | ")[-1].split("Error ")[-1][:-1])
    plot_points[reg]["loss"].append(final_error)

    for i, line in enumerate(data):
        if line.startswith("Iter"):
            data.pop(i)

    forward = data[0::2]
    backward = data[1::2]
    for i, point in enumerate(forward):
        forward[i] = int(point.split("forward NFE: ")[-1][:-1])
    for i, point in enumerate(backward):
        backward[i] = int(point.split("backward NFE: ")[-1][:-1])
    plot_points[reg]["all_forward"][lam] = forward[::20]
    plot_points[reg]["all_backward"][lam] = backward[::20]
    plot_points[reg]["forward"].append(np.mean(forward))
    plot_points[reg]["backward"].append(np.mean(backward))

fig, ax = plt.subplots()
for reg in plot_points.keys():
    x, y = plot_points[reg]["forward"], plot_points[reg]["loss"]
    ax.plot(x, y, "-o", label=str(reg))
    # for i, txt in enumerate(losses[reg]["lam"]):
    #     ax.annotate(txt, (x[i], y[i]))

plt.legend()
plt.xlabel("Forward NFE")
plt.ylabel("Loss")
plt.savefig("%s/forward.png" % dir)

fig, ax = plt.subplots()
for reg in plot_points.keys():
    x, y = plot_points[reg]["backward"], plot_points[reg]["loss"]
    ax.plot(x, y, "-o", label=str(reg))
    # for i, txt in enumerate(losses[reg]["lam"]):
    #     ax.annotate(txt, (x[i], y[i]))

plt.legend()
plt.xlabel("Backward NFE")
plt.ylabel("Loss")
plt.savefig("%s/backward.png" % dir)

for reg in plot_points.keys():
    print("Reg: %s" % reg)
    print(plot_points[reg]["lam"])
    print(plot_points[reg]["forward"])
    print(plot_points[reg]["backward"])
    print("")


for reg in plot_points.keys():
    fig, ax = plt.subplots()
    for lam in plot_points[reg]["all_forward"].keys():
        x, y = zip(*enumerate(plot_points[reg]["all_forward"][lam]))
        x = 20 * np.array(x)
        ax.plot(x, y, label=lam)

    plt.legend()
    plt.title("Reg: %s" % reg)
    plt.xlabel("Training")
    plt.ylabel("Forward NFE")
    plt.savefig("%s/%s_all_forward.png" % (dir, reg))


for reg in plot_points.keys():
    fig, ax = plt.subplots()
    for lam in plot_points[reg]["all_backward"].keys():
        x, y = zip(*enumerate(plot_points[reg]["all_backward"][lam]))
        x = 20 * np.array(x)
        ax.plot(x, y, label=lam)

    plt.legend()
    plt.title("Reg: %s" % reg)
    plt.xlabel("Training")
    plt.ylabel("Backward NFE")
    plt.savefig("%s/%s_all_backward.png" % (dir, reg))
