"""
For parsing and plotting the results of neural_odes.py
"""
import numpy as np

dir = "2019-09-17-16-04-06"
results_path = "%s/results.txt" % dir

file = open(results_path, "r")

lines = file.readlines()

inds = []
for i, line in enumerate(lines):
    if line.startswith("Reg:"):
        inds.append(i)

losses = {}
for ind, next_ind in zip(inds, inds[1:] + [len(lines)]):
    split = lines[ind][len("Reg: "):].split("Lambda ")
    reg_type, lam = split[0][:-1], float(split[1][:-1])
    if reg_type == "none":
        continue
    if reg_type not in losses:
        losses[reg_type] = []

    data = lines[ind + 1:next_ind]
    loss_lines = []
    for i, line in enumerate(data):
        if line.startswith("Iter"):
            loss_lines.append(line)

    for line in loss_lines:
        split = line.split(" | ")
        loss, reg = float(split[-2].split("Loss ")[-1][:-1]), float(split[-1].split("Regularization ")[-1][:-1])
        losses[reg_type].append(loss / reg)

for reg in losses.keys():
    print("Reg: %s, Loss/Reg: %.3f" % (reg, np.mean(losses[reg])))
