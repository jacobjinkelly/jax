"""
For parsing and plotting the results of neural_odes.py
"""
import numpy as np

dir = "2019-09-19-17-35-06"
results_path = "%s/results.txt" % dir

file = open(results_path, "r")

lines = file.readlines()

data = lines[1:][2::3]

reg_data = {"r0": [],
            "r1": []
            }
for line in data:
    split = line.split(" | ")
    loss = float(split[-len(reg_data.keys()) - 1].split("Loss ")[-1][:-1])
    reg_line = split[-len(reg_data.keys()):]
    for reg_name, reg_seg in zip(reg_data.keys(), reg_line):

        reg = float(reg_seg.split(reg_name + " ")[-1][:-1])
        reg_data[reg_name].append(loss / reg)

for reg in reg_data:
    print("Reg: %s, Loss/Reg: %.3f" % (reg, np.mean(reg_data[reg])))
