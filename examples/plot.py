"""
For parsing and plotting the results of neural_odes.py
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle

from jax.experimental import stax
from jax.experimental.stax import Dense, Tanh
from jax.experimental.ode import odeint
from jax import random
from jax.config import config
from jax.flatten_util import ravel_pytree

config.update('jax_enable_x64', True)

key = random.PRNGKey(0)

dirname = "2019-09-17-18-32-38"
results_path = "%s/results.txt" % dirname

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
        plot_points[reg]["all_loss"] = {}
        plot_points[reg]["all_forward"] = {}
        plot_points[reg]["all_backward"] = {}
        plot_points[reg]["forward"] = []
        plot_points[reg]["backward"] = []

    plot_points[reg]["lam"].append(lam)
    data = lines[ind + 1:next_ind]

    final_loss = float(data[-1].split(" | ")[-2].split("Loss ")[-1][:-1])
    plot_points[reg]["loss"].append(final_loss)

    nfe = []
    losses = []
    for line in data:
        if line.startswith("Iter"):
            losses.append(float(line.split(" | ")[-2].split("Loss ")[-1][:-1]))
        else:
            nfe.append(line)
    plot_points[reg]["all_loss"][lam] = losses

    forward = nfe[0::2]
    backward = nfe[1::2]
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
    for i, txt in enumerate(plot_points[reg]["lam"]):
        ax.annotate(txt, (x[i], y[i]))

plt.legend()
plt.xlabel("Forward NFE")
plt.ylabel("Loss")
plt.savefig("%s/forward.png" % dirname)
plt.clf()
plt.close(fig)

fig, ax = plt.subplots()
for reg in plot_points.keys():
    x, y = plot_points[reg]["backward"], plot_points[reg]["loss"]
    ax.plot(x, y, "-o", label=str(reg))
    # for i, txt in enumerate(losses[reg]["lam"]):
    #     ax.annotate(txt, (x[i], y[i]))

plt.legend()
plt.xlabel("Backward NFE")
plt.ylabel("Loss")
plt.savefig("%s/backward.png" % dirname)
plt.clf()
plt.close(fig)

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
    plt.savefig("%s/%s_all_forward.png" % (dirname, reg))
    plt.clf()
    plt.close(fig)


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
    plt.savefig("%s/%s_all_backward.png" % (dirname, reg))
    plt.clf()
    plt.close(fig)

for reg in plot_points.keys():
    fig, ax = plt.subplots()
    for lam in plot_points[reg]["all_loss"].keys():
        x, y = zip(*enumerate(plot_points[reg]["all_loss"][lam]))
        x = 20 * np.array(x)
        ax.plot(x, y, label=lam)

    plt.legend()
    plt.title("Reg: %s" % reg)
    plt.xlabel("Training")
    plt.ylabel("Loss")
    plt.savefig("%s/%s_loss.png" % (dirname, reg))
    plt.clf()
    plt.close(fig)

# plot the dynamics

import jax.numpy as np

true_y0 = np.array([2., 0.])  # (D,)
t = np.linspace(0., 25., 1000)
true_A = np.array([[-0.1, 2.0], [-2.0, -0.1]])


def true_func(y, t):
    """
    True dynamics function.
    """
    return np.matmul(y ** 3, true_A)


true_y, _ = odeint(true_func, (), true_y0, t, atol=1e-8, rtol=1e-8)

# expand to batched version for use in testing
true_y0 = np.expand_dims(true_y0, axis=0)

# set up input
r0 = np.zeros((1, 1))
true_y0_r0 = np.concatenate((true_y0, r0), axis=1)
flat_true_y0_r0, ravel_true_y0_r0 = ravel_pytree(true_y0_r0)

# set up MLP
init_random_params, predict = stax.serial(
    Dense(50), Tanh,
    Dense(2)
)

_, init_params = init_random_params(key, (-1, 2))

_, ravel_params = ravel_pytree(init_params)


def test_reg_dynamics(y_r, t, *args):
    """
    Augmented dynamics to implement regularization. (on test)
    """

    flat_params = args
    params = ravel_params(np.array(flat_params))

    # separate out state from augmented
    y_r = ravel_true_y0_r0(y_r)
    y, r = y_r[:, :2], y_r[:, 2]

    predictions = predict(params, y ** 3)

    if reg == "state":
        regularization = np.sum(y ** 2, axis=1) ** 0.5
    elif reg == "dynamics":
        regularization = np.sum(predictions ** 2, axis=1) ** 0.5
    else:
        regularization = np.zeros(1)

    regularization = np.expand_dims(regularization, axis=1)
    pred_reg = np.concatenate((predictions, regularization), axis=1)
    flat_pred_reg, _ = ravel_pytree(pred_reg)
    return flat_pred_reg


for reg in plot_points.keys():
    for lam in plot_points[reg]["lam"]:
        for itr in range(100, 1001, 100):
            # load params
            param_filename = "%s/reg_%s_lam_%.4e_%d_fargs.pickle" % (dirname, reg, lam, itr)
            param_file = open(param_filename, "rb")
            params = pickle.load(param_file)
            fargs = params

            # get predictions
            pred_y_r, _ = odeint(test_reg_dynamics, fargs, flat_true_y0_r0, t, atol=1e-8, rtol=1e-8)

            # plot each dim
            fig, ax = plt.subplots()
            for i in range(pred_y_r.shape[1] - 1):
                pred_y = pred_y_r[:, i]
                x, y = t, pred_y
                ax.plot(x, y, label=i)

            x, y = t, pred_y_r[:, -1]
            ax.plot(x, y, label="reg")
            plt.legend()
            plt.title("Reg: %s" % reg)
            plt.xlabel("Time")
            plt.ylabel("state")
            figname = "%s/reg_%s_lam_%.4e_%d.png" % (dirname, reg, lam, itr)
            plt.savefig(figname)
            plt.clf()
            plt.close(fig)

            param_file.close()
