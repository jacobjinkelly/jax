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

dirname = "2019-11-10-10-09-13"
results_path = "%s/results.txt" % dirname

file = open(results_path, "r")

lines = file.readlines()

# get inds for one when new training starts
inds = []
for i, line in enumerate(lines):
    if line.startswith("Reg:"):
        inds.append(i)

# parse data
plot_points = {}
for ind, next_ind in zip(inds, inds[1:] + [len(lines)]):
    # get hyperparms for training
    split = lines[ind][len("Reg: "):].split("Lambda ")
    reg, lam = split[0][:-1], float(split[1][:-1])

    # add to dict
    if reg not in plot_points:
        plot_points[reg] = {}
        plot_points[reg]["lam"] = []            # lam values for this reg method
        plot_points[reg]["loss"] = []           # final loss
        plot_points[reg]["all_loss"] = {}       # loss throughout training
        plot_points[reg]["all_r0"] = {}         # r0 throughout training
        plot_points[reg]["all_r1"] = {}         # r1 throughout training
        plot_points[reg]["r0"] = []             # final r0
        plot_points[reg]["r1"] = []             # final r1
        plot_points[reg]["all_forward"] = {}    # forward NFE throughout training
        plot_points[reg]["all_backward"] = {}   # backward NFE throughout training
        plot_points[reg]["forward"] = []        # avg forward NFE throughout training
        plot_points[reg]["backward"] = []       # avg backward NFE throughout training

    plot_points[reg]["lam"].append(lam)
    data = lines[ind + 1:next_ind]

    size = 1
    final_losses = data[::-3][:size]
    final_loss = np.mean([float(line.split(" | ")[2].split("Loss ")[-1][:-1]) for line in final_losses])
    plot_points[reg]["loss"].append(final_loss)

    nfe = []
    losses = []
    grad_loss = []
    grad_reg = []
    r0 = []
    r1 = []
    reg_scale = lam if lam else 1
    for line in data:
        if line.startswith("Iter"):
            losses.append(float(line.split(" | ")[2].split("Loss ")[-1][:-1]))
            r0.append(float(line.split(" | ")[3].split("r0 ")[-1][:-1]) / reg_scale)
            r1.append(float(line.split(" | ")[4].split("r1 ")[-1][:-1]) / reg_scale)
        else:
            nfe.append(line)
    plot_points[reg]["all_loss"][lam] = losses
    plot_points[reg]["all_r0"][lam] = r0
    plot_points[reg]["all_r1"][lam] = r1

    plot_points[reg]["r0"].append(r0[-1])
    plot_points[reg]["r1"].append(r1[-1])

    forward = nfe[0::2]
    backward = nfe[1::2]
    for i, point in enumerate(forward):
        forward[i] = int(point.split("forward NFE: ")[-1][:-1])
    for i, point in enumerate(backward):
        backward[i] = int(point.split("backward NFE: ")[-1][:-1])
    plot_points[reg]["all_forward"][lam] = forward
    plot_points[reg]["all_backward"][lam] = backward
    plot_points[reg]["forward"].append(np.mean(forward))
    plot_points[reg]["backward"].append(np.mean(backward))


def pareto_plot_nfe(nfe_type, xaxis, lam_slice):
    """
    Create pareto plot.
    """
    for reg in plot_points:
        fig, ax = plt.subplots()
        argsort_lam = np.argsort(plot_points[reg]["lam"])[lam_slice]
        x = np.array(plot_points[reg][nfe_type])[argsort_lam]
        y = np.array(plot_points[reg]["loss"])[argsort_lam]
        anno = np.array(plot_points[reg]["lam"])[argsort_lam]
        ax.plot(x, y, "-o", label=str(reg))
        for i, txt in enumerate(anno):
            ax.annotate(txt, (x[i], y[i]))

        plt.xlabel(xaxis)
        plt.ylabel("Loss")
        start_str = "{:02d}".format(0) if lam_slice.start is None else "{:02d}".format(lam_slice.start)
        # TODO: hardcoded 25
        stop_str = "{:02d}".format(25) if lam_slice.stop is None else "{:02d}".format(lam_slice.stop)
        plt.savefig("%s/pareto_method_%s_axis_%s_start_%s_stop_%s_step_%s.png" %
                    (dirname, reg, nfe_type, start_str, stop_str, lam_slice.step))
        plt.clf()
        plt.close(fig)


def pareto_plot_reg(reg_type, xaxis, lam_slice):
    """
    Create pareto plot.
    """
    for reg in plot_points:
        fig, ax = plt.subplots()
        argsort_lam = np.argsort(plot_points[reg]["lam"])[lam_slice]
        x = np.array(plot_points[reg][reg_type])[argsort_lam]
        y = np.array(plot_points[reg]["loss"])[argsort_lam]
        anno = np.array(plot_points[reg]["lam"])[argsort_lam]
        if reg != "none":
            x = x / anno

        if lam_slice.start is None and lam_slice.stop is None and reg != "none":
            print("============reg_type: %s, xaxis: %s, reg: %s============" % (reg_type, xaxis, reg))
            for a, b, c in zip(*(x, y, anno)):
                print(a, b, c),

        ax.plot(x, y, "-o", label=str(reg))
        for i, txt in enumerate(anno):
            ax.annotate(txt, (x[i], y[i]))

        plt.xlabel(xaxis)
        plt.ylabel("Loss")
        start_str = "{:02d}".format(0) if lam_slice.start is None else "{:02d}".format(lam_slice.start)
        # TODO: hardcoded 25
        stop_str = "{:02d}".format(25) if lam_slice.stop is None else "{:02d}".format(lam_slice.stop)
        plt.savefig("%s/pareto_method_%s_axis_%s_start_%s_stop_%s_step_%s.png" %
                    (dirname, reg, reg_type, start_str, stop_str, lam_slice.step))
        plt.clf()
        plt.close(fig)


def moving_average(a, n):
    """
    Calculate moving average over window of size n.
    Credit: stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[(n - 1):] / n


window_size = 20


def train_plot(yaxis_val, yaxis_name, lam_slice):
    """
    Plot progression over training.
    """
    for reg in plot_points:
        fig, ax = plt.subplots()
        lams = list(plot_points[reg][yaxis_val].keys())[lam_slice[reg]]
        for lam in lams:
            x, y = zip(*enumerate(moving_average(plot_points[reg][yaxis_val][lam], window_size)))
            x = np.array(x) + window_size
            ax.plot(x, y, label=lam)

        plt.legend()
        plt.title("Reg: %s" % reg)
        plt.xlabel("Training")
        plt.ylabel(yaxis_name)
        start_str = "{:02d}".format(0) if lam_slice[reg].start is None else "{:02d}".format(lam_slice[reg].start)
        # TODO: hardcoded 25
        stop_str = "{:02d}".format(25) if lam_slice[reg].stop is None else "{:02d}".format(lam_slice[reg].stop)
        plt.savefig("%s/train_%s_%s_start_%s_stop_%s_step_%s.png" %
                    (dirname, reg, yaxis_val, start_str, stop_str, lam_slice[reg].step))
        plt.clf()
        plt.close(fig)


def comp_train_plot(yaxis_val, yaxis_name, lam_slice):
    """
    Compare metric evolution over training on diff methods.
    """
    fig, ax = plt.subplots()
    for i, reg in enumerate(plot_points):
        lams = list(plot_points[reg][yaxis_val].keys())[lam_slice[reg]]
        for lam in lams:
            x, y = zip(*enumerate(moving_average(plot_points[reg][yaxis_val][lam], window_size)))
            x = np.array(x) + window_size
            ax.plot(x, y, label=str(reg) + " " + str(lam))

    plt.legend()
    plt.xlabel("Training")
    plt.ylabel(yaxis_name)
    start_str = "{:02d}".format(0) if lam_slice["r0"].start is None else "{:02d}".format(lam_slice["r0"].start)
    # TODO: hardcoded 25
    stop_str = "{:02d}".format(25) if lam_slice["r0"].stop is None else "{:02d}".format(lam_slice["r0"].stop)
    plt.savefig("%s/comp_train_%s_start_%s_stop_%s_step_%s.png" %
                (dirname, yaxis_val, start_str, stop_str, lam_slice["r0"].step))
    plt.clf()
    plt.close(fig)


# =================================================== PLOT DYNAMICS ===================================================

def dynamics(lam_slice):
    """
    Plot dynamics
    """

    import jax
    import jax.numpy as np

    D = 1
    start_points = np.linspace(-3, 3, num=20)
    DATA_SIZE = len(start_points)
    TOTAL_TIME_POINTS = 1000
    REGS = ['r0', 'r1']
    NUM_REGS = len(REGS)

    dim_fns = {"x^3": lambda x: x ** 3}

    true_y0 = np.repeat(np.expand_dims(start_points, axis=1), D, axis=1)  # (DATA_SIZE, D)
    total_t = np.linspace(0., 1., num=TOTAL_TIME_POINTS)  # (TOTAL_TIME_POINTS, )

    r = np.zeros((TOTAL_TIME_POINTS, DATA_SIZE, 1))
    allr = np.zeros((TOTAL_TIME_POINTS, DATA_SIZE, NUM_REGS))
    true_y_t_r_allr = np.concatenate((np.repeat(np.expand_dims(true_y0, axis=0), TOTAL_TIME_POINTS, axis=0),
                                      np.expand_dims(
                                          np.tile(total_t, (DATA_SIZE, 1)).T,
                                          axis=2),
                                      r,
                                      allr),
                                     axis=2)

    # parse_args.batch_time * parse_args.data_size * (D + 2 + NUM_REGS) |->
    #                                       (parse_args.batch_time, parse_args.data_size, D + 2 + NUM_REGS)
    _, ravel_true_y_t_r_allr = ravel_pytree(true_y_t_r_allr)

    # set up input
    r0 = np.zeros((DATA_SIZE, 1))
    allr = np.zeros((DATA_SIZE, NUM_REGS))
    true_y0_t_r0_allr = np.concatenate((true_y0,
                                        np.expand_dims(
                                            np.repeat(total_t[0], DATA_SIZE), axis=1),
                                        r0,
                                        allr), axis=1)

    # parse_args.data_size * (D + 2 + NUM_REGS) |-> (parse_args.data_size, D + 2 + NUM_REGS)
    flat_true_y0_t_r0_allr, ravel_true_y0_t_r0_allr = ravel_pytree(true_y0_t_r0_allr)

    # set up MLP
    init_random_params, predict = stax.serial(
        Dense(50), Tanh,
        Dense(D)
    )

    _, init_params = init_random_params(key, (-1, D + 1))
    _, ravel_params = ravel_pytree(init_params)

    @jax.jit
    def test_reg_dynamics(y_t_r_allr, t, *args):
        """
        Augmented dynamics to implement regularization. (on test)
        """

        flat_params = args
        params = ravel_params(np.array(flat_params))

        # separate out state from augmented
        # only difference between this and reg_dynamics is
        # ravelling over datasize instead of batch size
        y_t_r_allr = ravel_true_y0_t_r0_allr(y_t_r_allr)
        y_t = y_t_r_allr[:, :D + 1]
        y = y_t[:, :-1]

        predictions_y = predict(params, y_t)
        predictions = np.concatenate((predictions_y,
                                      np.ones((DATA_SIZE, 1))),
                                     axis=1)

        r0 = np.sum(y ** 2, axis=1) ** 0.5
        r1 = np.sum(predictions_y ** 2, axis=1) ** 0.5
        if reg == "r0":
            regularization = r0
        elif reg == "r1":
            regularization = r1
        else:
            regularization = np.zeros(DATA_SIZE)

        pred_reg = np.concatenate((predictions,
                                   np.expand_dims(regularization, axis=1),
                                   np.expand_dims(r0, axis=1),
                                   np.expand_dims(r1, axis=1)),
                                  axis=1)
        flat_pred_reg = np.reshape(pred_reg, (-1,))
        return flat_pred_reg

    for reg in plot_points:
        for lam_rank, lam in enumerate(sorted(plot_points[reg]["lam"][lam_slice[reg]])):
            num_epochs = 5
            iters_per_epoch = 1000

            for itr in range(iters_per_epoch, num_epochs * iters_per_epoch + 1, iters_per_epoch):

                # load params
                param_filename = "%s/reg_%s_lam_%.4e_%d_fargs.pickle" % (dirname, reg, lam, itr)
                param_file = open(param_filename, "rb")
                params = pickle.load(param_file)
                fargs = params

                pred_y = ravel_true_y_t_r_allr(
                    odeint(test_reg_dynamics, flat_true_y0_t_r0_allr, total_t, *fargs)[0])[:, :, :D]

                for i, fn_name in enumerate(dim_fns):

                    fig, ax = plt.subplots()

                    for data_point in range(DATA_SIZE):

                        x, y = total_t, pred_y[:, data_point, i]
                        y0 = start_points[data_point]
                        y1 = y[-1]
                        t1 = dim_fns[fn_name](y0)
                        ax.plot(x, y, label="%.2f |-> %.2f (%.2f)" % (y0, y1, t1))

                    plt.legend()
                    plt.title("Reg: %s, Fn: %s, Lam: %.4e" % (reg, fn_name, lam))
                    plt.xlabel("Time")
                    plt.ylabel("state")
                    figname = "{}/dyn_reg_{}_fn_{}_lam_rank_{:02d}_lam_{:4e}_{:04d}.png"\
                        .format(dirname, reg, fn_name, lam_rank, lam, itr)
                    plt.savefig(figname)
                    plt.clf()
                    plt.close(fig)

                param_file.close()


if __name__ == "__main__":
    # starts = [0, 5, 10, 15, 20]
    # for start in starts:
    #     if start == 0:
    #         start = None
    #     lam_slice = slice(start, None, None)
    #
    #     pareto_plot_reg("r0", "r0", lam_slice)
    #     pareto_plot_reg("r1", "r1", lam_slice)
    #
    #     pareto_plot_nfe("forward", "Forward NFE", lam_slice)
    #     pareto_plot_nfe("backward", "Backward NFE", lam_slice)
    #
    # ends = [5, 10, 15, 20, 25]
    # for end in ends:
    #     lam_slice = slice(None, end, None)
    #
    #     pareto_plot_reg("r0", "r0", lam_slice)
    #     pareto_plot_reg("r1", "r1", lam_slice)
    #
    #     pareto_plot_nfe("forward", "Forward NFE", lam_slice)
    #     pareto_plot_nfe("backward", "Backward NFE", lam_slice)
    #
    dyn_lam_slice = {
        "none": slice(None, None, None),
        "r0": slice(15, 20, None),
        "r1": slice(15, 20, None)
    }

    dynamics(dyn_lam_slice)
    #
    # lams_slices = [slice(None, 5, None), slice(15, 20, None)]
    # for lam_slice in lams_slices:
    #     lam_slice = {
    #         "none": slice(None, None, None),
    #         "r0": lam_slice,
    #         "r1": lam_slice
    #     }
    #
    #     train_plot("all_forward", "Forward NFE", lam_slice)
    #     train_plot("all_backward", "Backward NFE", lam_slice)
    #     train_plot("all_loss", "Loss", lam_slice)
    #     train_plot("all_r0", "r0", lam_slice)
    #     train_plot("all_r1", "r1", lam_slice)
    #
    #     comp_train_plot("all_r0", "r0", lam_slice)
    #     comp_train_plot("all_r1", "r1", lam_slice)
