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

dirname = "tmp6"
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
        plot_points[reg]["lam"] = []
        plot_points[reg]["loss"] = []
        plot_points[reg]["all_loss"] = {}
        plot_points[reg]["all_grad_loss"] = {}
        plot_points[reg]["all_grad_reg"] = {}
        plot_points[reg]["all_r0"] = {}
        plot_points[reg]["all_r1"] = {}
        plot_points[reg]["r0"] = []
        plot_points[reg]["r1"] = []
        plot_points[reg]["all_forward"] = {}
        plot_points[reg]["all_backward"] = {}
        plot_points[reg]["forward"] = []
        plot_points[reg]["backward"] = []

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
    for line in data:
        if line.startswith("Iter"):
            losses.append(float(line.split(" | ")[2].split("Loss ")[-1][:-1]))
            grad_loss.append(float(line.split(" | ")[3].split("Loss Grad Norm ")[-1][:-1]))
            grad_reg.append(float(line.split(" | ")[4].split("Reg Grad Norm ")[-1][:-1]))
            r0.append(float(line.split(" | ")[5].split("r0 ")[-1][:-1]))
            r1.append(float(line.split(" | ")[6].split("r1 ")[-1][:-1]))
        else:
            nfe.append(line)
    plot_points[reg]["all_loss"][lam] = losses
    plot_points[reg]["all_grad_loss"][lam] = grad_loss
    plot_points[reg]["all_grad_reg"][lam] = grad_reg
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
            # if i == 0 or i == len(anno) - 1:
            ax.annotate(txt, (x[i], y[i]))

        plt.xlabel(xaxis)
        plt.ylabel("Loss")
        start_str = "{:02d}".format(0) if lam_slice.start is None else "{:02d}".format(lam_slice.start)
        plt.savefig("%s/pareto_method_%s_axis_%s_start_%s_stop_%s_step_%s.png" %
                    (dirname, reg, nfe_type, start_str, lam_slice.stop, lam_slice.step))
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
            # if i == 0 or i == len(anno) - 1:
            ax.annotate(txt, (x[i], y[i]))

        plt.xlabel(xaxis)
        plt.ylabel("Loss")
        start_str = "{:02d}".format(0) if lam_slice.start is None else "{:02d}".format(lam_slice.start)
        plt.savefig("%s/pareto_method_%s_axis_%s_start_%s_stop_%s_step_%s.png" %
                    (dirname, reg, reg_type, start_str, lam_slice.stop, lam_slice.step))
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


def train_plot(yaxis_val, yaxis_name):
    """
    Plot progression over training.
    """
    for reg in plot_points:
        fig, ax = plt.subplots()
        for lam in plot_points[reg][yaxis_val]:
            x, y = zip(*enumerate(moving_average(plot_points[reg][yaxis_val][lam], window_size)))
            x = np.array(x) + window_size
            ax.plot(x, y, label=lam)

        plt.legend()
        plt.title("Reg: %s" % reg)
        plt.xlabel("Training")
        plt.ylabel(yaxis_name)
        plt.savefig("%s/train_%s_%s.png" % (dirname, reg, yaxis_val))
        plt.clf()
        plt.close(fig)


def make_patch_spines_invisible(ax):
    """
    Helper method for plotting multiple y-axis.
    """
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def comp_train_plot(yaxis_val, yaxis_name):
    """
    Compare metric evolution over training on diff methods.
    """
    fig, ax = plt.subplots()
    # fig.subplots_adjust(right=0.75)
    for i, reg in enumerate(plot_points):
        # cur_ax = ax.twinx()
        # cur_ax.spines["right"].set_position(("axes", 1 + .2 * i))
        # make_patch_spines_invisible(cur_ax)
        # cur_ax.spines["right"].set_visible(True)
        for lam in plot_points[reg][yaxis_val]:
            x, y = zip(*enumerate(moving_average(plot_points[reg][yaxis_val][lam], window_size)))
            x = np.array(x) + window_size
            ax.plot(x, y, label=str(reg) + " " + str(lam))
            # cur_ax.plot(x, y, label=str(reg) + " " + str(lam))

    plt.legend()
    plt.xlabel("Training")
    plt.ylabel(yaxis_name)
    plt.savefig("%s/comp_train_%s.png" % (dirname, yaxis_val))
    plt.clf()
    plt.close(fig)


def comp_grad_norm_plot(yaxis_vals, yaxis_name):
    """
    Compare metric evolution over training on diff methods.
    """
    # fig.subplots_adjust(right=0.75)
    for i, reg in enumerate(plot_points):
        # cur_ax = ax.twinx()
        # cur_ax.spines["right"].set_position(("axes", 1 + .2 * i))
        # make_patch_spines_invisible(cur_ax)
        # cur_ax.spines["right"].set_visible(True)
        for lam in plot_points[reg]["lam"]:
            fig, ax = plt.subplots()
            for yaxis_val in yaxis_vals:
                x, y = zip(*enumerate(moving_average(plot_points[reg][yaxis_val][lam], window_size)))
                x = np.array(x) + window_size
                ax.plot(x, y, label=yaxis_val)
                # cur_ax.plot(x, y, label=str(reg) + " " + str(lam))

            plt.legend()
            plt.title(str(reg) + " " + str(lam))
            plt.xlabel("Training")
            plt.ylabel(yaxis_name)
            plt.savefig("%s/comp_grad_norm_%s.png" % (dirname, str(reg) + "_" + str(lam)))
            plt.clf()
            plt.close(fig)


# print some statistics on the gradient norms
for reg in plot_points:
    for lam in plot_points[reg]["all_grad_loss"]:
        loss_grad = np.array(plot_points[reg]["all_grad_loss"][lam])
        reg_grad = np.array(plot_points[reg]["all_grad_reg"][lam])

        if float(lam) != 0:
            reg_grad /= lam

        print(reg, lam)
        print("Loss Grad Norm (Min, Max, Mean): %.3f, %.3f, %.3f," %
              (np.min(loss_grad), np.max(loss_grad), np.mean(loss_grad)))
        print("Reg Grad Norm (Min, Max, Mean): %.3f, %.3f, %.3f," %
              (np.min(reg_grad), np.max(reg_grad), np.mean(reg_grad)))
        print("Average Ratio: %.3f" % np.mean(loss_grad / reg_grad))


# =================================================== PLOT DYNAMICS ===================================================

def dynamics():

    import jax.numpy as np

    D = 3
    start_points = np.array([-2])
    DATA_SIZE = len(start_points)
    TIME_POINTS = 2
    TOTAL_TIME_POINTS = 1000
    REGS = ['r0', 'r1']
    NUM_REGS = len(REGS)

    dim_fns = {"x^2": lambda x: x ** 2,
               "x^3": lambda x: x ** 3,
               "x^4": lambda x: x ** 4
               }

    true_y0 = np.repeat(np.expand_dims(start_points, axis=1), D, axis=1)  # (DATA_SIZE, D)
    true_y1 = np.concatenate((np.expand_dims(dim_fns["x^2"](true_y0[:, 0]), axis=1),
                              np.expand_dims(dim_fns["x^2"](true_y0[:, 1]), axis=1),
                              np.expand_dims(dim_fns["x^2"](true_y0[:, 2]), axis=1)),
                             axis=1)
    true_y = np.concatenate((np.expand_dims(true_y0, axis=0),
                            np.expand_dims(true_y1, axis=0)),
                            axis=0)  # (TIME_POINTS, DATA_SIZE, D)
    t = np.array([0., 25.])  # (TIME_POINTS, )
    total_t = np.linspace(0., 25., num=TOTAL_TIME_POINTS)  # (TOTAL_TIME_POINTS, )

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


    def test_reg_dynamics(y_t_r_allr, t, *args):
        """
        Augmented dynamics to implement regularization. (on test)
        """

        flat_params = args
        params = ravel_params(np.array(flat_params))

        # separate out state from augmented
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
        flat_pred_reg, _ = ravel_pytree(pred_reg)
        return flat_pred_reg

    for reg in plot_points:
        for lam in plot_points[reg]["lam"]:
            num_epochs = 50
            iters_per_epoch = 50
            for itr in range(iters_per_epoch, num_epochs * iters_per_epoch + 1, iters_per_epoch):
                # load params
                param_filename = "%s/reg_%s_lam_%.4e_%d_fargs.pickle" % (dirname, reg, lam, itr)
                param_file = open(param_filename, "rb")
                params = pickle.load(param_file)
                fargs = params

                pred_y_t_r_allr, _ = odeint(test_reg_dynamics, fargs, flat_true_y0_t_r0_allr, total_t, atol=1e-8, rtol=1e-8)

                pred_y_t_r_allr = ravel_true_y_t_r_allr(pred_y_t_r_allr)
                pred_y = pred_y_t_r_allr[:, :, :D]
                pred_y_t_r = pred_y_t_r_allr[:, :, :D + 2]

                for data_point in range(pred_y.shape[1]):
                    # plot each dim
                    fig, ax = plt.subplots()
                    for i, fn_name in enumerate(dim_fns):
                        x, y = total_t, pred_y[:, data_point, i]
                        ax.plot(x, y, label=fn_name)
                        xy = (x[-1], y[-1])
                        ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')

                    xy = (total_t[0], pred_y[:, 0, 0][0])
                    ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')

                    # plot r0
                    x, y = total_t, np.sum(pred_y_t_r[:, data_point, :D] ** 2, axis=1) ** 0.5
                    ax.plot(x, y, label="r0")

                    # plot r1
                    x, y = total_t, np.sum(predict(ravel_params(fargs), pred_y_t_r[:, data_point, :D + 1]) ** 2, axis=1) ** 0.5
                    ax.plot(x, y, label="r1")

                    plt.legend()
                    plt.title("Reg: %s" % reg)
                    plt.xlabel("Time")
                    plt.ylabel("state")
                    figname = "{}/reg_{}_lam_{:4e}_{:04d}.png".format(dirname, reg, lam, itr)
                    plt.savefig(figname)
                    plt.clf()
                    plt.close(fig)

                param_file.close()


def new_dynamics():

    import jax
    import jax.numpy as np

    D = 1
    start_points = np.linspace(-0.5, 0.5, num=10)
    DATA_SIZE = len(start_points)
    TOTAL_TIME_POINTS = 1000
    REGS = ['r0', 'r1']
    NUM_REGS = len(REGS)

    dim_fns = {"2x": lambda x: 2*x,
               # "x^3": lambda x: x ** 3,
               # "x^4": lambda x: x ** 4
               }

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

    plot_points = {
        "none": {
            "lam": [0]
        }
    }

    for reg in plot_points:
        for lam_rank, lam in enumerate(sorted(plot_points[reg]["lam"])):
            num_epochs = 100
            iters_per_epoch = 20

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
                    figname = "{}/new_dyn_reg_{}_fn_{}_lam_rank_{:02d}_lam_{:4e}_{:04d}.png"\
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


    new_dynamics()

    # train_plot("all_forward", "Forward NFE")
    # train_plot("all_backward", "Backward NFE")
    # train_plot("all_loss", "Loss")
    # train_plot("all_r0", "r0")
    # train_plot("all_r1", "r1")
    #
    # comp_train_plot("all_r0", "r0")
    # comp_train_plot("all_r1", "r1")
    # comp_train_plot("all_grad_loss", "Loss Grad Norm")
    # comp_train_plot("all_grad_reg", "Reg Grad Norm")
    #
    # comp_grad_norm_plot(["all_grad_loss", "all_grad_reg"], "Norm")
