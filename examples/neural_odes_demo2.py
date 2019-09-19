"""
Neural ODEs in Jax.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import pickle

import jax
from jax.experimental import stax, optimizers
from jax.experimental.stax import Dense, Tanh
from jax.experimental.ode import odeint, grad_odeint
from jax import random, grad
import jax.numpy as np
from jax.config import config
from jax.flatten_util import ravel_pytree

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=20)
parser.add_argument('--lam', type=float, default=1)
parser.add_argument('--reg', type=str, choices=['none', 'weight', 'state', 'dynamics'], default='none')
parser.add_argument('--test_freq', type=int, default=1)
parser.add_argument('--save_freq', type=int, default=100)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
parse_args = parser.parse_args()


config.update('jax_enable_x64', True)

key = random.PRNGKey(0)

true_y0 = np.repeat(np.expand_dims(np.linspace(-3, 3, parse_args.data_size), axis=1), 2, axis=1)  # (N, D)
true_y1 = np.concatenate((np.expand_dims(true_y0[:, 0] ** 2, axis=1),
                          np.expand_dims(true_y0[:, 1] ** 3, axis=1)),
                         axis=1)
true_y = np.concatenate((np.expand_dims(true_y0, axis=0),
                        np.expand_dims(true_y1, axis=0)),
                        axis=0)  # (T, N, D)
t = np.array([0., 25.])  # (T)


def get_batch():
    """
    Get batch.
    """
    global key
    new_key, subkey = random.split(key)
    key = new_key
    s = random.shuffle(subkey, np.arange(parse_args.data_size, dtype=np.int64))[:parse_args.batch_size]
    batch_y0 = true_y0[s]       # (M, D)
    batch_t = t                 # (T)
    batch_y = true_y[:, s, :]   # (T, M, D)
    return batch_y0, batch_t, batch_y


def run(reg, lam):
    """
    Run the neural ODEs method.
    """
    print("Reg: %s Lambda %.4e" % (reg, lam))
    ii = 0

    # set up MLP
    init_random_params, predict = stax.serial(
        Dense(50), Tanh,
        Dense(2)
    )

    output_shape, init_params = init_random_params(key, (-1, 3))
    assert output_shape == (-1, 2)

    flat_params, ravel_params = ravel_pytree(init_params)
    batch_y0, batch_t, batch_y = get_batch()

    r0 = np.zeros((parse_args.batch_size, 1))
    batch_y0_r0 = np.concatenate((batch_y0, r0), axis=1)
    _, ravel_batch_y0_r0 = ravel_pytree(batch_y0_r0)

    _, ravel_batch_y0 = ravel_pytree(batch_y0)

    batch_y0_t = np.concatenate((batch_y0,
                                 np.expand_dims(
                                     np.repeat(batch_t[0], parse_args.batch_size),
                                     axis=1)
                                 ),
                                axis=1)
    _, ravel_batch_y0_t = ravel_pytree(batch_y0_t)

    batch_y0_t_r0 = np.concatenate((batch_y0_t, r0), axis=1)
    _, ravel_batch_y0_t_r0 = ravel_pytree(batch_y0_t_r0)

    r = np.zeros((parse_args.batch_time, parse_args.batch_size, 1))
    batch_y_r = np.concatenate((batch_y, r), axis=2)
    _, ravel_batch_y_r = ravel_pytree(batch_y_r)

    batch_y_t_r = np.concatenate((batch_y,
                                  np.expand_dims(
                                      np.tile(batch_t, (parse_args.batch_size, 1)).T,
                                      axis=2),
                                  r),
                                 axis=2)
    _, ravel_batch_y_t_r = ravel_pytree(batch_y_t_r)

    _, ravel_batch_y = ravel_pytree(batch_y)
    fargs = flat_params

    def dynamics(y_t, t, *args):
        """
        Time-augmented dynamics.
        """

        flat_params = args
        params = ravel_params(np.array(flat_params))

        y_t = ravel_batch_y0_t(y_t)

        predictions_y = predict(params, y_t)
        predictions = np.concatenate((predictions_y,
                                      np.ones((parse_args.batch_size, 1))),
                                     axis=1)

        flat_predictions, _ = ravel_pytree(predictions)
        return flat_predictions

    def reg_dynamics(y_t_r, t, *args):
        """
        Augmented dynamics to implement regularization.
        """

        flat_params = args
        params = ravel_params(np.array(flat_params))

        # separate out state from augmented
        y_r = ravel_batch_y0_t_r0(y_t_r)
        y_t, r = y_r[:, :-1], y_r[:, -1]
        y = y_t[:, :-1]

        predictions_y = predict(params, y_t)
        predictions = np.concatenate((predictions_y,
                                      np.ones((parse_args.batch_size, 1))),
                                     axis=1)

        if reg == "state":
            regularization = np.sum(y ** 2, axis=1) ** 0.5
        elif reg == "dynamics":
            regularization = np.sum(predictions_y ** 2, axis=1) ** 0.5
        else:
            regularization = np.zeros(parse_args.batch_size)

        regularization = np.expand_dims(regularization, axis=1)
        pred_reg = np.concatenate((predictions, regularization), axis=1)
        flat_pred_reg, _ = ravel_pytree(pred_reg)
        return flat_pred_reg

    def test_reg_dynamics(y_r, t, *args):
        """
        Augmented dynamics to implement regularization. (on test)
        """

        flat_params = args
        params = ravel_params(np.array(flat_params))

        # separate out state from augmented
        y_r = ravel_true_y0_t_r0(y_r)
        y_t, r = y_r[:, :-1], y_r[:, -1]
        y = y_t[:, :-1]

        predictions_y = predict(params, y_t)
        predictions = np.concatenate((predictions_y,
                                      np.ones((parse_args.data_size, 1))),
                                     axis=1)

        if reg == "state":
            regularization = np.sum(y ** 2, axis=1) ** 0.5
        elif reg == "dynamics":
            regularization = np.sum(predictions_y ** 2, axis=1) ** 0.5
        else:
            regularization = np.zeros(parse_args.data_size)

        regularization = np.expand_dims(regularization, axis=1)
        pred_reg = np.concatenate((predictions, regularization), axis=1)
        flat_pred_reg, _ = ravel_pytree(pred_reg)
        return flat_pred_reg

    @jax.jit
    def total_loss_fun(pred_y_r, target):
        """
        Loss function.
        """
        pred, reg = pred_y_r[:, :, :2], pred_y_r[:, :, 2]
        return np.mean(np.abs(pred - target)) + lam * np.mean(reg)

    @jax.jit
    def loss_fun(pred, target):
        """
        Mean absolute error.
        """
        return np.mean(np.abs(pred - target))

    ode_vjp = grad_odeint(dynamics, fargs)
    reg_ode_vjp = grad_odeint(reg_dynamics, fargs)
    grad_loss_fun = jax.jit(grad(total_loss_fun))

    opt_init, opt_update, get_params = optimizers.rmsprop(step_size=1e-3, gamma=0.99)

    opt_state = opt_init(fargs)
    for itr in range(1, parse_args.niters + 1):
        # get the next batch and pack it
        batch_y0, batch_t, batch_y = get_batch()
        batch_y0_t = np.concatenate((batch_y0,
                                    np.expand_dims(
                                        np.repeat(batch_t[0], parse_args.batch_size),
                                        axis=1)
                                     ),
                                    axis=1)
        flat_batch_y0_t, _ = ravel_pytree(batch_y0_t)
        r0 = np.zeros((parse_args.batch_size, 1))
        batch_y0_t_r0 = np.concatenate((batch_y0_t, r0), axis=1)
        flat_batch_y0_t_r0, _ = ravel_pytree(batch_y0_t_r0)

        fargs = get_params(opt_state)

        # integrate ODE (including reg) and count NFE (on unreg)
        tmp_pred_y_t, nfe = odeint(dynamics, fargs, flat_batch_y0_t, batch_t, atol=1e-8, rtol=1e-8)
        pred_y_t_r, _ = odeint(reg_dynamics, fargs, flat_batch_y0_t_r0, batch_t, atol=1e-8, rtol=1e-8)
        print("forward NFE: %d" % nfe)

        _, ravel_pred_y_t = ravel_pytree(tmp_pred_y_t)
        pred_y_t = ravel_pred_y_t(ravel_batch_y_t_r(pred_y_t_r)[:, :, :-1])
        _, ravel_pred_y_t_r = ravel_pytree(pred_y_t_r)

        # TODO: time in loss?
        loss_grad = grad_loss_fun(ravel_batch_y_t_r(pred_y_t_r), batch_y)

        # integrate adjoint ODE and count NFE
        nfe = ode_vjp(ravel_pred_y_t(loss_grad[:, :, :-1]), pred_y_t, batch_t)[-1]
        params_grad = reg_ode_vjp(ravel_pred_y_t_r(loss_grad), pred_y_t_r, batch_t)[:-1][3]
        print("backward NFE: %d" % nfe)

        opt_state = opt_update(itr, params_grad, opt_state)

        if itr % parse_args.test_freq == 0:
            fargs = get_params(opt_state)
            r0 = np.zeros((parse_args.data_size, 1))
            true_y0_t_r0 = np.concatenate((true_y0,
                                           np.expand_dims(
                                               np.repeat(t[0], parse_args.data_size), axis=1),
                                           r0), axis=1)
            flat_true_y0_t_r0, ravel_true_y0_t_r0 = ravel_pytree(true_y0_t_r0)
            pred_y_t_r, _ = odeint(test_reg_dynamics, fargs, flat_true_y0_t_r0, t, atol=1e-8, rtol=1e-8)

            pred_y0_t_r, pred_y1_t_r = pred_y_t_r[0, :], pred_y_t_r[1, :]
            pred_y0_t_r, pred_y1_t_r = ravel_true_y0_t_r0(pred_y0_t_r), ravel_true_y0_t_r0(pred_y1_t_r)
            pred_y = np.concatenate((np.expand_dims(pred_y0_t_r[:, :-2], axis=0),
                                     np.expand_dims(pred_y1_t_r[:, :-2], axis=0)),
                                    axis=0)
            pred_y_r = np.concatenate((np.expand_dims(pred_y0_t_r[:, [0, 1, -1]], axis=0),
                                       np.expand_dims(pred_y1_t_r[:, [0, 1, -1]], axis=0)),
                                      axis=0)

            loss = loss_fun(pred_y, true_y)
            total_loss = total_loss_fun(pred_y_r, true_y)
            print('Iter {:04d} | Total (Regularized) Loss {:.6f} | Loss {:.6f} | Regularization {:.6f}'.
                  format(itr, total_loss, loss, total_loss - loss))
            eprint('Iter {:04d} | Total (Regularized) Loss {:.6f} | Loss {:.6f} | Regularization {:.6f}'.
                  format(itr, total_loss, loss, total_loss - loss))
            ii += 1

        if itr % parse_args.save_freq == 0:
            param_filename = "%s/reg_%s_lam_%.4e_%d_fargs.pickle" % (dirname, reg, lam, itr)
            outfile = open(param_filename, "wb")
            pickle.dump(fargs, outfile)
            outfile.close()


if __name__ == "__main__":
    within_python = True
    if within_python:
        import os
        import datetime
        import sys

        dirname = datetime.datetime.now().strftime("%F-%H-%M-%S")
        os.mkdir(dirname)
        filename = "%s/results.txt" % dirname
        results_file = open(filename, "w")
        sys.stdout = results_file

        def eprint(*args, **kwargs):
            """
            Print to stderr.
            """
            print(*args, file=sys.stderr, **kwargs)

        hyperparams = {
                       "none": [0],
                       # "state": np.linspace(0, 2 * 0.057, 5),
                       # "dynamics": np.linspace(0, 2 * 0.265, 5)
                       }
        for reg in hyperparams.keys():
            for lam in hyperparams[reg]:
                eprint("Reg: %s\tLambda %.4e" % (reg, lam))
                run(str(reg), lam)
        results_file.close()
    else:
        run(parse_args.reg, parse_args.lam)
