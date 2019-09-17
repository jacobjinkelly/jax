"""
Neural ODEs in Jax.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

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
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=20)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lam', type=float, default=1)
parser.add_argument('--reg', type=str, choices=['none', 'weight', 'state', 'dynamics'], default='none')
parser.add_argument('--test_freq', type=int, default=1)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
parse_args = parser.parse_args()


config.update('jax_enable_x64', True)

key = random.PRNGKey(0)

true_y0 = np.array([2., 0.])  # (D,)
t = np.linspace(0., 25., parse_args.data_size)
true_A = np.array([[-0.1, 2.0], [-2.0, -0.1]])


def true_func(y, t):
    """
    True dynamics function.
    """
    return np.matmul(y ** 3, true_A)


true_y, _ = odeint(true_func, (), true_y0, t, atol=1e-8, rtol=1e-8)

# expand to batched version for use in testing
true_y0 = np.expand_dims(true_y0, axis=0)


def get_batch():
    """
    Get batch.
    """
    global key
    new_key, subkey = random.split(key)
    key = new_key
    s = random.shuffle(subkey, np.arange(parse_args.data_size - parse_args.batch_time, dtype=np.int64))[:parse_args.batch_size]
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:parse_args.batch_time]  # (T)
    batch_y = np.stack([true_y[s + i] for i in range(parse_args.batch_time)], axis=0)  # (T, M, D)
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

    output_shape, init_params = init_random_params(key, (-1, 2))
    assert output_shape == (-1, 2)

    flat_params, ravel_params = ravel_pytree(init_params)
    batch_y0, _, batch_y = get_batch()

    r0 = np.zeros((parse_args.batch_size, 1))
    batch_y0_r0 = np.concatenate((batch_y0, r0), axis=1)
    _, ravel_batch_y0_r0 = ravel_pytree(batch_y0_r0)

    _, ravel_batch_y0 = ravel_pytree(batch_y0)

    r = np.zeros((parse_args.batch_time, parse_args.batch_size, 1))
    batch_y_r = np.concatenate((batch_y, r), axis=2)
    _, ravel_batch_y_r = ravel_pytree(batch_y_r)

    _, ravel_batch_y = ravel_pytree(batch_y)
    fargs = flat_params

    def reg_dynamics(y_r, t, *args):
        """
        Augmented dynamics to implement regularization.
        """

        flat_params = args
        params = ravel_params(np.array(flat_params))

        # separate out state from augmented
        y_r = ravel_batch_y0_r0(y_r)
        y, r = y_r[:, :2], y_r[:, 2]

        predictions = predict(params, y ** 3)

        if reg == "state":
            regularization = np.sum(y ** 2, axis=1) ** 0.5
        elif reg == "dynamics":
            regularization = np.sum(predictions ** 2, axis=1) ** 0.5
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

    def total_loss_fun(pred_y_r, target):
        """
        Loss function.
        """
        pred, reg = pred_y_r[:, :, :2], pred_y_r[:, :, 2]
        return np.mean(np.abs(pred - target)) + lam * np.mean(reg)

    def loss_fun(pred, target):
        """
        Mean absolute error.
        """
        return np.mean(np.abs(pred - target))

    ode_vjp = grad_odeint(reg_dynamics, fargs)
    grad_loss_fun = grad(total_loss_fun)

    opt_init, opt_update, get_params = optimizers.rmsprop(step_size=1e-3, gamma=0.99)

    opt_state = opt_init(fargs)
    for itr in range(1, parse_args.niters + 1):
        # get the next batch and pack it
        batch_y0, batch_t, batch_y = get_batch()
        r0 = np.zeros((parse_args.batch_size, 1))
        batch_y0_r0 = np.concatenate((batch_y0, r0), axis=1)
        flat_batch_y0_r0, _ = ravel_pytree(batch_y0_r0)

        fargs = get_params(opt_state)

        # integrate ODE and count NFE
        pred_y_r, nfe = odeint(reg_dynamics, fargs, flat_batch_y0_r0, batch_t, atol=1e-8, rtol=1e-8)
        print("forward NFE: %d" % nfe)

        _, ravel_pred_y_r = ravel_pytree(pred_y_r)

        loss_grad = grad_loss_fun(ravel_batch_y_r(pred_y_r), batch_y)

        # integrate adjoint ODE and count NFE
        result = ode_vjp(ravel_pred_y_r(loss_grad), pred_y_r, batch_t)
        total_grad, nfe = result[:-1], result[-1]
        print("backward NFE: %d" % nfe)

        params_grad = total_grad[3]

        opt_state = opt_update(itr, params_grad, opt_state)

        if itr % parse_args.test_freq == 0:
            fargs = get_params(opt_state)
            r0 = np.zeros((1, 1))
            true_y0_r0 = np.concatenate((true_y0, r0), axis=1)
            flat_true_y0_r0, ravel_true_y0_r0 = ravel_pytree(true_y0_r0)
            pred_y_r, _ = odeint(test_reg_dynamics, fargs, flat_true_y0_r0, t, atol=1e-8, rtol=1e-8)
            pred_y = pred_y_r[:, :2]
            loss = loss_fun(pred_y, true_y)
            total_loss = total_loss_fun(np.expand_dims(pred_y_r, axis=1), np.expand_dims(true_y, axis=1))
            print('Iter {:04d} | Total (Regularized) Loss {:.6f} | Loss {:.6f} | Regularization {:.6f}'.
                  format(itr, total_loss, loss, total_loss - loss))
            ii += 1


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

        hyperparams = {"none": [0],
                       # "state": np.linspace(0, 2 * 1.185, 10),
                       # "dynamics": np.linspace(0, 2 * 5.350, 10)
                       }
        for reg in hyperparams.keys():
            for lam in hyperparams[reg]:
                eprint("Reg: %s\tLambda %.4e" % (reg, lam))
                run(str(reg), lam)
        results_file.close()
    else:
        run(parse_args.reg, parse_args.lam)
