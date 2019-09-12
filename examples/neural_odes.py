"""
Neural ODEs in Jax.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from jax.experimental import stax
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
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lam', type=float, default=1)
parser.add_argument('--test_freq', type=int, default=1)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()


config.update('jax_enable_x64', True)

key = random.PRNGKey(0)

true_y0 = np.array([2., 0.])  # (D,)
t = np.linspace(0., 25., args.data_size)
true_A = np.array([[-0.1, 2.0], [-2.0, -0.1]])


def true_func(y, t):
    """
    True dynamics function.
    """
    return np.matmul(y ** 3, true_A)


# TODO: no grad?
true_y = odeint(true_func, (), true_y0, t, atol=1e-8, rtol=1e-8)


def get_batch():
    """
    Get batch.
    """
    global key
    new_key, subkey = random.split(key)
    key = new_key
    s = random.shuffle(subkey, np.arange(args.data_size - args.batch_time, dtype=np.int64))[:args.batch_size]
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = np.stack([true_y[s + i] for i in range(args.batch_time)], axis=0)  # (T, M, D)
    return batch_y0, batch_t, batch_y


if __name__ == "__main__":

    ii = 0

    NFE_COUNT = [0]

    # set up MLP
    init_random_params, predict = stax.serial(
        Dense(50), Tanh,
        Dense(2)
    )

    output_shape, init_params = init_random_params(key, (-1, 2))
    assert output_shape == (-1, 2)

    flat_params, ravel_params = ravel_pytree(init_params)
    batch_y0, _, batch_y = get_batch()

    r0 = np.zeros((args.batch_size, 1))
    batch_y0_r0 = np.concatenate((batch_y0, r0), axis=1)
    _, ravel_batch_y0_r0 = ravel_pytree(batch_y0_r0)

    _, ravel_batch_y0 = ravel_pytree(batch_y0)

    r = np.zeros((args.batch_time, args.batch_size, 1))
    batch_y_r = np.concatenate((batch_y, r), axis=2)
    _, ravel_batch_y_r = ravel_pytree(batch_y_r)

    _, ravel_batch_y = ravel_pytree(batch_y)
    fargs = flat_params

    def f(y, t, *args):
        """
        Simple MLP.
        """
        flat_params = args
        # convert flat_params from Tuple to DeviceArray, then ravel
        flat_params, _ = ravel_pytree(flat_params)
        params = ravel_params(flat_params)

        y = ravel_batch_y0(y)
        predictions = predict(params, y ** 3)

        flat_predictions, _ = ravel_pytree(predictions)
        return flat_predictions

    def reg_dynamics(y_r, t, *args):
        """
        Augmented dynamics to implement regularization.
        """
        NFE_COUNT[0] += 1

        flat_params = args
        # convert flat_params from Tuple to DeviceArray, then ravel
        flat_params, _ = ravel_pytree(flat_params)
        params = ravel_params(flat_params)

        # separate out state from augmented
        y_r = ravel_batch_y0_r0(y_r)
        # TODO: do we need to use jax.index here?
        y, r = y_r[:, :2], y_r[:, 2]

        predictions = predict(params, y ** 3)
        # STATE
        regularization = np.sum(y ** 2, axis=1) ** 0.5
        # DYNAMICS
        regularization = np.sum(predictions ** 2, axis=1) ** 0.5

        regularization = np.expand_dims(regularization, axis=1)
        pred_reg = np.concatenate((predictions, regularization), axis=1)
        flat_pred_reg, _ = ravel_pytree(pred_reg)
        return flat_pred_reg

    def loss_fun(pred_y_r, target):
        """
        Loss function.
        """
        pred, reg = pred_y_r[:, :, :2], pred_y_r[:, :, 2]
        # TODO: not making a silly math mistake here right?
        return np.mean(np.abs(pred - target)) + args.lam * np.mean(reg)

    def error_fun(pred, target):
        """
        Mean absolute error.
        """
        return np.mean(np.abs(pred - target))

    grad_loss_fun = grad(loss_fun)

    for itr in range(1, args.niters + 1):
        batch_y0, batch_t, batch_y = get_batch()
        r0 = np.zeros((args.batch_size, 1))
        batch_y0_r0 = np.concatenate((batch_y0, r0), axis=1)
        flat_batch_y0_r0, _ = ravel_pytree(batch_y0_r0)

        # integrate ODE and count NFE
        NFE_COUNT[0] = 0
        pred_y_r = odeint(reg_dynamics, fargs, flat_batch_y0_r0, batch_t, atol=1e-8, rtol=1e-8)
        print("forward NFE: %d" % NFE_COUNT[0])

        _, ravel_pred_y_r = ravel_pytree(pred_y_r)

        ode_vjp = grad_odeint(reg_dynamics, fargs)
        loss_grad = grad_loss_fun(ravel_batch_y_r(pred_y_r), batch_y)

        # integrate adjoint ODE and count NFE
        NFE_COUNT[0] = 0
        total_grad = ode_vjp(ravel_pred_y_r(loss_grad), pred_y_r, batch_t)
        print("backward NFE: %d" % NFE_COUNT[0])

        params_grad = total_grad[3]
        fargs -= args.lr * params_grad

        if itr % args.test_freq == 0:
            pred_y = ravel_batch_y_r(pred_y_r)[:, :, :2]
            error = error_fun(ravel_batch_y(pred_y), batch_y)
            loss = loss_fun(ravel_batch_y_r(pred_y_r), batch_y)
            print('Iter {:04d} | Total Loss {:.6f} | Error {:.6f}'.format(itr, loss, error))
            ii += 1
