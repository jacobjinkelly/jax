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
from jax import random
import jax.numpy as np
from jax.config import config
from jax.flatten_util import ravel_pytree

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
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
    # TODO: different batch each time?
    _, subkey = random.split(key)
    s = random.shuffle(subkey, np.arange(args.data_size - args.batch_time, dtype=np.int64))[:args.batch_size]
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = np.stack([true_y[s + i] for i in range(args.batch_time)], axis=0)  # (T, M, D)
    return batch_y0, batch_t, batch_y


def visualize(true_y, pred_y, odefunc, itr):
    """
    Visualize the model's predictions.
    """
    # TODO
    pass


if __name__ == "__main__":

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
    _, ravel_batch_y0 = ravel_pytree(batch_y0)
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

    for itr in range(1, args.niters + 1):
        batch_y0, batch_t, batch_y = get_batch()
        flat_batch_y0, _ = ravel_pytree(batch_y0)
        pred_y = odeint(f, fargs, flat_batch_y0, batch_t, atol=1e-8, rtol=1e-8)
        pred_y = ravel_batch_y(pred_y)
        loss = np.mean(np.abs(pred_y - batch_y))

        # TODO: one step of SGD

        # t0 = 0.1
        # t1 = 0.2
        # y0 = np.linspace(0.1, 0.9, dim)
        # ys = odeint(f, fargs, y0, np.array([t0, t1]), atol=1e-8, rtol=1e-8)
        # ode_vjp = grad_odeint(f, fargs)
        # g = np.ones_like(ys)
        # exact_grad, _ = ravel_pytree(ode_vjp(g, ys, np.array([t0, t1])))

        if itr % args.test_freq == 0:
            # print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss))
            # visualize(true_y, pred_y, f, ii)
            print(ii)
            ii += 1
