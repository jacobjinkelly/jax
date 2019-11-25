"""
Neural ODEs on MNIST (using simple MLP).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import itertools
import os
import pickle
import sys

import numpy.random as npr

import jax
import jax.numpy as np
from jax.examples import datasets
from jax import random, grad
from jax.flatten_util import ravel_pytree
from jax.experimental import stax, optimizers
from jax.experimental.ode import odeint, build_odeint, vjp_odeint
from jax.experimental.stax import Dense, Sigmoid, LogSoftmax

REGS = ['r0', 'r1']
NUM_REGS = len(REGS)

parser = argparse.ArgumentParser('ODE MNIST')
parser.add_argument('--method', type=str, choices=['dopri5'], default='dopri5')
parser.add_argument('--batch_time', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--nepochs', type=int, default=160)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--mom', type=float, default=0.9)
parser.add_argument('--lam', type=float, default=0)
parser.add_argument('--reg', type=str, choices=['none'] + REGS, default='none')
parser.add_argument('--test_freq', type=int, default=1)
parser.add_argument('--save_freq', type=int, default=2)
parser.add_argument('--dirname', type=str, default='tmp16')
parse_args = parser.parse_args()


def run(reg, lam, key, dirname):
    """
    Run the neural ODEs method.
    """
    print("Reg: %s\tLambda %.4e" % (reg, lam))
    print("Reg: %s\tLambda %.4e" % (reg, lam), file=sys.stderr)

    odefunc_init_random_params, odefunc_predict = stax.serial(
        Dense(64), Sigmoid
    )

    def dynamics(flat_y_t, t, *flat_params):
        """
        Dynamics of the ODEBlock.
        """
        params = ravel_ode_params(np.array(flat_params))
        y_t = np.reshape(flat_y_t, (-1, 65))
        out_y = odefunc_predict(params, y_t)
        out_y_t = np.concatenate((out_y,
                                  np.ones((out_y.shape[0], 1))),
                                 axis=1)
        flat_out_y_t = np.reshape(out_y_t, (-1,))
        return flat_out_y_t
    nodes_odeint = build_odeint(dynamics)

    def ODEBlock():
        """
        A stax module for an ODENet
        """

        def init_fun(rng, input_shape):
            time_cat_input_shape = (input_shape[0], input_shape[1] + 1)
            output_shape, odefunc_params = odefunc_init_random_params(rng, time_cat_input_shape)
            return output_shape, odefunc_params

        def apply_fun(params, inputs, **kwargs):
            inputs_t = np.concatenate((inputs,
                                       np.ones((inputs.shape[0], 1)) * t[1]),
                                      axis=1)
            flat_inputs_t = np.reshape(inputs_t, (-1,))
            flat_outputs = nodes_odeint(flat_inputs_t, t, *params)[-1]  # system at t[-1]
            outputs = np.reshape(flat_outputs, (-1, 65))
            # remove time from result
            return outputs[:, :-1]

        return init_fun, apply_fun

    # set up MLP
    init_random_params, predict = stax.serial(
        Dense(64), Sigmoid,
        ODEBlock(), Sigmoid,
        Dense(10), LogSoftmax
    )

    _, init_params = init_random_params(key, (-1, 784))

    flat_ode_params, ravel_ode_params = ravel_pytree(init_params[2])

    flat_init_params = [init_params[0],
                        init_params[1],
                        flat_ode_params,
                        init_params[3],
                        init_params[4],
                        init_params[5]
    ]

    t = np.array([0., 1.])

    train_images, train_labels, test_images, test_labels = datasets.mnist()

    num_train = train_images.shape[-1]
    num_complete_batches, leftover = divmod(num_train, parse_args.batch_size)
    num_batches = num_complete_batches + bool(leftover)

    def data_stream():
        rng = npr.RandomState(0)
        while True:
            perm = rng.permutation(num_train)
            for i in range(num_batches):
                batch_idx = perm[i * parse_args.batch_size:(i + 1) * parse_args.batch_size]
                yield train_images[batch_idx], train_labels[batch_idx]
    batches = data_stream()

    @jax.jit
    def total_loss_fun(pred_y_t_r, target):
        """
        Loss function.
        """
        # TODO: implement, and use this inside loss
        pred, reg = pred_y_t_r[:, :, :D], pred_y_t_r[:, :, D + 1]
        return loss_fun(pred, target) + lam * reg_loss(reg)

    @jax.jit
    def reg_loss(reg):
        """
        Regularization loss function.
        """
        return np.mean(reg)

    @jax.jit
    def loss_fun(preds, targets):
        """
        Mean squared error.
        """
        return -np.mean(np.sum(preds * targets, axis=1))

    @jax.jit
    def loss(params, batch):
        inputs, targets = batch
        preds = predict(params, inputs)
        return loss_fun(preds, targets)

    opt_init, opt_update, get_params = optimizers.momentum(parse_args.lr, mass=parse_args.mom)

    @jax.jit
    def update(i, opt_state, batch):
        """
        Update the params based on grad for current batch.
        """
        params = get_params(opt_state)
        return opt_update(i, grad(loss)(params, batch), opt_state)

    @jax.jit
    def accuracy(params, batch):
        inputs, targets = batch
        target_class = np.argmax(targets, axis=1)
        predicted_class = np.argmax(predict(params, inputs), axis=1)
        return np.mean(predicted_class == target_class)

    opt_state = opt_init(flat_init_params)
    itercount = itertools.count()

    for epoch in range(parse_args.nepochs):
        for i in range(num_batches):
            print(i)
            opt_state = update(next(itercount), opt_state, next(batches))

        if epoch % parse_args.test_freq == 0:
            params = get_params(opt_state)
            train_acc = accuracy(params, (train_images, train_labels))
            test_acc = accuracy(params, (test_images, test_labels))
            print("Epoch {} | Train Acc. {} | Test Acc {}".format(epoch, train_acc, test_acc))


if __name__ == "__main__":
    assert os.path.exists(parse_args.dirname)
    run(parse_args.reg, parse_args.lam, random.PRNGKey(0), parse_args.dirname)
