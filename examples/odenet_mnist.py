"""
Neural ODEs in Jax.
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
from jax.experimental import stax, optimizers
from jax.experimental.ode import odeint, build_odeint, vjp_odeint
from jax.experimental.stax import GeneralConv, Conv, Dense, Identity, AvgPool, Flatten, FanInConcat, Relu, LogSoftmax

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

    def FanOutConcatTime():
        """
        Custom stax layer to fan out and concat [out, t] |-> [out|t, t]
        """
        def init_fun(rng, input_shape):
            output_shape = ((input_shape[0][0], input_shape[0][1], input_shape[0][2], input_shape[0][3] + 1),
                            input_shape[1])
            return output_shape, ()

        def apply_fun(params, inputs, **kwargs):
            out, t = inputs
            tt = np.ones_like(out[:, :, :, :1]) * t
            ttx = np.concatenate((out, tt), axis=-1)
            return (ttx, t)

        return init_fun, apply_fun

    def FanInConcatTime():
        """
        Custom stax layer to fan in and concat [out, t] |-> out|t
        """
        def init_fun(rng, input_shape):
            output_shape = (input_shape[0][0], input_shape[0][1], input_shape[0][2], input_shape[0][3] + 1)
            return output_shape, ()

        def apply_fun(params, inputs, **kwargs):
            out, t = inputs
            tt = np.ones_like(out[:, :, :, :1]) * t
            ttx = np.concatenate((out, tt), axis=-1)
            return ttx

        return init_fun, apply_fun

    odefunc_init_random_params, odefunc_predict = stax.serial(
        stax.parallel(Relu, Identity),
        FanOutConcatTime(),                                         # [out, t] |-> [out|t, t]
        stax.parallel(
            stax.serial(Conv(64, (3, 3), padding=((1, 1), (1, 1))),
                        Relu),
            Identity),
        FanInConcatTime(),                                            # [out, t] |-> out|t
        Conv(64, (3, 3), padding=((1, 1), (1, 1)))
    )

    def dynamics(y_t, t, *args):
        """
        Dynamics of the ODEBlock.
        """
        flat_y, t = y_t[:-1], y_t[-1:]
        y = np.reshape(flat_y, (-1, 6, 6, 64))
        out_y = odefunc_predict(args, (y, t))
        flat_out_y = np.reshape(out_y, (-1,))
        return np.concatenate((flat_out_y, np.ones((1,))))
    nodes_odeint = build_odeint(dynamics)

    def ODEBlock():
        """
        A stax module for an ODENet
        """

        def init_fun(rng, input_shape):
            output_shape, odefunc_params = odefunc_init_random_params(rng, (input_shape, (1,)))
            return output_shape, odefunc_params

        def apply_fun(params, inputs, **kwargs):
            flat_inputs = np.reshape(inputs, (-1,))
            outputs = nodes_odeint(np.concatenate((flat_inputs, t[:1])), t, *params)[-1]  # system at t[-1]
            # remove time from result, reshape
            return np.reshape(outputs[:-1], (-1, 6, 6, 64))

        return init_fun, apply_fun

    # set up MLP
    init_random_params, predict = stax.serial(
        GeneralConv(('HWCN', 'OIHW', 'NHWC'), 64, (3, 3), padding="VALID"), Relu,
        Conv(64, (4, 4), (2,) * len((4, 4)), ((1, 1), (1, 1))), Relu,
        Conv(64, (4, 4), (2,) * len((4, 4)), ((1, 1), (1, 1))),
        ODEBlock(), Relu,
        AvgPool((1, 1)), Flatten,
        Dense(10), LogSoftmax
    )

    _, init_params = init_random_params(key, (28, 28, 1, -1))

    t = np.array([0., 1.])

    train_images, train_labels, test_images, test_labels = datasets.mnist()
    train_images = np.reshape(train_images, (28, 28, 1, -1)).astype(np.float64)
    test_images = np.reshape(test_images, (28, 28, 1, -1)).astype(np.float64)

    num_train = train_images.shape[-1]
    num_complete_batches, leftover = divmod(num_train, parse_args.batch_size)
    num_batches = num_complete_batches + bool(leftover)

    def data_stream():
        rng = npr.RandomState(0)
        while True:
            perm = rng.permutation(num_train)
            for i in range(num_batches):
                batch_idx = perm[i * parse_args.batch_size:(i + 1) * parse_args.batch_size]
                yield train_images[:, :, :, batch_idx], train_labels[batch_idx]
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

    opt_state = opt_init(init_params)
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
