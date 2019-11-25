"""
Neural ODEs on MNIST (using simple MLP, implemented w/o stax).
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
from jax.experimental import optimizers
from jax.experimental.ode import build_odeint, odeint, vjp_odeint
from jax.flatten_util import ravel_pytree
from jax.nn import sigmoid, log_softmax
from jax.nn.initializers import glorot_normal, normal

regs = ['r0', 'r1']
num_regs = len(regs)

parser = argparse.ArgumentParser('ODE MNIST')
parser.add_argument('--method', type=str, choices=['dopri5'], default='dopri5')
parser.add_argument('--batch_time', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--nepochs', type=int, default=160)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--mom', type=float, default=0.9)
parser.add_argument('--lam', type=float, default=0)
parser.add_argument('--reg', type=str, choices=['none'] + regs, default='none')
parser.add_argument('--test_freq', type=int, default=1)
parser.add_argument('--save_freq', type=int, default=500)
parser.add_argument('--dirname', type=str, default='tmp16')
parse_args = parser.parse_args()

img_dim = 784
ode_dim = 64
n_classes = 10


def run(reg, lam, rng, dirname):
    """
    Run the neural ODEs method.
    """
    print("Reg: %s\tLambda %.4e" % (reg, lam))
    print("Reg: %s\tLambda %.4e" % (reg, lam), file=sys.stderr)

    def dynamics(flat_y_t, t, *flat_params):
        """
        Dynamics of the ODEBlock.
        """
        params = ravel_ode_params(np.array(flat_params))
        y_t = np.reshape(flat_y_t, (-1, ode_dim + 1))
        w_dyn, b_dyn = params
        out_y = sigmoid(np.dot(y_t, w_dyn) + np.expand_dims(b_dyn, axis=0))
        out_y_t = np.concatenate((out_y,
                                  np.ones((out_y.shape[0], 1))),
                                 axis=1)
        flat_out_y_t = np.reshape(out_y_t, (-1,))
        return flat_out_y_t

    def reg_dynamics(flat_y_t_r_allr, t, *flat_params):
        """
        Dynamics of the ODEBlock.
        """
        params = ravel_ode_params(np.array(flat_params))

        y_t_r_allr = np.reshape(flat_y_t_r_allr, (-1, ode_dim + 2 + num_regs))
        y_t = y_t_r_allr[:, :ode_dim + 1]
        y = y_t[:, :-1]

        w_dyn, b_dyn = params
        out_y = sigmoid(np.dot(y_t, w_dyn) + np.expand_dims(b_dyn, axis=0))
        out_y_t = np.concatenate((out_y,
                                  np.ones((out_y.shape[0], 1))),
                                 axis=1)

        r0 = np.sum(y ** 2, axis=1) ** 0.5
        r1 = np.sum(out_y ** 2, axis=1)
        if reg == "r0":
            regularization = r0
        elif reg == "r1":
            regularization = r1
        else:
            regularization = np.zeros(out_y.shape[0])

        pred_reg = np.concatenate((out_y_t,
                                   np.expand_dims(regularization, axis=1),
                                   np.expand_dims(r0, axis=1),
                                   np.expand_dims(r1, axis=1)),
                                  axis=1)
        flat_pred_reg = np.reshape(pred_reg, (-1,))

        return flat_pred_reg
    nodes_odeint = build_odeint(reg_dynamics)

    def mlp_1(args, x):
        w_1, b_1 = args
        out_1 = np.dot(x, w_1) + np.expand_dims(b_1, axis=0)
        return out_1

    def mlp_2(args, out_ode):
        w_2, = args
        in_2 = sigmoid(out_ode)
        out_2 = np.dot(in_2, w_2)
        out = log_softmax(out_2)
        return out

    def predict(args, x):
        """
        The prediction function for our neural net.
        """
        params = ravel_params(args)

        flat_ode_params = params[1]

        out_1 = mlp_1(params[0], x)

        in_ode = np.concatenate((out_1,
                                 np.ones((out_1.shape[0], 1)) * t[1],
                                 np.zeros((out_1.shape[0], num_regs + 1))),
                                axis=1)
        flat_in_ode = np.reshape(in_ode, (-1,))
        flat_out_ode = nodes_odeint(flat_in_ode, t, *flat_ode_params)[-1]
        out_t_r_allr_ode = np.reshape(flat_out_ode, (-1, ode_dim + 2 + num_regs))
        out_ode = out_t_r_allr_ode[:, :ode_dim]

        out = mlp_2(params[2], out_ode)

        return out, out_t_r_allr_ode[:, ode_dim + 1], out_t_r_allr_ode[:, ode_dim + 2:]

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
    def sep_loss_fun(opt_state, batch):
        """
        Return total loss and loss
        """
        params = get_params(opt_state)
        inputs, targets = batch
        preds = predict(params, inputs)
        pred, reg, allregs = preds
        loss_ = loss_fun(pred, targets)
        reg_ = lam * reg_loss(reg)
        r0_reg = reg_loss(allregs[:, 0])
        r1_reg = reg_loss(allregs[:, 1])
        return loss_ + reg_, loss_, r0_reg, r1_reg

    @jax.jit
    def total_loss_fun(preds_reg, target):
        """
        Loss function.
        """
        pred, reg, _ = preds_reg
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
        return total_loss_fun(preds, targets)

    @jax.jit
    def partial_loss(out_ode, targets, args):
        preds = mlp_2(args, out_ode)
        return loss_fun(preds, targets)
    grad_partial_loss = grad(partial_loss)

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

    # initialize the parameters
    rng, layer_rng = random.split(rng)
    k1, k2 = random.split(layer_rng)
    w_1, b_1 = glorot_normal()(k1, (img_dim, ode_dim)), normal()(k2, (ode_dim,))

    rng, layer_rng = random.split(rng)
    k1, k2 = random.split(layer_rng)
    w_dyn, b_dyn = glorot_normal()(k1, (ode_dim + 1, ode_dim)), normal()(k2, (ode_dim,))

    rng, layer_rng = random.split(rng)
    k1, k2 = random.split(layer_rng)
    w_2 = glorot_normal()(k1, (ode_dim, n_classes))

    flat_ode_params, ravel_ode_params = ravel_pytree((w_dyn, b_dyn))
    init_params = [(w_1, b_1), flat_ode_params, (w_2,)]

    # train
    flat_params, ravel_params = ravel_pytree(init_params)
    opt_state = opt_init(flat_params)
    itercount = itertools.count()

    # unregularized system for counting NFE
    unreg_nodes_odeint = jax.jit(lambda y0, t, args: odeint(dynamics, y0, t, *args))
    unreg_nodes_odeint_vjp = jax.jit(lambda cotangent, y0, t, args:
                                     vjp_odeint(dynamics, y0, t, *args, nfe=True)[1](np.reshape(cotangent,
                                                                                                (parse_args.batch_time,
                                                                                                 parse_args.batch_size *
                                                                                                 (ode_dim + 1))))[-1])

    @jax.jit
    def count_nfe(opt_state, batch):
        """
        Count NFE.
        """
        inputs, targets = batch
        params = ravel_params(get_params(opt_state))
        out_1 = mlp_1(params[0], inputs)

        in_ode = np.concatenate((out_1,
                                 np.ones((out_1.shape[0], 1)) * t[1]),
                                axis=1)
        flat_in_ode = np.reshape(in_ode, (-1,))
        flat_out_ode, f_nfe = unreg_nodes_odeint(flat_in_ode, t, flat_ode_params)
        out_t_ode = np.reshape(flat_out_ode[-1:], (-1, ode_dim + 1))
        out_ode = out_t_ode[:, :ode_dim]

        grad_partial_loss_ = grad_partial_loss(out_ode, targets, params[2])
        full_grad_partial_loss = np.concatenate((np.zeros((1, parse_args.batch_size, ode_dim)),
                                                 np.expand_dims(grad_partial_loss_, axis=0)),
                                                axis=0)
        cotangent = np.concatenate((full_grad_partial_loss,
                                    np.zeros((parse_args.batch_time, parse_args.batch_size, 1))),
                                   axis=2)
        b_nfe = unreg_nodes_odeint_vjp(cotangent, np.reshape(in_ode, (-1, )), t, flat_ode_params)

        return f_nfe, b_nfe

    for epoch in range(parse_args.nepochs):
        for i in range(num_batches):
            batch = next(batches)
            itr = next(itercount)

            # f_nfe, b_nfe = count_nfe(opt_state, batch)

            # print("forward NFE: %d" % f_nfe)
            # print("backward NFE: %d" % b_nfe)

            # opt_state = update(itr, opt_state, batch)

            if itr % parse_args.test_freq == 0:

                total_loss, loss, r0_reg, r1_reg = sep_loss_fun(opt_state, (train_images, train_labels))

                print('Iter {:04d} | Total (Regularized) Loss {:.6f} | Loss {:.6f} | r0 {:.6f} | r1 {:.6f}'.
                      format(itr, total_loss, loss, r0_reg, r1_reg))
                print('Iter {:04d} | Total (Regularized) Loss {:.6f} | Loss {:.6f} | r0 {:.6f} | r1 {:.6f}'.
                      format(itr, total_loss, loss, r0_reg, r1_reg),
                      file=sys.stderr)

            if itr % parse_args.save_freq == 0:
                param_filename = "%s/reg_%s_lam_%.4e_%d_fargs.pickle" % (dirname, reg, lam, itr)
                fargs = get_params(opt_state)
                outfile = open(param_filename, "wb")
                pickle.dump(fargs, outfile)
                outfile.close()


if __name__ == "__main__":
    assert os.path.exists(parse_args.dirname)
    run(parse_args.reg, parse_args.lam, random.PRNGKey(0), parse_args.dirname)
