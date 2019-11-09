"""
Neural ODEs in Jax.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pickle
import sys

import jax
import jax.numpy as np
from jax import random, grad, lax
from jax.config import config
from jax.experimental import stax, optimizers
from jax.experimental.ode import odeint, build_odeint, vjp_odeint
from jax.experimental.stax import Dense, Tanh
from jax.flatten_util import ravel_pytree

REGS = ['r0', 'r1']
NUM_REGS = len(REGS)

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--nepochs', type=int, default=100)
parser.add_argument('--lam', type=float, default=0)
parser.add_argument('--reg', type=str, choices=['none'] + REGS, default='none')
parser.add_argument('--test_freq', type=int, default=10)
parser.add_argument('--dirname', type=str, default='tmp14')
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
parse_args = parser.parse_args()


config.update('jax_enable_x64', True)

D = 1
true_y0 = np.repeat(np.expand_dims(np.linspace(-1, 0, parse_args.data_size), axis=1), D, axis=1)  # (N, D)
# true_y1 = np.concatenate((np.expand_dims(true_y0[:, 0] ** 2, axis=1),
#                           np.expand_dims(true_y0[:, 1] ** 3, axis=1),
#                           np.expand_dims(true_y0[:, 2] ** 4, axis=1)
#                           ),
#                          axis=1)
true_y1 = np.expand_dims(true_y0[:, 0] ** 2 + true_y0[:, 0], axis=1)
true_y = np.concatenate((np.expand_dims(true_y0, axis=0),
                        np.expand_dims(true_y1, axis=0)),
                        axis=0)  # (T, N, D)
t = np.array([0., 1.])  # (T)


@jax.jit
def get_batch(shuffled_inds, batch):
    """
    Get batch.
    """
    s = lax.dynamic_slice(shuffled_inds, [parse_args.batch_size * batch], [parse_args.batch_size])
    batch_y0 = true_y0[s]       # (M, D)
    batch_t = t                 # (T)
    batch_y = true_y[:, s, :]   # (T, M, D)
    return batch_y0, batch_t, batch_y

@jax.jit
def pack_batch(shuffled_inds, batch):
    """
    Get batch and package it for augmented system integration.
    """
    batch_y0, batch_t, batch_y = get_batch(shuffled_inds, batch)
    batch_y0_t = np.concatenate((batch_y0,
                                 np.expand_dims(
                                     np.repeat(batch_t[0], parse_args.batch_size),
                                     axis=1)
                                 ),
                                axis=1)
    flat_batch_y0_t = np.reshape(batch_y0_t, (-1,))
    r0 = np.zeros((parse_args.batch_size, 1))
    allr0 = np.zeros((parse_args.batch_size, NUM_REGS))
    batch_y0_t_r0_allr0 = np.concatenate((batch_y0_t, r0, allr0), axis=1)
    flat_batch_y0_t_r0_allr0 = np.reshape(batch_y0_t_r0_allr0, (-1,))
    return flat_batch_y0_t, flat_batch_y0_t_r0_allr0, batch_t, batch_y


def run(reg, lam, key, dirname):
    """
    Run the neural ODEs method.
    """
    print("Reg: %s\tLambda %.4e" % (reg, lam))
    print("Reg: %s\tLambda %.4e" % (reg, lam), file=sys.stderr)

    # set up MLP
    init_random_params, predict = stax.serial(
        Dense(50), Tanh,
        Dense(D)
    )

    output_shape, init_params = init_random_params(key, (-1, D + 1))
    assert output_shape == (-1, D)

    # define ravel objects

    flat_params, ravel_params = ravel_pytree(init_params)

    batch_y0, batch_t, batch_y = get_batch(np.arange(parse_args.data_size, dtype=np.int64), 0)

    r0 = np.zeros((parse_args.batch_size, 1))
    allr0 = np.zeros((parse_args.batch_size, NUM_REGS))
    r = np.zeros((parse_args.batch_time, parse_args.batch_size, 1))
    allr = np.zeros((parse_args.batch_time, parse_args.batch_size, NUM_REGS))
    test_r = np.zeros((parse_args.batch_time, parse_args.data_size, 1))
    test_allr = np.zeros((parse_args.batch_time, parse_args.data_size, NUM_REGS))
    test_r0 = np.zeros((parse_args.data_size, 1))
    test_allr0 = np.zeros((parse_args.data_size, NUM_REGS))

    batch_y0_t = np.concatenate((batch_y0,
                                 np.expand_dims(
                                     np.repeat(batch_t[0], parse_args.batch_size),
                                     axis=1)
                                 ),
                                axis=1)
    # parse_args.batch_size * (D + 1) |-> (parse_args.batch_size, D + 1)
    _, ravel_batch_y0_t = ravel_pytree(batch_y0_t)

    batch_y_t = np.concatenate((batch_y,
                                np.expand_dims(
                                    np.tile(batch_t, (parse_args.batch_size, 1)).T,
                                    axis=2)
                                ),
                               axis=2)
    # parse_args.batch_time * parse_args.batch_size * (D + 1) |-> (parse_args.batch_time, parse_args.batch_size, D + 1)
    _, ravel_batch_y_t = ravel_pytree(batch_y_t)

    batch_y0_t_r0_allr0 = np.concatenate((batch_y0_t, r0, allr0), axis=1)
    # parse_args.batch_size * (D + 2 + NUM_REGS) |-> (parse_args.batch_size, D + 2 + NUM_REGS)
    _, ravel_batch_y0_t_r0_allr0 = ravel_pytree(batch_y0_t_r0_allr0)

    batch_y_t_r_allr = np.concatenate((batch_y,
                                       np.expand_dims(
                                           np.tile(batch_t, (parse_args.batch_size, 1)).T,
                                           axis=2),
                                       r,
                                       allr),
                                      axis=2)
    # parse_args.batch_time * parse_args.batch_size * (D + 2 + NUM_REGS) |->
    #                                                   (parse_args.batch_time, parse_args.batch_size, D + 2 + NUM_REGS)
    _, ravel_batch_y_t_r_allr = ravel_pytree(batch_y_t_r_allr)

    true_y_t_r_allr = np.concatenate((true_y,
                                      np.expand_dims(
                                          np.tile(batch_t, (parse_args.data_size, 1)).T,
                                          axis=2),
                                      test_r,
                                      test_allr),
                                     axis=2)
    # parse_args.batch_time * parse_args.data_size * (D + 2 + NUM_REGS) |->
    #                                       (parse_args.batch_time, parse_args.data_size, D + 2 + NUM_REGS)
    _, ravel_true_y_t_r_allr = ravel_pytree(true_y_t_r_allr)

    true_y0_t_r0_allr = np.concatenate((true_y0,
                                        np.expand_dims(
                                            np.repeat(t[0], parse_args.data_size), axis=1),
                                        test_r0,
                                        test_allr0), axis=1)
    # parse_args.data_size * (D + 2 + NUM_REGS) |-> (parse_args.data_size, D + 2 + NUM_REGS)
    flat_true_y0_t_r0_allr, ravel_true_y0_t_r0_allr = ravel_pytree(true_y0_t_r0_allr)

    fargs = flat_params

    @jax.jit
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

        flat_predictions = np.reshape(predictions, (-1,))
        return flat_predictions

    @jax.jit
    def reg_dynamics(y_t_r_allr, t, *args):
        """
        Augmented dynamics to implement regularization.
        """

        flat_params = args
        params = ravel_params(np.array(flat_params))

        # separate out state from augmented
        y_t_r_allr = ravel_batch_y0_t_r0_allr0(y_t_r_allr)
        y_t = y_t_r_allr[:, :D + 1]
        y = y_t[:, :-1]

        predictions_y = predict(params, y_t)
        predictions = np.concatenate((predictions_y,
                                      np.ones((parse_args.batch_size, 1))),
                                     axis=1)

        r0 = np.sum(y ** 2, axis=1) ** 0.5
        r1 = np.sum(predictions_y ** 2, axis=1) ** 0.5
        if reg == "r0":
            regularization = r0
        elif reg == "r1":
            regularization = r1
        else:
            regularization = np.zeros(parse_args.batch_size)

        pred_reg = np.concatenate((predictions,
                                   np.expand_dims(regularization, axis=1),
                                   np.expand_dims(r0, axis=1),
                                   np.expand_dims(r1, axis=1)),
                                  axis=1)
        flat_pred_reg = np.reshape(pred_reg, (-1,))
        return flat_pred_reg

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
                                      np.ones((parse_args.data_size, 1))),
                                     axis=1)

        r0 = np.sum(y ** 2, axis=1) ** 0.5
        r1 = np.sum(predictions_y ** 2, axis=1) ** 0.5
        if reg == "r0":
            regularization = r0
        elif reg == "r1":
            regularization = r1
        else:
            regularization = np.zeros(parse_args.data_size)

        pred_reg = np.concatenate((predictions,
                                   np.expand_dims(regularization, axis=1),
                                   np.expand_dims(r0, axis=1),
                                   np.expand_dims(r1, axis=1)),
                                  axis=1)
        flat_pred_reg = np.reshape(pred_reg, (-1,))
        return flat_pred_reg

    @jax.jit
    def total_loss_fun(pred_y_t_r, target):
        """
        Loss function.
        """
        pred, reg = pred_y_t_r[:, :, :D], pred_y_t_r[:, :, D + 1]
        return loss_fun(pred, target) + lam * reg_loss(reg)

    @jax.jit
    def reg_loss(reg):
        """
        Regularization loss function.
        """
        return np.mean(reg)

    @jax.jit
    def loss_fun(pred, target):
        """
        Mean squared error.
        """
        return np.mean((pred - target) ** 2)

    @jax.jit
    def nodes_predict(args):
        """
        Evaluate loss on model's predictions.
        """
        true_ys, odeint_args = args[0], args[1:]
        ys = ravel_batch_y_t_r_allr(nodes_odeint(*odeint_args))
        return total_loss_fun(ys, true_ys)

    # unregularized system for counting NFE
    unreg_nodes_odeint = jax.jit(lambda y0, t, args: odeint(dynamics, y0, t, *args))
    unreg_nodes_odeint_vjp = jax.jit(lambda cotangent, y0, t, args:
                                     vjp_odeint(dynamics, y0, t, *args, nfe=True)[1](np.reshape(cotangent,
                                                                                                (parse_args.batch_time,
                                                                                                 parse_args.batch_size *
                                                                                                 (D + 1))))[-1])
    grad_loss_fn = grad(loss_fun)

    # full system for training
    nodes_odeint = build_odeint(reg_dynamics)
    grad_predict = jax.jit(grad(nodes_predict))

    # for testing
    nodes_odeint_test = build_odeint(test_reg_dynamics)

    opt_init, opt_update, get_params = optimizers.rmsprop(step_size=1e-3, gamma=0.99)
    opt_state = opt_init(fargs)

    assert parse_args.data_size % parse_args.batch_size == 0
    batch_per_epoch = parse_args.data_size // parse_args.batch_size

    for epoch in range(parse_args.nepochs):

        new_key, subkey = random.split(key)
        key = new_key
        shuffled_inds = random.shuffle(subkey, np.arange(parse_args.data_size, dtype=np.int64))

        for batch in range(batch_per_epoch):
            itr = epoch * batch_per_epoch + batch + 1

            flat_batch_y0_t, flat_batch_y0_t_r0_allr0, batch_t, batch_y = pack_batch(shuffled_inds, batch)

            fargs = get_params(opt_state)

            # integrate unregularized system and count NFE
            pred_y_t, nfe = unreg_nodes_odeint(flat_batch_y0_t, batch_t, fargs)
            print("forward NFE: %d" % nfe)

            # integrate adjoint ODE to count NFE
            grad_loss = grad_loss_fn(ravel_batch_y_t(pred_y_t)[:, :, :D], batch_y)
            cotangent = np.concatenate((grad_loss,
                                        np.zeros((parse_args.batch_time, parse_args.batch_size, 1))),
                                       axis=2)
            nfe = unreg_nodes_odeint_vjp(cotangent, flat_batch_y0_t, batch_t, fargs)
            print("backward NFE: %d" % nfe)

            params_grad = np.array(grad_predict((batch_y, flat_batch_y0_t_r0_allr0, batch_t, *fargs))[3:])
            opt_state = opt_update(itr, params_grad, opt_state)

            if itr % parse_args.test_freq == 0:
                fargs = get_params(opt_state)

                pred_y_t_r_allr = ravel_true_y_t_r_allr(nodes_odeint_test(flat_true_y0_t_r0_allr, t, *fargs))

                pred_y = pred_y_t_r_allr[:, :, :D]
                pred_y_t_r = pred_y_t_r_allr[:, :, :D + 2]

                loss = loss_fun(pred_y, true_y)

                total_loss = total_loss_fun(pred_y_t_r, true_y)

                r0_reg = np.mean(pred_y_t_r_allr[1, :, -2])
                r1_reg = np.mean(pred_y_t_r_allr[1, :, -1])

                print('Iter {:04d} | Total (Regularized) Loss {:.6f} | Loss {:.6f} | '
                      'r0 {:.6f} | r1 {:.6f}'.
                      format(itr, total_loss, loss, r0_reg, r1_reg))
                print('Iter {:04d} | Total (Regularized) Loss {:.6f} | Loss {:.6f} | '
                      'r0 {:.6f} | r1 {:.6f}'.
                      format(itr, total_loss, loss, r0_reg, r1_reg),
                      file=sys.stderr)

        param_filename = "%s/reg_%s_lam_%.4e_%d_fargs.pickle" % (dirname, reg, lam, (epoch + 1) * batch_per_epoch)
        outfile = open(param_filename, "wb")
        pickle.dump(fargs, outfile)
        outfile.close()


if __name__ == "__main__":
    assert os.path.exists(parse_args.dirname)
    run(parse_args.reg, parse_args.lam, random.PRNGKey(0), parse_args.dirname)