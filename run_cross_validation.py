"""
Run k-fold cross validation to find the best hyperparameters for the models.
Plot the results. We have to set hyperparameters manually in run_final_training.py
after analyzing the results of the cross validation.
"""
import pickle
from functools import partial

from matplotlib import pyplot as plt

import implementations as imp
from hyperparameters import *
from exponential_lr import ExponentialLR

OUTPUT_PATH = OUTPUT_PATH / 'cross_validation'


def run_cross_validation(x_validate: np.ndarray, y_validate: np.ndarray, k: int = K):
    """
    Run k-fold cross validation to find the best hyperparameters for the models.
    :param x_validate: the validation data
    :param y_validate: the validation labels
    :param k: the number of folds
    :return: the best model and the best exponential_lr
    """
    if not OUTPUT_PATH.exists():
        OUTPUT_PATH.mkdir()
    np.random.seed(0)
    # we fix this model (don't even try the others as we know they perform worse)
    model = imp.reg_logistic_regression
    # we first find the best exponential_lr with lambda fixed to 0
    model_lambda_zero = partial(model, lambda_=0)
    # find the best initial_learning rate for the model
    train_losses = np.zeros(len(INITIAL_GAMMAS))
    test_losses = np.zeros(len(INITIAL_GAMMAS))
    best_initial_gamma, best_loss = 0, np.inf

    for i, initial_gamma in enumerate(INITIAL_GAMMAS):
        exponential_lr = ExponentialLR(initial_gamma=initial_gamma, update_gamma_ratio=1)
        print(f'ExponentialLR : {exponential_lr}')
        loss_tr, loss_te = imp.perform_k_fold_cross_validation(y=y_validate,
                                                               x=x_validate,
                                                               k=k,
                                                               model=model_lambda_zero,
                                                               exponential_lr=exponential_lr,
                                                               max_iters=FOLD_VALIDATION_INITIAL_GAMMA_MAX_ITERS,
                                                               minibatch_size=K_FOLD_MINIBATCH_SIZE)
        train_losses[i] = loss_tr
        test_losses[i] = loss_te
        print(f'ExponentialLR : {exponential_lr}, train loss = {loss_tr}, test loss = {loss_te}')
        if loss_te < best_loss:
            best_loss = loss_te
            best_initial_gamma = initial_gamma
            
    # plot the results
    plt.plot(INITIAL_GAMMAS, train_losses, label='train')
    plt.plot(INITIAL_GAMMAS, test_losses, label='test')
    plt.xlabel('initial_gamma')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(OUTPUT_PATH / 'initial_gammas.png')
    plt.clf()
    # find the best exponential_lr
    train_losses = np.zeros(len(GAMMA_UPDATES))
    test_losses = np.zeros(len(GAMMA_UPDATES))
    best_exponential_lr, best_loss = None, np.inf
    for i, gamma_update in enumerate(GAMMA_UPDATES):
        exponential_lr = ExponentialLR(initial_gamma=best_initial_gamma,
                                       update_gamma_ratio=gamma_update)
        print(f'ExponentialLR : {exponential_lr}')
        loss_tr, loss_te = imp.perform_k_fold_cross_validation(y=y_validate,
                                                               x=x_validate,
                                                               k=k,
                                                               model=model_lambda_zero,
                                                               exponential_lr=exponential_lr,
                                                               max_iters=FOLD_VALIDATION_UPDATE_GAMMA_MAX_ITERS,
                                                               minibatch_size=K_FOLD_MINIBATCH_SIZE)
        train_losses[i] = loss_tr
        test_losses[i] = loss_te
        print(f'ExponentialLR : {exponential_lr}, train loss = {loss_tr}, test loss = {loss_te}')
        if loss_te < best_loss:
            best_loss = loss_te
            best_exponential_lr = exponential_lr
    # plot the results
    plt.plot(GAMMA_UPDATES, train_losses, label='train')
    plt.plot(GAMMA_UPDATES, test_losses, label='test')
    plt.xlabel('gamma_update')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(OUTPUT_PATH / 'gamma_updates.png')
    plt.clf()
    # save best_exponential_lr
    pickle.dump(best_exponential_lr, open(OUTPUT_PATH / 'best_exponential_lr.pickle', 'wb'))
    print(f'Best exponential_lr : {best_exponential_lr}')
    # lambdas:
    # find the best lambda for the model given the best exponential_lr (Nx1 points)
    train_losses = np.zeros(len(LAMBDAS))
    test_losses = np.zeros(len(LAMBDAS))
    best_lambda = None
    best_loss = np.inf
    for i, lambda_ in enumerate(LAMBDAS):
        print(f'Lambda : {lambda_}')
        loss_tr, loss_te = imp.perform_k_fold_cross_validation(y=y_validate,
                                                               x=x_validate,
                                                               k=k,
                                                               model=partial(model, lambda_=lambda_),
                                                               exponential_lr=best_exponential_lr,
                                                               max_iters=FOLD_VALIDATION_LAMBDA_MAX_ITERS,
                                                               minibatch_size=K_FOLD_MINIBATCH_SIZE)
        print(f'Lambda : {lambda_}, train loss = {loss_tr}, test loss = {loss_te}')
        train_losses[i] = loss_tr
        test_losses[i] = loss_te
        if loss_te < best_loss:
            best_loss = loss_te
            best_lambda = lambda_
    print(f'Best lambda : {best_lambda}')
    # plot the losses as a function of lambda
    plt.plot(LAMBDAS, train_losses, label='train')
    plt.plot(LAMBDAS, test_losses, label='test')
    plt.xlabel('lambda')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(OUTPUT_PATH / 'lambdas.png')
    plt.clf()
    best_model = partial(model, lambda_=best_lambda)
    # save best_model
    pickle.dump(best_model, open(OUTPUT_PATH / 'best_model.pickle', 'wb'))
    return best_model, best_exponential_lr


def load_best_model():
    """
    Load the best model and exponential_lr found by the cross validation.
    :return: (model, exponential_lr) the best model and the best exponential_lr
    """
    return (pickle.load(open(OUTPUT_PATH / 'best_model.pickle', 'rb')),
            pickle.load(open(OUTPUT_PATH / 'best_exponential_lr.pickle', 'rb')))


if __name__ == '__main__':
    from run_prepare_datasets import load_datasets

    try:
        _, (x_validate, y_validate), _, _ = load_datasets()
    except FileNotFoundError:
        raise Exception("Datasets not found. Please prepare datasets first !")
    run_cross_validation(x_validate, y_validate)
