"""
File containing the implementation to all ML functions, as well a preprosessing data and training models.
"""
from typing import List
import numpy as np

from exponential_lr import ExponentialLR
from hyperparameters import *

PRINT_INTERVAL = 100

def preprocess_data(x_train: np.ndarray, y_train: np.ndarray, x_final_evaluation: np.ndarray, bias: bool = True, normalize: bool = True) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Remove NaN features in y (datapoints with no label), replace labels {-1, 1} -> {0, 1}
    replace NaNs in x with the mean of the column, kick features that are all NaN in x,
    and normalize the data. Add bias term. Shuffle the thraining dataset.
    :param x_train: train data
    :param y_train: train labels
    :param x_final_evaluation: test data
    :param bias: whether to add a bias term
    :param normalize: whether to normalize the data
    :return: preprocessed data (x_train, y_train, x_final_evaluation)
    """
    # replace labels {-1, 1} -> {0, 1}
    y_train = (y_train + 1) / 2

    # remove NaNs features in x (by replacing them with the mean of the feature on train dataset)
    x_train_NaN_features = np.isnan(x_train)
    replacement_values = np.nanmean(x_train, axis=0)
    x_train[x_train_NaN_features] = np.take(replacement_values, np.where(x_train_NaN_features)[1])
    x_final_evaluation_NaN_features = np.isnan(x_final_evaluation)
    x_final_evaluation[x_final_evaluation_NaN_features] = np.take(replacement_values,np.where(x_final_evaluation_NaN_features)[1])

    # remove features that are still NaN in x
    valid_features = ~np.isnan(x_train).all(axis=0) & ~np.isnan(x_final_evaluation).all(axis=0)
    x_train = x_train[:, valid_features]
    x_final_evaluation = x_final_evaluation[:, valid_features]

    # remove datapoints that have NaN label from train set
    y_not_nan_features = ~np.isnan(y_train)
    x_train, y_train = x_train[y_not_nan_features], y_train[y_not_nan_features]

    # normalize the data and
    # kick 0 variance features (useless features, including the bias term)
    mu, sigma = np.mean(x_train, axis=0), np.std(x_train, axis=0)
    sigma_not_zero = sigma != 0
    # kick 0 variance features
    x_train, x_final_evaluation = x_train[:, sigma_not_zero], x_final_evaluation[:, sigma_not_zero]
    sigma = sigma[sigma_not_zero]
    mu = mu[sigma_not_zero]
    # normalize
    if normalize:
        x_train, x_final_evaluation = (x_train - mu) / sigma, (x_final_evaluation - mu) / sigma

    # add bias term
    if bias:
        x_train = np.c_[np.ones(len(x_train)), x_train]
        x_final_evaluation = np.c_[np.ones(len(x_final_evaluation)), x_final_evaluation]

    # randomize the training data order
    random_indices = np.random.permutation(np.arange(len(y_train)))
    x_train, y_train = x_train[random_indices], y_train[random_indices]
    return x_train, y_train, x_final_evaluation


def split_dataset(x_train: np.ndarray, y_train: np.ndarray, validation_set_ratio: float,test_set_ratio: float) -> ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray),(np.ndarray, np.ndarray)):
    """
    Split the dataset into train, validation and test sets.
    :param x_train: the train data
    :param y_train: the train labels
    :param validation_set_ratio: the proportion of the train set to use for validation
    :param test_set_ratio: the proportion of the train set to use for testing
    :return: (x_train, y_train), (x_validation, y_validation), (x_test, y_test)
    """
    validation_test_size = int(validation_set_ratio * len(y_train))
    test_test_size = int(test_set_ratio * len(y_train))
    x_validation, y_validation = x_train[:validation_test_size], y_train[:validation_test_size]
    x_test, y_test = x_train[validation_test_size:validation_test_size + test_test_size], \
        y_train[validation_test_size:validation_test_size + test_test_size]
    x_train, y_train = x_train[validation_test_size + test_test_size:], y_train[validation_test_size + test_test_size:]
    return (x_train, y_train, x_validation, y_validation, x_test, y_test)



def compute_mse_loss(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> float:
    """
    Compute the MSE loss.
    :param y: the labels
    :param tx: the features
    :param w: the weights
    :return: the loss
    """
    return np.sum((y - tx @ w) ** 2) / (2 * len(y))


def compute_mse_gradient(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Compute the MSE gradient.
    :param y: the labels
    :param tx: the features
    :param w: the weights
    :return: the gradient
    """
    return -tx.T @ (y - tx @ w) / len(y)


def compute_mse_updated_weights(y: np.ndarray, tx: np.ndarray, w: np.ndarray, gamma: float) -> np.ndarray:
    """
    Compute the updated weights.
    :param y: the labels
    :param tx: the features
    :param w: the weights
    :param gamma: the learning rate
    :return: the updated weights
    """
    return w - gamma * compute_mse_gradient(y, tx, w)



def mean_squared_error_gd(y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray, max_iters: int, gamma: float) -> (np.ndarray, float):
    """
    Run gradient descent using MSE.
    :param y: the labels
    :param tx: the features
    :param initial_w: the initial weights
    :param max_iters: the maximum number of iterations
    :param gamma: the learning rate
    :return: the weights and the loss
    """
    w = initial_w
    for n_iter in range(max_iters):
        # update w
        w = compute_mse_updated_weights(y, tx, w, gamma)
        if n_iter % PRINT_INTERVAL == 1:
            loss = compute_mse_loss(y, tx, w)
            print(f'Iteration={n_iter}, loss={loss}', end='\r')
    loss = compute_mse_loss(y, tx, w)
    return w, loss



def mean_squared_error_sgd(y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray, max_iters: int, gamma: float) -> (np.ndarray, float):
    """
    Run stochastic gradient descent using MSE.
    :param y: the labels
    :param tx: the features
    :param initial_w: the initial weights
    :param max_iters: the maximum number of iterations
    :param gamma: the learning rate
    :return: the weights and the loss
    """
    mini_batch_size = 1
    w = initial_w
    for n_iter in range(max_iters):
        # sample a mini-batch
        mini_batch = np.random.choice(len(y), mini_batch_size)
        y_mini_batch = y[mini_batch]
        tx_mini_batch = tx[mini_batch]
        # update w
        w = compute_mse_updated_weights(y_mini_batch, tx_mini_batch, w, gamma)
        if n_iter % PRINT_INTERVAL == 1:
            loss = compute_mse_loss(y, tx, w)
            print(f'Iteration={n_iter}, loss={loss}', end='\r')
    loss = compute_mse_loss(y, tx, w)
    return w, loss



def least_squares(y: np.ndarray, tx: np.ndarray) -> (np.ndarray, float):
    """
    Calculate the least squares solution.
    :param y: the labels
    :param tx: the features
    :return: the weights and the loss
    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    return w, compute_mse_loss(y, tx, w)



def ridge_regression(y: np.ndarray, tx: np.ndarray, lambda_: float) -> (np.ndarray, float):
    """
    Calculate the ridge regression solution.
    :param y: the labels
    :param tx: the features
    :param lambda_: the regularization parameter
    :return: the weights and the loss
    """
    _, n_features = tx.shape
    # Compute the Ridge regression weights
    w = np.linalg.solve(tx.T @ tx + 2 * tx.shape[0] * lambda_ * np.eye(n_features), tx.T @ y)
    # Compute the loss with the updated weights
    loss = compute_mse_loss(y, tx, w)
    return w, loss


def compute_sigmoid(tx: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid function (for logistic regression).
    :param tx: the features
    :param w: the weights
    :return: the sigmoid function
    """
    return 1 / (1 + np.exp(-tx @ w))


def compute_logistic_loss(y: np.ndarray, tx: np.ndarray, w: np.ndarray, lambda_: float = 0) -> float:
    """
    Compute the logistic loss.
    :param y: the labels
    :param tx: the features
    :param w: the weights
    :param lambda_: the regularization parameter
    :return: the loss
    """
    y_pred = compute_sigmoid(tx, w)
    # Add epsilon to prevent log(0) or log(1)
    eps = 1e-9
    y_pred = np.clip(y_pred, eps, 1 - eps)
    regularization = lambda_ * np.linalg.norm(w) ** 2
    return np.sum(-y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)) / len(y) + regularization


def compute_logistic_gradient(y: np.ndarray, tx: np.ndarray, w: np.ndarray, lambda_: float = 0) -> np.ndarray:
    """
    Compute the logistic gradient.
    :param y: the labels
    :param tx: the features
    :param w: the weights
    :param lambda_: the regularization parameter
    :return: the gradient
    """
    y_pred = compute_sigmoid(tx, w)
    regularization = 2 * lambda_ * w
    return tx.T @ (y_pred - y) / len(y) + regularization


def compute_logistic_updated_weights(y: np.ndarray, tx: np.ndarray, w: np.ndarray, gamma: float, lambda_: float = 0) -> np.ndarray:
    """
    Compute the updated weights.
    :param y: the labels
    :param tx: the features
    :param w: the weights
    :param lambda_: the regularization parameter
    :param gamma: the learning rate
    :return: the updated weights
    """
    return w - gamma * compute_logistic_gradient(y, tx, w, lambda_=lambda_)



def logistic_regression(y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray, max_iters: int, gamma: float) -> (np.ndarray, float):
    """
    Calculate the logistic regression solution.
    :param y: the labels
    :param tx: the features
    :param initial_w: the initial weights
    :param max_iters: the maximum number of iterations
    :param gamma: the learning rate
    :return: the weights and the loss
    """
    w = initial_w
    for n_iter in range(max_iters):
        # update w
        w = compute_logistic_updated_weights(y, tx, w, gamma)
        if n_iter % PRINT_INTERVAL == 1:
            loss = compute_logistic_loss(y, tx, w)
            print(f'Iteration={n_iter}, loss={loss}', end='\r')
    loss = compute_logistic_loss(y, tx, w)
    return w, loss



def reg_logistic_regression(y: np.ndarray, tx: np.ndarray, lambda_: float, initial_w: np.ndarray, max_iters: int, gamma: float) -> (np.ndarray, float):
    """
    Calculate the regularized logistic regression solution.
    :param y: the labels
    :param tx: the features
    :param lambda_: the regularization parameter
    :param initial_w: the initial weights
    :param max_iters: the maximum number of iterations
    :param gamma: the learning rate
    :return: the weights and the loss
    """
    w = initial_w
    for n_iter in range(max_iters):
        # update w
        w = compute_logistic_updated_weights(y, tx, w, gamma, lambda_=lambda_)
        if n_iter % PRINT_INTERVAL == 1:
            loss = compute_logistic_loss(y, tx, w, lambda_=lambda_)
            print(f'Iteration={n_iter}, loss={loss}', end='\r')
    loss = compute_logistic_loss(y, tx, w)
    return w, loss


def train_model(model: '(tx:np.ndarray, y:np.ndarray, w:np.ndarray, gamma:float) -> '
                       '(np.ndarray, float)',
                y_train: np.ndarray,
                x_train: np.ndarray,
                y_test: np.ndarray,
                x_test: np.ndarray,
                max_iters: int,
                exponential_lr: ExponentialLR,
                minibatch_size: int) -> (np.ndarray, List[float], List[float]):
    """
    Train the given model on the given data.
    :param model: the model to train
    :param loss_function: the loss function of the model
    :param y_train: the train labels
    :param x_train: the train data
    :param y_test: the test labels
    :param x_test: the test data
    :param max_iters: the maximum number of iterations
    :param exponential_lr: the exponential_lr to use
    :param minibatch_size: the size of the minibatches (1 for full sgd)
    :return: the weights of the trained model, the train losses and the test losses
    """
    exponential_lr.reset()
    w = np.random.rand(x_train.shape[1])
    train_losses, test_losses = [], []

    print(
        f"""Training model {str(model)} with exponential_lr {str(exponential_lr)} (max_iters={max_iters})...
        Initial loss: 
            train_loss={model(y=y_train, tx=x_train, initial_w=w, gamma=0., max_iters=0)[1]}
            test_loss={model(y=y_test, tx=x_test, initial_w=w, gamma=0., max_iters=0)[1]}'
        """
    )

    for i in range(max_iters):
        # update w
        minibatch_indices = np.random.choice(len(y_train), size=minibatch_size, replace=False)
        y = y_train[minibatch_indices]
        x = x_train[minibatch_indices]
        w, _ = model(y=y, tx=x, initial_w=w, max_iters=1, gamma=exponential_lr.gamma)
        exponential_lr.step()
        if i % PRINT_INTERVAL == 1:
            _, test_loss = model(y=y_test, tx=x_test, initial_w=w, max_iters=0, gamma=exponential_lr.gamma)
            train_losses.append(test_loss)

            _, train_loss = model(y=y_train, tx=x_train, initial_w=w, max_iters=0, gamma=exponential_lr.gamma)
            test_losses.append(train_loss)
            
            print(f'Iteration={i}/{max_iters}, train_loss={train_loss}, test_loss={test_loss}, lr={exponential_lr.gamma}',end='\r')

    print(
        f"""final loss:
        train_loss={train_losses[-1]}
        test_loss={test_losses[-1]}
        """
        )
    return w, train_losses, test_losses

def perform_k_fold_cross_validation(y: np.ndarray,
                                    x: np.ndarray,
                                    k: int,
                                    model: '(np.ndarray, np.ndarray, np.ndarray, float) -> '
                                           '(np.ndarray, float)',
                                    exponential_lr: ExponentialLR,
                                    max_iters: int,
                                    minibatch_size: int) -> (float, float):
    """
    Perform cross validation on the given data on the given models (use partial to specify the hyperparameters).
    :param y: the labels
    :param x: the data
    :param k: the number of folds
    :param model: the function to validate. (y, tx, w, gamma) -> (w, loss)
    :param exponential_lr: the exponential_lr to use
    :param max_iters: maximum number of iterations
    :param minibatch_size: the size of the minibatches (1 for full sgd)
    :return: (average loss on train data, average loss on test data)
    """
    folds = np.array_split(np.random.permutation(np.arange(len(y))), k)
    cumulated_loss_tr, cumulated_loss_te = 0, 0
    for i in range(len(folds)):
        train_indices = np.concatenate(folds[:i] + folds[i + 1:])
        test_indices = folds[i]
        x_train, y_train = x[train_indices], y[train_indices]
        x_test, y_test = x[test_indices], y[test_indices]
        w, train_loss, test_loss = train_model(model=model,
                                               y_train=y_train,
                                               x_train=x_train,
                                               y_test=y_test,
                                               x_test=x_test,
                                               max_iters=max_iters,
                                               exponential_lr=exponential_lr,
                                               minibatch_size=minibatch_size)
        cumulated_loss_tr += train_loss[-1]
        cumulated_loss_te += test_loss[-1]
        print(f'Fold {i + 1} / {k} : train_loss={train_loss[-1]}, test_loss={test_loss[-1]}')
    return cumulated_loss_tr / k, cumulated_loss_te / k
