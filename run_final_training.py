from functools import partial

import matplotlib.pyplot as plt

import implementations as imp
from hyperparameters import *
from run_cross_validation import load_best_model

OUTPUT_PATH = OUTPUT_PATH / 'final_training'


def load_best_weights() -> np.ndarray:
    """
    Load the weights of the best model found by the cross validation.
    :return: the weights of the best model
    """
    return np.load(OUTPUT_PATH / 'w.npy')


def run_final_training(x_train: np.ndarray,
                       y_train: np.ndarray,
                       x_test: np.ndarray,
                       y_test: np.ndarray,
                       model, exponential_lr) -> np.ndarray:
    """
    Train the final model on the training data and evaluate it on the test data.
    :param x_test: the test data
    :param y_test: the test labels
    :param x_train: the training data
    :param y_train: the training labels
    :param model: the model to train
    :param exponential_lr: the exponential_lr to use
    :return: the weights of the trained model
    """
    np.random.seed(0)
    if not OUTPUT_PATH.exists():
        OUTPUT_PATH.mkdir()
    # hard set model after analysing cross validation results :
    w, train_losses, test_losses = imp.train_model(y_train=y_train,
                                                   x_train=x_train,
                                                   y_test=y_test,
                                                   x_test=x_test,
                                                   max_iters=FINAL_TRAINING_MAX_ITERS,
                                                   model=model,
                                                   exponential_lr=exponential_lr,
                                                   minibatch_size=MINIBATCH_SIZE)

    # save the model
    np.save(OUTPUT_PATH / 'w.npy', w)
    # save the losses for plotting latter
    np.save(OUTPUT_PATH / 'train_losses.npy', train_losses)
    np.save(OUTPUT_PATH / 'test_losses.npy', test_losses)
    plt.plot(range(0, FINAL_TRAINING_MAX_ITERS, 100), train_losses, label='train')
    plt.plot(range(0, FINAL_TRAINING_MAX_ITERS, 100), test_losses, label='test')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(OUTPUT_PATH / f'training_loss.png')
    plt.clf()
    return w


if __name__ == '__main__':
    from run_prepare_datasets import load_datasets

    try:
        ((x_train, y_train),
         _,
         (x_test, y_test),
         x_final_evaluation) = load_datasets()
    except FileNotFoundError:
        raise Exception('Datasets not found. Please prepare dataset first !')
    try:
        model, exponential_lr = load_best_model()
    except FileNotFoundError:
        raise Exception('Cross validation not done. Please run cross validation first !')
    run_final_training(x_train=x_train,
                       y_train=y_train,
                       x_test=x_test,
                       y_test=y_test,
                       model=model,
                       exponential_lr=exponential_lr)
