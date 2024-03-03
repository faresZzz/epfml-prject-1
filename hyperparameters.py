import pathlib

import numpy as np

# ------------------ K-fold cross validation ------------------
K = 6
K_FOLD_MINIBATCH_SIZE = 100  # size of the minibatches for kfold cross validation
# maximum number of iterations for finding initial gamma in cross validation
FOLD_VALIDATION_INITIAL_GAMMA_MAX_ITERS = 200
# maximum number of iterations for finding update gamma in cross validation
FOLD_VALIDATION_UPDATE_GAMMA_MAX_ITERS = 10000
# maximum number of iterations for finding lambda in cross validation
FOLD_VALIDATION_LAMBDA_MAX_ITERS = 5000
N_POINTS = 20  # number of points to test for each hyperparameter
INITIAL_GAMMAS = np.linspace(0.3, 0.8, N_POINTS)
GAMMA_UPDATES = np.linspace(.9995, 1, N_POINTS)
LAMBDAS = np.linspace(0, 0.004, N_POINTS)

# ------------------ final training ------------------
FINAL_TRAINING_MAX_ITERS = 10000  # maximum number of iterations for final training
MINIBATCH_SIZE = 10000  # size of the minibatches for final training

# ------------------ dataset ------------------
VALIDATION_SET_RATIO = .1  # proportion of the training set used for cross validation
TEST_SET_RATIO = .2  # proportion of the training set used for testing
MAX_ROWS = None  # maximum number of rows to load from the training data (Set to None to load all rows)
OUTPUT_PATH = pathlib.Path(__file__).resolve().parent / 'output'
