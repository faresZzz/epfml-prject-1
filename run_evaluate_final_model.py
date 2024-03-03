import numpy as np
from matplotlib import pyplot as plt

import implementations as imp
from hyperparameters import *
import material.helpers as h

OUTPUT_PATH = OUTPUT_PATH / 'final_evaluation'


def run_evaluate_final_model(w: np.ndarray,
                             x_test: np.ndarray,
                             y_test: np.ndarray,
                             x_final_evaluation: np.ndarray):
    """
    Evaluate the final model on the test set and generate the submission file.
    :param x_test: test data
    :param y_test: test labels
    :param x_final_evaluation: final evaluation data
    :param model: the model to evaluate
    :param loss_function: the loss of the function to evaluate
    :return: the loss on the test set
    """
    np.random.seed(0)
    if not OUTPUT_PATH.exists():
        OUTPUT_PATH.mkdir()
    print('Reading final submission IDs')
    _, _, _, _, final_submission_ids = h.load_csv_data('material/data')
    y_pred_test_unrounded = np.float32(1 / (1 + np.exp(-x_test @ w)))
    y_final_evaluation_unrounded = np.float32(1 / (1 + np.exp(-x_final_evaluation @ w)))
    best_f1 = 0
    f1_scores = []
    thresholds = np.linspace(0, 0.5, 100)
    final_eval_pred_best = None
    for threshold in thresholds:
        print('-' * 20)
        print(f'Final evaluation for threshold {threshold} :')
        y_pred = np.float32(y_pred_test_unrounded > threshold)
        final_eval_pred = np.float32(y_final_evaluation_unrounded > threshold)
        # evaluate recall accuracy and precision
        tp = y_test[y_pred == 1].sum()
        fp = len(y_pred[y_pred == 1]) - tp
        fn = len(y_test[y_test == 1]) - tp
        tn = len(y_test[y_test == 0]) - fp
        recall = np.float32(tp / (tp + fn))
        accuracy = np.float32((tp + tn) / (tp + tn + fp + fn))
        precision = np.float32(tp / (tp + fp))
        f1score = 2 * (precision * recall) / (precision + recall)
        f1_scores.append(f1score)
        print(f'TP  |  FP\n{tp}|{fp}')
        print(f'FN  |  TN\n{fn}|{tn}')
        print(f'Recall : {recall}')
        print(f'Accuracy : {accuracy}')
        print(f'Precision : {precision}')
        print(f'F1 score : {f1score}')
        if best_f1 < f1score:
            best_f1 = f1score
            final_eval_pred_best = final_eval_pred
    # plot the results
    plt.plot(thresholds, f1_scores)
    plt.xlabel('threshold')
    plt.ylabel('F1 score')
    plt.savefig(OUTPUT_PATH / 'thresholds.png')
    plt.clf()
    print('-' * 20)
    print(f'Best F1 score : {best_f1}')
    # generate submission file
    print(f'Generating submission file...')
    h.create_csv_submission(final_submission_ids, final_eval_pred_best * 2 - 1,
                            (OUTPUT_PATH / f'submission.csv').resolve())
    print('Done.')


if __name__ == '__main__':
    from run_prepare_datasets import load_datasets
    from run_final_training import load_best_weights

    try:
        _, _, (x_test, y_test), x_final_evaluation = load_datasets()
    except FileNotFoundError:
        raise Exception('Datasets not found. Please prepare dataset first !')
    try:
        w = load_best_weights()
    except FileNotFoundError:
        raise Exception('Best weights not found. Please train final model first !')
    run_evaluate_final_model(w=w,
                             x_test=x_test,
                             y_test=y_test,
                             x_final_evaluation=x_final_evaluation)
