from functools import partial

from run_prepare_datasets import run_prepare_datasets
from run_cross_validation import run_cross_validation
from run_final_training import run_final_training
from run_evaluate_final_model import run_evaluate_final_model


def run():
    """
    Run the whole pipeline.
    - Prepare the datasets
        - Load the datasets
        - Preprocess the datasets (NaNs, standardization, bias, ...)
        - Split the training datasets (training, validation, test)
        - Save the datasets to disk (to execute other scripts without having to recompute the datasets)
    - Run cross validation
        - Tries every interesting combination of hyperparameters (gamma, lambda)
        for all the models (mse gd, mse sgd, logistic regression, reg logistic regression)
        - Plot the results
        - Save the model with the best hyperparameters to disk as a pickled partial function
        (to execute other scripts without having to recompute the cross validation)
    - Train the final model
        - Train the best model found by the cross validation on the training data
        - Plot the loss (test and train) as a function of the number of iterations
        - Save the weights of the trained model to disk (to execute other scripts without having to recompute the training)
    - Evaluate the final model
        - Evaluate the final model on the test set
        - Prints the recall accuracy and precision, TP, FP, FN, TN
        - Generate the submission file
    :return:
    """
    print('Preparing datasets...')
    datasets = run_prepare_datasets()
    print('Datasets prepared.')


    print('Running cross validation...')
    model, exponential_lr = run_cross_validation(x_validate=datasets["x_validate"], y_validate=datasets["y_validate"])
    print('Cross validation done.')


    print('Training final model...')
    w = run_final_training(x_train=datasets["x_train"], y_train=datasets["y_train"], x_test=datasets["x_test"], y_test=datasets["y_test"], model=model, exponential_lr=exponential_lr)
    print('Final model trained.')


    print('Evaluating final model...')
    run_evaluate_final_model(w=w, x_test=datasets["x_test"], y_test=datasets["y_test"], x_final_evaluation=datasets["x_final_evaluation"])
    print('Final model evaluated.')


if __name__ == '__main__':
    run()
