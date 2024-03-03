"""
Used in order to prepares all the training testing and validation datasets from the provided datas
First preprosess the data then split them
"""
from pathlib import Path
from hyperparameters import *
from implementations import preprocess_data, split_dataset
import numpy as np
import material.helpers as helper

OUTPUT_PATH = OUTPUT_PATH / 'datasets'


def create_directories(output_path: Path):
    if not output_path.parent.exists():
        output_path.parent.mkdir()
    if not output_path.exists():
        output_path.mkdir()

def save_preprocessed_datasets(output_path: Path, datasets:dict):
    print('Saving preprocessed datasets...')
    for name, dataset in datasets.items():
        np.save(output_path / f'{name}.npy', dataset)
    print('Done.')

def run_prepare_datasets() -> ((np.ndarray, np.ndarray),
                               (np.ndarray, np.ndarray),
                               (np.ndarray, np.ndarray),
                               np.ndarray):
    """
    Load the datasets prepares them and saves them preprocessed to disk.
    :return: the training, validation, test and final evaluation datasets
    (x_train, y_train), (x_validate, y_validate), (x_test, y_test), x_final_evaluation
    """
    create_directories(OUTPUT_PATH)
    np.random.seed(0)

    helper.load_csv_data('material/data')
    x_train, x_final_evaluation, y_train, train_ids, final_evaluation_ids = helper.load_csv_data('material/data')

    x_train, y_train, x_final_evaluation = preprocess_data(x_train, y_train, x_final_evaluation)
    datasets = { name : dataset for name , dataset in zip(
        ("x_train", "y_train", "x_validate", "y_validate", "x_test", "y_test"), 
        split_dataset(x_train, y_train, validation_set_ratio=VALIDATION_SET_RATIO, test_set_ratio=TEST_SET_RATIO))
        }

    datasets["x_final_evaluation"] = x_final_evaluation
    save_preprocessed_datasets(OUTPUT_PATH, datasets)

    
    return datasets

def load_datasets() -> ((np.ndarray, np.ndarray),
                        (np.ndarray, np.ndarray),
                        (np.ndarray, np.ndarray),
                        np.ndarray):
    """
    Load the datasets from disk.
    :return: the training, validation, test and final evaluation datasets
    (x_train, y_train), (x_validate, y_validate), (x_test, y_test), x_final_evaluation
    """
    x_train = np.load(OUTPUT_PATH / 'x_train.npy')
    y_train = np.load(OUTPUT_PATH / 'y_train.npy')
    x_validate = np.load(OUTPUT_PATH / 'x_validate.npy')
    y_validate = np.load(OUTPUT_PATH / 'y_validate.npy')
    x_test = np.load(OUTPUT_PATH / 'x_test.npy')
    y_test = np.load(OUTPUT_PATH / 'y_test.npy')
    x_final_evaluation = np.load(OUTPUT_PATH / 'x_final_evaluation.npy')
    return (x_train, y_train), (x_validate, y_validate), (x_test, y_test), x_final_evaluation

if __name__ == '__main__':
    print('Preparing datasets...')
    run_prepare_datasets()
    print('Datasets prepared.')

