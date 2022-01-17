from pathlib import Path


class Paths:
    data = Path('data')
    data_train = data / 'train'
    data_test = data / 'test'
    data_validation = data / 'validation'
    validation_labels = data_validation / 'labels.json'

    output = Path('output')

