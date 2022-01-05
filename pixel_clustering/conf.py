from pathlib import Path


class Paths:
    data = Path.home() / 'data_WiSAR/data'
    data_train = data / 'train'
    data_test = data / 'test'
    validation_labels = data / 'validation/labels.json'

    output = Path.home() / 'data_WiSAR/output1'

