import warnings

from temporal_diff.temporal_diff import main

if __name__ == '__main__':
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    warnings.filterwarnings("ignore", message="divide by zero encountered")
    main(dataset='val')
    # main(dataset='test')
