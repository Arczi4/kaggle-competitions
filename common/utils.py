from typing import Iterator

def get_next_feature_from_iter(features: Iterator):
    try:
        feature = next(features)
        return feature
    except StopIteration:
        print("No features left")
        return