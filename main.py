from classifier import NaiveBayesBOW
from datasets import get_dataset, Tweet
from pathlib import Path
from typing import List


def main() -> None:
    # Get training set and train models
    training: List[Tweet] = get_dataset("data/covid_training.tsv", has_header=True)
    original: NaiveBayesBOW = NaiveBayesBOW(training)
    filtered: NaiveBayesBOW = NaiveBayesBOW(training, min_occurrence=2)

    # Ensure results directory exists
    results_directory = "results/"
    Path(results_directory).mkdir(parents=True, exist_ok=True)

    # Get testing set and verify
    test: List[Tweet] = get_dataset("data/covid_test_public.tsv")
    original.classify_all(test, "results", "NB-BOW-OV.txt")
    filtered.classify_all(test, "results", "NB-BOW-FV.txt")
    pass


if __name__ == "__main__":
    main()
