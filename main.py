import sys
from classifier import NaiveBayesBOW
from datasets import get_dataset, Tweet
from pathlib import Path
from typing import List


def main(args: List[str]) -> None:
    """
    Program entry point
    :param args: Program arguments
    """

    # Get path to training and testing files
    training_path: str = args[1] if len(args) >= 2 else "data/covid_training.tsv"
    testing_path: str  = args[2] if len(args) >= 3 else "data/covid_test_public.tsv"

    # Get datasets
    training: List[Tweet] = get_dataset(training_path, has_header=True)
    test: List[Tweet] = get_dataset(testing_path)
    # Train models
    original: NaiveBayesBOW = NaiveBayesBOW(training, name="NB-BOW-OV")
    filtered: NaiveBayesBOW = NaiveBayesBOW(training, name="NB-BOW-FV", min_occurrence=2)

    # Ensure results directory exists
    Path("results/").mkdir(parents=True, exist_ok=True)
    # Testing
    original.classify_all(test, "results")
    filtered.classify_all(test, "results")

    # Exit
    print("\nData processing complete!")


if __name__ == "__main__":
    main(sys.argv)
