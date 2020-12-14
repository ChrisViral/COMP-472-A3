from __future__ import annotations
import csv
from typing import List, NamedTuple


class Tweet(NamedTuple):
    """
    Tweet tuple, contains it's ID, tokenized text, and class
    """

    id: str
    text: List[str]
    is_factual: bool

    @staticmethod
    def from_row(row: List[str]) -> Tweet:
        """
        Makes a new tweet from a dataset row
        :param row: Row to make the tweet from
        :return:    Created tweet
        """

        # Filter out empty strings from the text
        text: List[str] = list(filter(None, row[1].lower().split(' ')))
        factual: bool = row[2] == "yes"
        return Tweet(row[0], text, factual)


def get_dataset(path: str, has_header: bool = False) -> List[Tweet]:
    """
    Gets the dataset at a specified location
    :param path: Path to the dataset file
    :param has_header: If the first line of the file is a header that must be skipped, defaults to False
    :return: A list of all the loaded tweets
    """

    try:
        print(f"Parsing dataset {path}...")
        with open(path, 'r', newline='', encoding="utf-8") as tsv_file:
            # Open the .tsv file for reading
            reader = csv.reader(tsv_file, delimiter='\t')
            # Skip the first line if necessary
            if has_header:
                next(reader)
            # Get all data: get tweet id, tokenize tweet, remove empty strings, get class
            dataset: List[Tweet] = [Tweet.from_row(row) for row in reader]
            # Print info about the dataset
            print(f'Info for dataset "{path}"')
            print(f"Length:          {len(dataset)}")
            print(f"Word count:      {sum(len(tweet) for tweet in dataset)}")
            print(f'Count for "yes": {sum(1 for tweet in dataset if tweet.is_factual)}')
            print(f'Count for "no":  {sum(1 for tweet in dataset if not tweet.is_factual)}\n')
            return dataset
    except Exception as e:
        print(f"Error while parsing dataset\n{str(e)}")
