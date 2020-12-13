import csv
from typing import List, NamedTuple


class Tweet(NamedTuple):
    """
    Tweet tuple, contains it's ID, tokenized text, and class
    """

    id: str
    text: List[str]
    is_factual: bool


def get_dataset(path: str, has_header: bool = False) -> List[Tweet]:
    """
    Gets the dataset at a specified location
    :param path: Path to the dataset file
    :param has_header: If the first line of the file is a header that must be skipped, defaults to False
    :return: A list of all the loaded tweets
    """

    with open(path, 'r', newline='', encoding="utf-8") as tsv_file:
        # Open the .tsv file for reading
        reader = csv.reader(tsv_file, delimiter='\t')
        # Skip the first line if necessary
        if has_header:
            next(reader)
        # Get all data: get tweet id, tokenize tweet, remove empty strings, get class
        return [Tweet(row[0], list(filter(None, row[1].lower().split(' '))), row[2] == "yes") for row in reader]