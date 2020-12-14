import math
from collections import Counter, defaultdict
from datasets import Tweet
from itertools import chain
from metrics import *
from typing import List, DefaultDict, Tuple, Final


class Class:
    """
    Naive-Bayes Classifier result Class
    """

    def __init__(self, vocabulary: Counter, tweets: List[List[str]], total_tweets: int, delta: float) -> None:
        """
        Creates a new category for a specific class
        :param vocabulary:   Vocabulary of the classifier
        :param tweets:       Tweet texts in the class
        :param total_tweets: Total amount of tweets in the classifier
        :param delta:        Smoothing parameter
        """

        # Calculate the prior
        self.prior: Final[float] = math.log10(len(tweets) / total_tweets)

        # Calculate the word frequency within the class
        frequencies: Counter[str] = Counter()
        word_count: float = 0.0
        for tweet in tweets:
            word_count += len(tweet)
            frequencies.update(filter(vocabulary.__contains__, tweet))

        # Add smoothing to the word count
        word_count += len(vocabulary) * delta
        # Calculate the conditionals
        self.conditionals: Final[DefaultDict[str, float]] = defaultdict(float)
        for word in vocabulary.keys():
            self.conditionals[word] = math.log10((frequencies[word] + delta) / word_count)

    def score(self, tweet: List[str]):
        """
        Calculates the score of a tweet
        :param tweet: Tweet text to score
        :return:      The score of the tweet
        """

        return self.prior + sum(self.conditionals[word] for word in tweet)


class NaiveBayesBOW:
    """
    A Naive-Bayes Bag of Words classifier
    """

    def __init__(self, tweets: List[Tweet], min_occurrence: int = 1, delta: float = 0.01, name: str = "NB-BOW") -> None:
        """
        Creates a new Classifier with the specified documents
        :param tweets:         Tweets to create the classifier with
        :param min_occurrence: Minimum number of occurrences for a word to be considered in the vocabulary, defaults to 1
        :param delta:          Smoothing parameter, defaults to 0.01
        :param name:           Name of the model for trace and file storage purpose
        """

        # Start by creating vocabulary
        print(f"Training Naive-Bayes Bag-of-Words model for {name}...")
        self.name: str = name
        self.vocabulary: Counter[str] = Counter(chain.from_iterable([tweet.text for tweet in tweets]))

        # Filter out words that do not appear enough
        filtered: List[str] = []
        for word, count in self.vocabulary.items():
            if count < min_occurrence:
                filtered.append(word)

        # Remove filtered words
        for word in filtered:
            del self.vocabulary[word]

        # Create all classes
        total_tweets: int = len(tweets)
        self.factual: Class = Class(self.vocabulary, [tweet.text for tweet in tweets if tweet.is_factual], total_tweets, delta)
        self.not_factual: Class = Class(self.vocabulary, [tweet.text for tweet in tweets if not tweet.is_factual], total_tweets, delta)

    def classify(self, tweet: Tweet) -> Tuple[bool, float]:
        """
        Classifies a given tweet into it's most likely class, factual (True), or not factual (False)
        :param tweet: The tweet to classify
        :return:      The predicted class of the tweet and it's resulting score
        """

        # Calculate both scores, return largest
        score_factual: float = self.factual.score(tweet.text)
        score_not_factual: float = self.not_factual.score(tweet.text)
        return score_factual > score_not_factual, max(score_factual, score_not_factual)

    def classify_all(self, tweets: List[Tweet], folder: str) -> None:
        """
        Classifies all the passed tweets and writes the results to the disk
        :param tweets: Tweets to classify
        :param folder: Folder to store the results in
        """

        # Store results
        print(f"\nClassifying test set for Naive-Bayes Bag-of-Words model {self.name}...")
        results: List[bool] = []
        with open(f"{folder}/trace_{self.name}.txt", 'w') as f:
            # Classify each tweet
            for tweet in tweets:
                result, score = self.classify(tweet)
                correct: bool = result == tweet.is_factual
                f.write(f"{tweet.id}  {'yes' if result else 'no'}  {score:.2E}  {'yes' if tweet.is_factual else 'no'}  {'right' if correct else 'wrong'}\r")
                results.append(result)

        # Store metrics
        print(f"Saving metrics data for Naive-Bayes Bag-of-Words model {self.name}...")
        expected: List[bool] = [tweet.is_factual for tweet in tweets]
        with open(f"{folder}/eval_{self.name}.txt", 'w') as f:
            f.write(f"{accuracy(results, expected)}\r")
            f.write(f"{precision(results, expected, True)}  {precision(results, expected, False)}\r")
            f.write(f"{recall(results, expected, True)}  {recall(results, expected, False)}\r")
            f.write(f"{f1_measure(results, expected, True)}  {f1_measure(results, expected, False)}\r")
