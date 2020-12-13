from typing import List


def accuracy(results: List[bool], expected: List[bool]) -> float:
    """
    Calculates the accuracy of a set of data
    :param results: Target results data
    :param expected: Output results data
    :return: The accuracy of the results
    """

    n: int = len(results)
    return sum(1 for i in range(n) if results[i] == expected[i]) / n


def precision(results: List[bool], expected: List[bool], target_class: bool) -> float:
    """
    Calculates the precision value of a set of results for a given class
    :param results:       Resulting class predictions
    :param expected:      Expected classes
    :param target_class:  Class to calculate the precision for
    :return:              The precision of the results for the specified class
    """

    # Go through all result pairs
    labelled: int = 0
    true_positive: int = 0
    for result, target in zip(results, expected):
        if result == target_class:
            # Count all results labelled as the specified class
            labelled += 1
            if result == target:
                # And all results correctly labelled
                true_positive += 1
    # Return the precision value for the class
    return true_positive / labelled if labelled != 0 else 0.0


def recall(results: List[bool], expected: List[bool], target_class: bool) -> float:
    """
    Calculates the recall value of a set of results for a given class
    :param results:       Resulting class predictions
    :param expected:      Expected classes
    :param target_class:  Class to calculate the recall for
    :return:              The recall of the results for the specified class
    """

    # Go through all result pairs
    actual: int = 0
    true_positive: int = 0
    for result, target in zip(results, expected):
        if target == target_class:
            # Count all results actually in the target class
            actual += 1
            if result == target:
                # And all results correctly labelled
                true_positive += 1
    # Return the precision value for the class
    return true_positive / actual if actual != 0 else 0.0


def f1_measure(results: List[bool], expected: List[bool], target_class: bool, b: float = 1.0) -> float:
    """
    Calculates the F1-measure of a set of data for one class
    :param results:      Resulting class predictions
    :param expected:     Expected classes
    :param target_class: Class to calculate the F1-measure for
    :param b:            Beta value to use, b>1 gives more importance to recall, b<1 gives more importance to precision, b=1 puts both equal. Defaults to 1
    :return:             The F1-measure of the results for the specified class
    """

    # Calculate precision and recall
    p: float = precision(expected, results, target_class)
    r: float = recall(expected, results, target_class)
    b **= 2

    try:
        # Calculate F1-measure
        return ((b + 1) * p * r) / ((b * p) + r)
    except ZeroDivisionError:
        # If a division by zero happens, return 0
        return 0.0
