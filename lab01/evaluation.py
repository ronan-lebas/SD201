from typing import List

def precision_recall(expected_results: List[bool], actual_results: List[bool]) -> (float, float):
    """Compute the precision and recall of a series of predictions

    Parameters
    ----------
        expected_results : List[bool]
            The true results, that is the results that the predictor
            should have find.
        actual_results : List[bool]
            The predicted results, that have to be evaluated.

    Returns
    -------
        float
            The precision of the predicted results.
        float
            The recall of the predicted results.
    """
    true_pos, false_pos, false_neg = 0,0,0
    for i in range(len(expected_results)):
        if expected_results[i] & actual_results[i]:
            true_pos += 1
        elif expected_results[i] & (not actual_results[i]):
            false_neg += 1
        elif (not expected_results[i]) & actual_results[i]:
            false_pos += 1
    precision = true_pos/(true_pos+false_pos) if true_pos+false_pos != 0 else 0
    recall = true_pos/(true_pos+false_neg) if true_pos+false_neg != 0 else 0
    return (precision, recall)
    
def F1_score(expected_results: List[bool], actual_results: List[bool]) -> float:
    """Compute the F1-score of a series of predictions

    Parameters
    ----------
        expected_results : List[bool]
            The true results, that is the results that the predictor
            should have find.
        actual_results : List[bool]
            The predicted results, that have to be evaluated.

    Returns
    -------
        float
            The F1-score of the predicted results.
    """
    p, r = precision_recall(expected_results, actual_results)
    return 2*r*p/(r+p) if r+p != 0 else 0