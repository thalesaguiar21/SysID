from numpy import array
from random import random


def draft(values, nturns):
    """ Draft n numbers from values in weighted probability
    Args:
        draftables (list): a Draftable list of possible objects
        nturns (int): number of drafts

    Returns:
        drafts (array): array with the indexes of picked values
    """
    chances = _build_chances(values)
    drafts = {_pick(chances, values) for _ in range(nturns)}
    return drafts


def _build_chances(values):
    """ Build an array with the acumulative chance of each value """
    maximum = max(values)
    if maximum == 0:
        return array([i for i in range(len(values))])
    chances = [values[0] / maximum]
    for i in range(1, len(values)):
        chances.append(chances[i - 1] + values[i] / maximum)
    return array(chances)


def _pick(chances, values):
    lucky = random() * chances.max()
    for i in range(len(chances)):
        if lucky <= chances[i]:
            return i
