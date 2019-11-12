from math import pi


def clip(value, a_min, a_max):
    return max(a_min, min(value, a_max))


def rad(degree):
    return pi * degree / 180


def degree(rad):
    return 180 * rad / pi
