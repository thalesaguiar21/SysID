from sample.utils import clip, degree, rad
from math import sqrt, atan, exp, pi
from numpy import prod

FIELDMIN = -0.998
FIELDMAX = 0.998
NOT_VISIBLE = 999.999
CAM_NOISE = rad(1)
FALSE_POS = 0.03
FALSE_NEG = 0.005


class Robot:

    def __init__(self, x, y, theta, weight=1.):
        self.x = x
        self.y = y
        self.ang = theta
        self.weight = weight

    def __lt__(self, other):
        return self.weight < other.weight


class Coord:

    def __init__(self, x, y):
        self.x = x
        self.y = y


MARKERS = [Coord(0, 1), Coord(0, -1), Coord(1, 0), Coord(-1, 0)]


def iscolision(r1, r2):
    return r1.x == r2.x and r1.y == r2.y


def put_in_limits(robot):
    """ Clip the robot atributes to the field and vision limits """
    robot.x = clip(robot.x, FIELDMIN, FIELDMAX)
    robot.y = clip(robot.y, FIELDMIN, FIELDMAX)
    if robot.ang is not None:
        robot.ang = clip(robot.ang, -180, 180)


def visibility(robot, other):
    if _distance(robot, other) != NOT_VISIBLE:
        return _angle_between(robot, other)
    return NOT_VISIBLE


def _distance(robot, other):
    sqr_dx = (other.x - robot.x) ** 2.
    sqr_dy = (other.y - robot.y) ** 2.
    dist = sqrt(sqr_dx + sqr_dy)
    if dist < 0.1 or dist > 2.0:
        return NOT_VISIBLE
    else:
        return dist


def _angle_between(robot, other):
    """ Compute the angle between two robots with respect to x axis """
    dy = other.y - robot.y
    dx = other.x - robot.x
    if dx == 0:
        ang = -1 * robot.ang
    else:
        ang = degree(atan(dy / dx) - robot.ang)
    if ang < -90 or ang > 90:
        return NOT_VISIBLE
    else:
        return rad(ang)


def robot_similarity(robot_estims, measures):
    product = 1.
    for est, meas in zip(robot_estims, measures):
        if est == NOT_VISIBLE and meas == NOT_VISIBLE:
            product *= 1.
        elif est == NOT_VISIBLE and meas != NOT_VISIBLE:
            product *= FALSE_NEG
        elif est != NOT_VISIBLE and meas == NOT_VISIBLE:
            product *= FALSE_POS
        else:
            dmeasure = meas - est
            denom = sqrt(2. * pi * CAM_NOISE ** 2)
            product *= exp(-(0.5 * dmeasure / CAM_NOISE) ** 2) / denom
    return product


def addrobot(r, robots):
    if not hascolision(r, robots):
        return robots.append(r)


def hascolision(r, robots):
    for robot in robots:
        if iscolision(r, robot):
            return True
    return False


class Particle(object):
    """ """

    def __init__(self, robots=[], weight=1.):
        self.robots = robots
        self.weight = weight

    def __str__(self):
        rstr = [str(robot) for robot in self.robots]
        return '\n'.join(rstr)

    def __lt__(self, other):
        return self.weight < other.weight

    def robots_xs(self):
        return [r.x for r in self.robots]

    def robots_ys(self):
        return [r.y for r in self.robots]


def evalparticles(particles, measure):
    for part in particles:
        for robot, meas in zip(part.robots, measure):
            estimations = []
            for obj in part.robots + MARKERS:
                estimations.append(visibility(robot, obj))
            robot.weight = robot_similarity(estimations, meas)
        part.weight = prod([r.weight for r in part.robots])
