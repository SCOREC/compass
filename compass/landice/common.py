"""Contains utilities common to all meshing methods"""

from compass.landice.settings import ADAPTIVE, CELL_SIZE


class Edge:
    def __init__(self, v1, v2):
        self.v1 = v1
        self.v2 = v2

    def swap(self, swap=True):
        if swap:
            return Edge(self.v2, self.v1)
        else:
            return Edge(self.v1, self.v2)


class Adapt:
    def __init__(self, isAdaptive=ADAPTIVE, cellSize=CELL_SIZE):
        self.isAdaptive = isAdaptive
        self.cellSize = cellSize

    def adapt(self, v0, v1):
        """v0 and v1 are numbers of opposite sign. This returns how
           far you need to interpolate from v0 to v1 to get to 0."""
        assert (v1 > 0) != (v0 > 0), "v0 and v1 do not have opposite sign"
        if self.isAdaptive:
            return (0 - v0) / (v1 - v0) * self.cellSize
        else:
            return 0.5 * self.cellSize


def frange(start, stop, step=1):
    """Like range, but works for floats"""
    v = start
    while v < stop:
        yield v
        v += step
