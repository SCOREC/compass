"""Contains utilities common to 2d meshing methods"""
# based on https://github.com/BorisTheBrave/mc-dc
# master@a165b326849d8814fb03c963ad33a9faf6cc6dea

import math

from compass.landice.common import frange
from compass.landice.settings import CELL_SIZE, XMAX, XMIN, YMAX, YMIN


class V2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def normalize(self):
        d = math.sqrt(self.x * self.x + self.y * self.y)
        return V2(self.x / d, self.y / d)


def element(e, **kwargs):
    """Utility function used for rendering svg"""
    s = "<" + e
    for key, value in kwargs.items():
        s += " {}='{}'".format(key, value)
    s += "/>\n"
    return s


def make_svg(file, edges, f, xmin=XMIN, xmax=XMAX, ymin=YMIN, ymax=YMAX,
             cellSize=CELL_SIZE):
    """Writes an svg file showing the given edges and solidity"""
    scale = 50
    file.write("<?xml version='1.0' encoding='UTF-8'?>\n")
    url = "http://www.w3.org/2000/svg"
    file.write("<svg version='1.1' xmlns='{}' viewBox='{} {} {} {}'>\n"
               .format(url, xmin * scale,
                       ymin * scale,
                       (xmax - xmin) * scale,
                       (ymax - ymin) * scale))

    file.write("<g transform='scale({})'>\n".format(scale))
    # Draw grid
    styleSpec = "stroke: grey; stroke-width: 0.02; fill: none"
    for x in frange(xmin, xmax, cellSize):
        for y in frange(ymin, ymax, cellSize):
            file.write(element("rect", x=x, y=y, width=cellSize,
                               height=cellSize,
                               style=styleSpec))

    # Draw filled / unfilled circles
    styleSpec = "stroke: black; stroke-width: 0.02; fill: "
    for x in frange(xmin, xmax + cellSize, cellSize):
        for y in frange(ymin, ymax + cellSize, cellSize):
            is_solid = f(x, y) > 0
            fill_color = ("black" if is_solid else "white")
            file.write(element("circle", cx=x, cy=y, r=0.05,
                               style=styleSpec + fill_color))
    # Draw edges
    for edge in edges:
        file.write(element("line", x1=edge.v1.x, y1=edge.v1.y,
                           x2=edge.v2.x, y2=edge.v2.y,
                           style='stroke:rgb(255,0,0);stroke-width:0.04'))

    # Draw vertices
    r = 0.05
    for v in [v for edge in edges for v in (edge.v1, edge.v2)]:
        file.write(element("rect", x=(v.x - r), y=(v.y - r),
                           width=2 * r, height=2 * r,
                           style='fill: red'))

    file.write("</g>\n")
    file.write("</svg>\n")
