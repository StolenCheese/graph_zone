
from re import sub
from graph import *


if __name__ == "__main__":

    @OnGraphChanged
    def QuickHull(graph: "Graph"):

        if graph.Step("Finding First line"):
            return

        start = max(graph.vertices.keys(), key=graph.X)
        end = min(graph.vertices.keys(), key=graph.X)

        graph.edges = {start: [(end, (255, 255, 255))]}

        hull = {start, end}

        # generate subset
        if graph.Step("Generate Subsets"):
            return

        subset = set(graph.vertices.keys()).difference(hull)

        graph.groups = [(frozenset(subset), (255, 0, 255)), (hull, (0, 0, 255))]

        # split over line

    Graph().JitterPoints(3, 2).Run()
