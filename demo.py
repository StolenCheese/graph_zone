
import random
from graph import Graph, OnGraphChanged


if __name__ == "__main__":

    # Click to drag nodes

    # Scroll to progress algorithm (in this case, voronoi)

    @OnGraphChanged
    def ShowVoronoi(graph: "Graph"):
        # Grouping displayed by colouring a voronoi diagram;
        # give everything a random colour in their own group to see it.
        graph.groups = []

        random.seed(42)

        for v in sorted(graph.vertices.keys()):

            if graph.Step(f"Adding {v}"):
                break

            graph.groups.append([v])

    Graph().JitterPoints(3, 3).Run()
