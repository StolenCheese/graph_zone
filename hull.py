
from re import sub
from graph import *


if __name__ == "__main__":

    @OnGraphChanged
    def QuickHull(graph: "Graph"):
        # reset graph
        graph.groups = []
        graph.edges = {}

        if graph.Step("Finding First line"):
            return

        start = max(graph.vertices.keys(), key=graph.X)
        end = min(graph.vertices.keys(), key=graph.X)

        graph.edges = {start: [(end, (255, 255, 255))]}

        hull = {start, end}

        subset = set(graph.vertices.keys()).difference(hull)

        graph.groups = [subset, hull]

        # generate subsets
        if graph.Step("Split over line"):
            return

        split0, split1 = SplitAlongLine(graph, graph.Pos(end), graph.Pos(start), subset)

        graph.groups = [split0, split1, hull]

        hull_poly = [start]

        hull_poly += EliminateTriangle(graph, hull, split0, start,  end,  [0])

        hull_poly.append(end)

        hull_poly += EliminateTriangle(graph, hull, split1,  end, start, [1])

        # generate subsets
        if graph.Step("Showing edges"):
            return

        graph.edges = {}
        for s, e in zip(hull_poly, hull_poly[1::] + [hull_poly[0]]):
            graph.edges[s] = [(e, (255, 255, 255))]

    def SplitAlongLine(graph: "Graph", line0: "Vector2", line1: "Vector2", subset: "set[str]"):

        line_normal = T(line1 - line0)

        split0 = set()
        split1 = set()

        for vert in subset:

            vert_direction = graph.Pos(vert) - line0

            # Determin what side of the line we are on
            split = line_normal.dot(vert_direction)

            if split >= 0:
                split0.add(vert)
            else:
                split1.add(vert)

        return split0, split1

    def EliminateTriangle(graph: "Graph", hull: "set[str]", subset: "set[str]", line_start: "str", line_end: "str", label):
        if not subset:
            return

        # Find the most extreme point in the subset, then eliminate all points inside the triangle

        if graph.Step(f"Find extreme point in subset {label}"):
            return

        max_dist = 0
        extreme = None

        line0 = graph.Pos(line_start)
        line1 = graph.Pos(line_end)

        line_normal = T(line1 - line0).normalize()

        for vert in subset:
            # calculate distance from this line - project onto the normal of the line
            dist = abs(line_normal.dot(graph.vertices[vert].pos - line0))

            if dist > max_dist:

                max_dist = dist
                extreme = vert

        assert extreme != None

        # connect the extreme point to the line to show this triangle
        graph.edges[extreme] = [(line_start, (255, 255, 255)), (line_end, (255, 255, 255))]

        if graph.Step(f"Remove points in triangle {line_start} {line_end} {extreme}"):
            return

        graph.groups.remove(subset)

        # move the extreme point to the hull
        subset.remove(extreme)
        hull.add(extreme)

        if not subset:
            # this could have been the only point
            yield extreme
            return

        # record points not in triangle

        remaining = set()

        for vert in subset:
            if not TriangleContains(graph.Pos(line_start), graph.Pos(line_end), graph.Pos(extreme), graph.Pos(vert)):
                remaining.add(vert)

        graph.groups.append(remaining)

        if graph.Step(f"Split over line ({line_start} {line_end}) to {extreme}"):
            return

        # split on which side of the triangle we are on and recurse
        split0, split1 = SplitAlongLine(graph, graph.Pos(extreme), (line1 + line0) * 0.5,   remaining)

        graph.groups.remove(remaining)
        graph.groups.append(split0)
        graph.groups.append(split1)
        # recourse, giving the closer edge of the triangle to each set for a larger triangle
        for h in EliminateTriangle(graph, hull, split0, line_start, extreme, label + [0]):
            yield h

        yield extreme

        for h in EliminateTriangle(graph, hull, split1, line_end, extreme, label + [1]):
            yield h

    Graph().JitterPoints(4, 4).Run()
