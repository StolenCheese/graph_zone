import itertools
from graph import *
from triangles import *

if __name__ == "__main__":

    @OnGraphChanged
    def QuickHull(graph: "Graph"):

        max_x = max(v.pos.x for v in graph.vertices.values())
        max_y = max(v.pos.y for v in graph.vertices.values())
        min_x = min(v.pos.x for v in graph.vertices.values())
        min_y = min(v.pos.y for v in graph.vertices.values())

        d = max(max_y - min_y, max_x - min_x)

        points = {
            tl: Vertex(min_x - d * 2, min_y - d),
            bm: Vertex((min_x + max_x) * 0.5, max_y + d * 2),
            tr: Vertex(max_x + d * 2, min_y - d)} | graph.vertices

        graph.triangulation = Mesh.FromTriangle(tl, bm, tr)

        v = None
        for k in graph.vertices.keys():
            if graph.Step(f"Adding {k}"):
                break

            v = graph.triangulation.Insert(graph.triangulation.FindTri(
                lambda t: TriangleContains(
                    *[points[k].pos for k in [t.v0.v, t.v1.v, t.v2.v, k]]
                )), k
            )
        graph.focused_tris = []
        t0, t1 = itertools.islice(graph.triangulation.TrianglesOfVertexCCW(v), 2)
        if not graph.Step(f"Flipping {repr(t0)} and {repr(t1)}"):

            graph.triangulation.FlipDelaunay(t0, t1)
        else:

            for t in graph.triangulation.TrianglesOfVertexCCW(v):
                # print(t)
                graph.focused_tris.append(t.verts)

    Graph().JitterPoints(3, 2).Run()
