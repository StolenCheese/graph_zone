
from collections import deque
from dataclasses import dataclass
import enum
from typing import Callable
import numpy as np
from pygame import Vector2

VALIDATE = False


class InvalidEdgeException(Exception):
    ...


def TriangleContains(a: "Vector2", b: "Vector2", c: "Vector2", pos: "Vector2") -> "bool":
    # one of the ones from https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle

    v0 = b - a
    v1 = c - a
    v2 = pos - a

    d00 = v0.dot(v0)
    d01 = v0.dot(v1)
    d11 = v1.dot(v1)
    d20 = v2.dot(v0)
    d21 = v2.dot(v1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w

    if 0 <= v <= 1 and 0 <= w <= 1 and 0 <= u <= 1:
        # valid barycentric to be contained within the triangle
        return True
    return False


def T(v: "Vector2") -> "Vector2":
    """
    Transpose a Vector2: `(x,y) -> (-y,x)`
    """
    return Vector2(-v.y, v.x)


def CalculateCircumcircle(a: "Vector2", b: "Vector2", c: "Vector2") -> "tuple[Vector2,float]":
    # calculate the circumcircle for the first 3 points,
    mid0 = (a + b) * 0.5
    mid1 = (a + c)*0.5
    dir0 = T(a - b)
    dir1 = T(a - c)

    # Find the point at which these lines intersect relative to traversal along line 1
    t = (mid1.y * dir0.x - mid1.x * dir0.y + mid0.x*dir0.y - mid0.y*dir0.x) / (dir1.x * dir0.y - dir1.y * dir0.x)

    # point at which lines intersect is the center of the triangle
    center = mid1 + t * dir1
    rad = center.distance_squared_to(a)

    if VALIDATE:
        # every point should be (roughly) equidistant to this
        assert rad - center.distance_squared_to(b) < 0.01 and rad-center.distance_squared_to(c) < 0.01

    return center, rad


@dataclass(slots=True)
class nbr:
    """
    Structure to index neighbours' edges adjacent to triangle `t`
    """
    t: "Tri"

    def __getitem__(self, key) -> "Edge":
        return (self.t.e0, self.t.e1, self.t.e2)[key]

    def __iter__(self):
        return iter((self.t.e0, self.t.e1, self.t.e2))

    def __setitem__(self, key: "int", item: "Edge"):
        assert isinstance(item, Edge), "invalid assignment"

        match key:
            case 0:
                self.t.e0 = item
            case 1:
                self.t.e1 = item
            case 2:
                self.t.e2 = item
            case _:
                raise IndexError(f"Index {key} out of bounds")


@dataclass(slots=True)
class verts:
    """
    Index vertexes of triangle `t`
    """
    t: "Tri"

    def __iter__(self):
        return iter((self.t.v0, self.t.v1, self.t.v2))

    def __getitem__(self, key: "int") -> "Vert":
        return (self.t.v0, self.t.v1, self.t.v2)[key]

    def __setitem__(self, key: "int", item: "Vert") -> "None":
        assert isinstance(item, Vert)

        match key:
            case 0:
                self.t.v0 = item
            case 1:
                self.t.v1 = item
            case 2:
                self.t.v2 = item
            case _:
                raise IndexError(f"Index {key} out of bounds")


@dataclass(repr=False, slots=True)
class Tri:
    """
    Triangle within triangle edge neighbour mesh
    ### `v[0..=2]`
    Vertexes of the triangle. Assumed to be immutable for `__hash__`.

    ### `e[0..=2]`
    Adjacent edges to the triangle.

    An edge ek is the edge originating from vertex vk, with counterclockwise 
    """
    e0: "Edge"
    e1: "Edge"
    e2: "Edge"
    v0: "Vert"
    v1: "Vert"
    v2: "Vert"

    @property
    def nbr(self) -> "nbr":
        return nbr(self)

    @property
    def v(self) -> "verts":
        return verts(self)

    @property
    def verts(self) -> "frozenset[str]":
        return frozenset({self.v0.v, self.v1.v, self.v2.v})

    def __eq__(self, __o: object) -> bool:
        """
        Triangles are the same if they share vertexes,
        Edges not importantant in most operations
        """
        if isinstance(__o, Tri):
            return self.v0 == __o.v0 and self.v1 == __o.v1 and self.v2 == __o.v2
        return False

    def OtherPoint(self, a: "Vert", b: "Vert"):
        """
        Get the vertex that is not a or b
        """
        x, = {self.v0, self.v1, self.v2}.difference({a, b})
        return x

    def GetNeighourOnEdge(self, a, b):
        index = GetSharedEdgeIndexUnordered(a, b, self)

        neighbour = self.nbr[index].t

        return neighbour

    def __contains__(self, __o):
        """
        Vert v in t?
        """
        match __o:
            case Vert(x, _):
                return x in self.verts
            case str(s):
                return s in self.verts
            case _:
                return False

    # normal dataclass structures based only of vertexes,
    # as the adjacent edges of a triangle may change in normal operations we cannot use them for the hash.

    def __hash__(self) -> int:
        return hash((self.v0, self.v1, self.v2))

    def __repr__(self) -> str:
        return f"{self.v0}-{self.v1}-{self.v2}"

    def __format__(self, __format_spec: str) -> str:
        return self.__repr__()

    def __str__(self) -> str:
        return self.__repr__()


@dataclass
class Edge:
    """
    i-th edge of triangle t
    """

    t: "Tri"
    i: "int"  # 0,1,2

    def __iter__(self):
        return iter((self.t, self.i))


@dataclass(repr=False)
class Vert:
    """
    Vertexes are reprosented by a label, which can be translated into a postition,
    but also used in other algorithms for convince.

    As with triangles, edges can change but should not require a rehash.
    """
    v: "str"
    e: "Edge"  # any edge leaving this vertex

    def __iter__(self):
        return iter((self.v, self.e))

    def __hash__(self) -> int:
        return hash(self.v)

    def __repr__(self) -> str:
        return self.v


def GetEdgeIndex(i: "Vert", j: "Vert", tri: "Tri"):
    """
    Get the index of the edge i->j on triangle `tri`
    """
    if tri.v0 == i and tri.v1 == j:
        return 0
    elif tri.v1 == i and tri.v2 == j:
        return 1
    elif tri.v2 == i and tri.v0 == j:
        return 2
    else:
        raise InvalidEdgeException(f"Edge {i} {j} does not appear on triangle {tri}")


def GetSharedEdgeIndexUnordered(shared0, shared1, other: "Tri"):
    """
    Find the index of any edge shared0 <-> shared1 in other.
    """
    # points will be in opposite order to each other
    try:
        return GetEdgeIndex(shared0, shared1, other)
    except InvalidEdgeException:
        return GetEdgeIndex(shared1, shared0, other)


def NeedsFlipDelaunay(p: "Callable[[str],Vector2]", i: "str", j: "str", k: "str", l: "str", boundary: "set[str]"):
    #    k
    #  /   \
    # i <-> k
    #  \   /
    #    l

    intersection = {i, j, k, l}.intersection(boundary)
    match list(intersection):
        # case [x] if x == i or x == j:
        #     # switch edge to make up part of hull,
        #     # if the point in intersection x is not also part of the hull

        #     # tell between

        #     #         / ---- k
        #     #    /----     /
        #     #   i ------- j
        #     #    \----     \
        #     #         \---  l

        #     # and

        #     #      k
        #     #    /  \
        #     #   i----j
        #     #    \  /
        #     #      l

        #     # as we should never flip a concave hull

        #     message[0] = "Switching to maintain hull"
        #     return True
        case [x] if x == k or x == l:
            # already in hull position, dont move it
            # message[0] = "Not switching to maintain hull"
            return False

        case _:
            # message[0] = f"Switching for circumcircle, {len(intersection)}"
            # calculate the circumcircle for the first 3 points,
            center, rad = CalculateCircumcircle(p(i), p(j), p(k))

            return center.distance_squared_to(p(l)) < rad


@dataclass
class Mesh:
    tris: "set[Tri]"
    verts: "dict[str,Vert]"

    def FindTri(self, pred: "Callable[[Tri],bool]") -> "Tri":
        for t in self.tris:
            if pred(t):
                return t

    def VerifyTriangle(self, t: "Tri"):
        for n in t.nbr:
            if n.t != None:
                assert n.t in self.tris, f"Neighbour of {t} is {n.t} which does not exist"
                assert len(n.t.verts.intersection(t.verts)) == 2, f"Neighbour of {t} does not share 2 verts"
                assert n.t.nbr[n.i].t == t, f"{n} Neighbour relation {n.t.nbr[n.i]} non-returning"

    def CheckValid(self):
        """
        Assert that the triangulation is valid
        """
        if not VALIDATE:
            return

        for v in self.verts.values():
            assert v.e.t in self.tris, f"{v.e.t} not in graph"
            assert v == v.e.t.v[v.e.i], f"{v} not at correct pos in {v.e.t}"

        for t in self.tris:
            self.VerifyTriangle(t)

        # for t in self.tris:
        #     for j in range(3):
        #         assert t.nbr[j].t.nbr[t.nbr[j].i].t == t

    def TrianglesOfVertexCCW(self, v: "Vert"):
        """Iterate through every triangle of vertex v, counterclockwise"""
        if v != None:
            t, i = v.e
            f = True
            while t != None and (f or t.verts != v.e.t.verts):

                if VALIDATE:
                    assert v in t, "Gotten off track"

                yield t
                t, i = t.nbr[i]
                i = (i + 1) % 3
                f = False

    def TrianglesOfVertexCW(self, v: "Vert"):
        """Iterate through every triangle of vertex v, clockwise"""
        if v != None:
            t, i = v.e
            f = True
            while t != None and (f or t.verts != v.e.t.verts):

                if VALIDATE:
                    assert v in t, "Gotten off track"

                yield t
                t, i = t.nbr[i]
                i = (i + 1) % 3
                f = False

    @classmethod
    def FromTriangle(cls, v0, v1, v2):
        """
        Create a triangulation from a single triangle
        """
        vert0 = Vert(v0, None)
        vert1 = Vert(v1, None)
        vert2 = Vert(v2, None)

        t = Tri(Edge(None, -1), Edge(None, -1), Edge(None, -1), vert0, vert1, vert2)
        # vert 0 is 0th edge (v0 -> ), etc
        vert0.e = Edge(t, 0)
        vert1.e = Edge(t, 1)
        vert2.e = Edge(t, 2)

        return Mesh({t}, {v0: vert0, v1: vert1, v2: vert2})

    def FindTrianglesOnEdge(self, a: "Vert", b: "Vert"):
        # first look around counterclockwise
        for tri in self.TrianglesOfVertexCCW(a):
            if b in tri:
                # we have one side of the edge, just get the other
                return (tri, tri.GetNeighourOnEdge(a, b))

        # then look clockwise in case the hull stopped our iteration
        for tri in self.TrianglesOfVertexCW(a):
            if b in tri:
                return (tri, tri.GetNeighourOnEdge(a, b))

        for tri in self.tris:
            if b in tri and a in tri:
                return (tri, tri.GetNeighourOnEdge(a, b))

    def Insert(self, t: "Tri", v: "str") -> "Vert":
        vert = Vert(v, None)
        self.tris.remove(t)
        self.verts[v] = vert

        # counterclockwise
        v0, v1, v2 = t.v

        new_ts = [
            Tri(None, None, None, p0, p1, p2)
            for p0, p1, p2 in
            [(v0, v1, vert), (v1, v2, vert), (v2, v0, vert)]
        ]
        # update possibly now bad v[k] edges

        v0.e = Edge(new_ts[0], 0)
        v1.e = Edge(new_ts[1], 0)
        v2.e = Edge(new_ts[2], 0)

        for i, outer_nbr in enumerate(t.nbr):
            left_i = (i+1) % 3
            right_i = (i-1) % 3

            # get original triangle's edge in this direction

            # edge k in {0,1,2} shares vertexes k and k+1
            # so find the correct offset for the outer_e

            # add more edges
            left_nbr = Edge(new_ts[left_i], GetEdgeIndex(new_ts[i].v2, new_ts[i].v1, new_ts[left_i]))
            right_nbr = Edge(new_ts[right_i], GetEdgeIndex(new_ts[i].v0, new_ts[i].v2, new_ts[right_i]))

            new_ts[i].nbr[0] = outer_nbr  # shares p0 and p1
            new_ts[i].nbr[1] = left_nbr  # shares p1 and p2
            new_ts[i].nbr[2] = right_nbr  # shares p2 and p0

            self.tris.add(new_ts[i])

        vert.e = new_ts[0].e1

        # update the old neighbours to now point to our 3 created triangles

        # new triangles and neighbours are in the same order (i hope)
        for new_t, outer_nbr in zip(new_ts, t.nbr):
            if outer_nbr.t != None:
                # connecting edge is edge 0 on the new triangles
                assert outer_nbr.t.nbr[outer_nbr.i].t == t, f"This edge: {outer_nbr}, other edge: {outer_nbr.t.nbr[outer_nbr.i]}, this: {t}"

                outer_nbr.t.nbr[outer_nbr.i] = Edge(new_t, 0)

        self.CheckValid()

        return vert

    def FlipDelaunay(self, t0: "Tri", t1: "Tri"):
        """
        Flip the edge connecting these two triangles to connect the two points they do not share.

        Then fix all edges on neighbours invalidated by the change in triangles
        """
        # get edge connecting them
        i, j = set(t0.v).intersection(t1.v)
        k, = set(t0.v).difference(t1.v)
        l, = set(t1.v).difference(t0.v)

        try:
            GetEdgeIndex(i, j, t1)
        except Exception:
            # edges must be wrong order

            i, j = j, i

        ij_edge_t1 = GetEdgeIndex(i, j, t1)
        ji_edge_t0 = GetEdgeIndex(j, i, t0)

        # now garmented to have
        #          i
        #       /  |  \
        #     /    |    \
        #   k  t0  | t1  l
        #     \    |    /
        #       \  |  /
        #          j

        assert i != j != k != l

        self.tris.remove(t0)
        self.tris.remove(t1)

        # need to get the 4 triangles that neighbour this one to set neighbours

        new_t0 = Tri(None, None, None, k, l, i)
        new_t1 = Tri(None, None, None, k, j, l)

        self.tris.add(new_t0)
        self.tris.add(new_t1)

        #          i
        #       /    \
        #     / new_t0 \
        #   k  --------- l
        #     \ new_t1 /
        #       \    /
        #          j

        # update new triangle neighbour's
        new_t0.nbr[0] = Edge(new_t1, 2)  # k-l (0) maps to l-k (2)
        new_t0.nbr[1] = t1.nbr[GetEdgeIndex(new_t0.v1, new_t0.v2, t1)]  # same as old l-i from t1
        new_t0.nbr[2] = t0.nbr[GetEdgeIndex(new_t0.v2, new_t0.v0, t0)]  # same as old i-k from t0

        new_t1.nbr[2] = Edge(new_t0, 0)  # l-k (2) maps to k-l (0)
        new_t1.nbr[0] = t0.nbr[GetEdgeIndex(new_t1.v0, new_t1.v1, t0)]  # k-j, in t0
        new_t1.nbr[1] = t1.nbr[GetEdgeIndex(new_t1.v1, new_t1.v2, t1)]  # j-l, in t1

        # update old triangle neighbour's neighbour to this
        ik_edge = (ji_edge_t0 + 1) % 3
        kj_edge = (ji_edge_t0 + 2) % 3

        jl_edge = (ij_edge_t1 + 1) % 3
        li_edge = (ij_edge_t1 + 2) % 3

        ik_nbr = t0.nbr[ik_edge]  # i-k edge
        kj_nbr = t0.nbr[kj_edge]  # k-j edge

        jl_nbr = t1.nbr[jl_edge]  # j-l edge
        li_nbr = t1.nbr[li_edge]  # l-i edge

        # update the triangle we point to, and the edge we connect to
        if ik_nbr.t != None:
            ik_nbr.t.nbr[ik_nbr.i] = Edge(new_t0, GetEdgeIndex(t0.v[ik_edge], t0.v[kj_edge],  new_t0))

            self.VerifyTriangle(ik_nbr.t)

        if kj_nbr.t != None:
            kj_nbr.t.nbr[kj_nbr.i] = Edge(new_t1, GetEdgeIndex(t0.v[kj_edge], t0.v[ji_edge_t0],  new_t1))

            self.VerifyTriangle(kj_nbr.t)

        if jl_nbr.t != None:
            jl_nbr.t.nbr[jl_nbr.i] = Edge(new_t1, GetEdgeIndex(t1.v[jl_edge], t1.v[li_edge],  new_t1))

            self.VerifyTriangle(jl_nbr.t)

        if li_nbr.t != None:
            li_nbr.t.nbr[li_nbr.i] = Edge(new_t0, GetEdgeIndex(t1.v[li_edge], t1.v[ij_edge_t1],  new_t0))

            self.VerifyTriangle(li_nbr.t)

        i.e = Edge(new_t0, GetEdgeIndex(i, k, new_t0))
        # is be +1 of last one because counterclockwise order
        k.e = Edge(new_t0, (i.e.i + 1) % 3)

        j.e = Edge(new_t1, GetEdgeIndex(j, l, new_t1))
        l.e = Edge(new_t1, (j.e.i + 1) % 3)

        # do check on the created triangle's neighbours

        self.VerifyTriangle(new_t0)
        self.VerifyTriangle(new_t1)

        self.CheckValid()

    def InsertPointDelaunay(self, p: "Callable[[str],Vector2]", label: "str", boundary: "set[str]"):
        """
        Add this vert to the triangulation:
        - Split the triangle containing it (assumed to exist) into 3
        - flip all edges that invalidate the triangulation
        """

        # if self.Step(f"Adding {label}"):
        #     return

        # 1. find the triangle that contains this
        # TODO: histroy graph, currently this is O(n)
        tri = self.FindTri(lambda tri: TriangleContains(*[p(v) for v in tri.verts], p(label)))

        vert = self.Insert(tri, label)

        # perform flips on edges incident to vert - edges originting from the inserted vertex are known to be valid
        good_edges = {frozenset({vert, tri.v0}), frozenset({vert, tri.v1}), frozenset({vert, tri.v2})}

        bad_edges = deque([frozenset({tri.v0, tri.v1}), frozenset({tri.v1, tri.v2}), frozenset({tri.v2, tri.v0})])

        # find triangle on flipside of tri[0] <-> tri[1]
        # if this doesn't exist, it then is not important; hull of graph

        # DONE: better graph reprosentation for this

        while len(bad_edges) > 0:

            j, i = bad_edges.pop()
            # http://web.mit.edu/alexmv/Public/6.850-lectures/lecture09.pdf

            # if self.Step(f"Looking at {j} - {i}"):
            #     return

            # case 1
            if j.v in boundary and i.v in boundary:
                continue

            found = self.FindTrianglesOnEdge(i, j)

            assert len(found) == 2, f"Cannot find pair for {i}-{j}; {found}"

            t0, t1 = found
            k = t0.OtherPoint(i, j)
            l = t1.OtherPoint(i, j)

            self.focused_tris = [t0.verts, t1.verts]

            # we never need to flip triangles on boundary

            if NeedsFlipDelaunay(p, i.v, j.v, k.v, l.v, boundary):
                # if self.Step(m[0]):
                #     return

                # flip!
                self.FlipDelaunay(t0, t1)

                # created edge is good
                good_edges.add(frozenset({l, k}))
                # continue in other edges
                for i in [{k, j}, {k, i}, {l, j}, {l, i}]:
                    i = frozenset(i)
                    if i not in good_edges:
                        bad_edges.append(i)
