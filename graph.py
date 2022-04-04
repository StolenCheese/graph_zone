
import pstats
import abc
import cProfile
from collections import deque
from dataclasses import dataclass
import math
from multiprocessing import Event
import random
from typing import Any, Callable, Iterable
import pygame
from pygame.math import Vector2

from triangles import *
pygame.font.init()  # you have to call this at the start,
# if you want to use this module.
font = pygame.font.SysFont('Comic Sans MS', 30)

PROFILE_DELAUNAY = False


class Drawable(abc.ABC):
    @abc.abstractmethod
    def Draw(self, surface: "pygame.Surface", id: "str"):
        ...


def f8(x):

    return "%6dµs" % (x * 1000000)


pstats.f8 = f8


@dataclass
class Vertex(Drawable):
    x: float
    y: float

    @property
    def draw_pos(self) -> "tuple[int,int]":
        return (int(self.x), int(self.y))

    @property
    def pos(self) -> "Vector2":
        return Vector2(self.x, self.y)

    def Translate(self, rel):
        self.x += rel[0]
        self.y += rel[1]

    def Draw(self, surface: "pygame.Surface", id: "str"):
        pygame.draw.circle(surface, (255, 255, 255), self.draw_pos, 20)
        textsurface = font.render(id, False, (0, 0, 0))
        r = textsurface.get_rect()
        r.center = self.draw_pos
        surface.blit(textsurface, r)


tl = "tl"
tr = "tr"
bm = "bm"
boundary = frozenset({tl, tr, bm})


class Graph(Drawable):

    def __init__(self, draw_delaunay=False):
        self.vertices: "dict[str,Vertex]" = {}
        self.edges: "dict[str,list[tuple[str, tuple[int,int,int]]]]" = {}
        self.triangulation: "Mesh" = None
        self.focused_tris = []
        self.status = "Starting"

        self.draw_delaunay = draw_delaunay

        self.groups: "list[Iterable[str]]" = []
        self.mouse = (0, 0)
        self.w = 512
        self.h = 512

        self.step = 0
        self.maxstep = 1

        self.selected_node: "str" = None

    def RandomPoints(self: "Graph", n: "int") -> "Graph":
        """
        Add `n` random points to the graph
        """
        v = {chr(i + 57): Vertex(random.random() * 512, random.random() * 512) for i in range(n)}

        self.vertices |= v
        return self

    def X(self, p: "str") -> "float":
        """
        Get x coordinate of vertex labelled `p`
        """
        return self.vertices[p].x

    def Pos(self, p: "str") -> "Vector2":
        """
        Get coordinate of vertex `p`
        """
        return self.vertices[p].pos

    def JitterPoints(self: "Graph", w: "int", h: "int") -> "Graph":
        """
        Add `w * h` Jittered points to the graph

        Surface is divided into a `w` by `h` grid, and a single node is added to every tile
        """
        gap_x = self.w / w
        gap_y = self.h / h
        for x in range(w):
            for y in range(h):

                l = chr(x + y * w + 97)
                self.vertices[l] = Vertex((x + random.random()) * gap_x, (y + random.random()) * gap_y)
        return self

    def RandomConnections(self: "Graph", p: "float") -> "Graph":
        """
        Randomly connect any two vertexes in the graph with probability `p`
        """
        e = {l: [(l, (255, 255, 255)) for l in self.vertices.keys() if random.random() < p] for l in self.vertices.keys()}

        self.edges |= e

        return self

    def FindVertexAt(self, pos) -> "str":
        for l, vert in self.vertices.items():
            rx = pos[0] - vert.x
            ry = pos[1] - vert.y
            if rx * rx + ry * ry < 20*20:
                return l
        return None

    def Step(self, message):
        """
        Perform a step in some algorithm
        - returns: `True` if the algorithm should stop
        """
        self.step += 1
        if self.step >= self.maxstep:
            return True
        else:
            self.status = message
            return False

    def Run(self) -> "None":
        pygame.init()
        surf = pygame.display.set_mode(size=(512, 512), flags=pygame.RESIZABLE)
        running = True
        dirty = False
        clock = pygame.time.Clock()

        while running:
            dt = clock.tick(144)
            for event in pygame.event.get():

                # Python has switch statements now, pog
                match event.type:
                    case pygame.QUIT:
                        running = False

                    case pygame.MOUSEMOTION:
                        # pos, rel, buttons, touch
                        if self.selected_node != None:
                            self.vertices[self.selected_node].Translate(event.rel)
                            dirty = True
                        self.mouse = event.pos

                    case pygame.MOUSEBUTTONUP:
                        # pos, button, touch
                        self.selected_node = None

                        self.mouse = event.pos
                    case pygame.MOUSEBUTTONDOWN:
                        # pos, button, touch
                        match event.button:
                            case 1:
                                self.selected_node = self.FindVertexAt(event.pos)

                                self.mouse = event.pos
                            case 4:
                                self.maxstep += 1
                            case 5:
                                self.maxstep -= 1

                        self.maxstep = max(self.maxstep, 0)

                        dirty = True

                    case pygame.VIDEORESIZE:
                        # size, w, h
                        ...

                if dirty:
                    # regenerate the sections, so do voronoi diagram
                    if PROFILE_DELAUNAY:
                        with cProfile.Profile(timeunit=1000) as pr:
                            self.Delaunay()
                        pr.print_stats('cumtime')
                    else:
                        self.Delaunay()

                    self.step = 0
                    [h(self) for h in handlers]
                    dirty = False

                # s = self.status
                # self.status += f" fps: {clock.get_fps()}"

                self.Draw(surf)
                # self.status = s

    def Delaunay(self) -> "None":

        max_x = max(v.pos.x for v in self.vertices.values())
        max_y = max(v.pos.y for v in self.vertices.values())
        min_x = min(v.pos.x for v in self.vertices.values())
        min_y = min(v.pos.y for v in self.vertices.values())

        d = max(max_y - min_y, max_x - min_x)

        points = {
            tl: Vertex(min_x - d * 3, min_y - d),
            bm: Vertex((min_x + max_x) * 0.5, max_y + d * 3),
            tr: Vertex(max_x + d * 3, min_y - d)} | self.vertices

        self.triangulation = Mesh.FromTriangle(tl, bm, tr)

        for label in self.vertices.keys():

            self.triangulation.InsertPointDelaunay(lambda p: points[p].pos, label, boundary)

          # [t for t in triangles if t.isdisjoint({tl, tr, bm})]

    def Draw(self: "Graph", surf: "pygame.Surface"):
        """
        Draw structure of the graph, depending on flags
        - `draw_delaunay` : Draw the created triangulation
        """
        surf.fill((0, 0, 0))

        s = random.getstate()
        random.seed(2)

        max_x = max(v.pos.x for v in self.vertices.values())
        max_y = max(v.pos.y for v in self.vertices.values())
        min_x = min(v.pos.x for v in self.vertices.values())
        min_y = min(v.pos.y for v in self.vertices.values())

        d = max(max_y - min_y, max_x - min_x)

        points = {
            tl: Vertex(min_x - d * 3, min_y - d),
            bm: Vertex((min_x + max_x) * 0.5, max_y + d * 3),
            tr: Vertex(max_x + d * 3, min_y - d)}

        v = self.vertices | points

        # generate the same random colours every frame
        random.seed(67)

        for verts in self.groups:
            # draw dual graph of the vertexes to make a voronoi
            #   The polygon for a node is the center of circumcircles for the surrounding nodes
            col = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            for vert in verts:
                if vert in self.triangulation.verts:
                    poly = [
                        CalculateCircumcircle(*[v[vs].pos for vs in tri.verts])[0]
                        for tri in self.triangulation.TrianglesOfVertexCCW(self.triangulation.verts[vert])
                    ]
                    pygame.draw.polygon(surf, col, poly)

        if self.triangulation != None and self.draw_delaunay:
            for tri in self.triangulation.tris:
                a, b, c = tri.verts

                col = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                pa, pb, pc = [v[i].pos for i in [a, b, c]]
                if TriangleContains(pa, pb, pc, Vector2(self.mouse)):
                    # cursor inside this triangle

                    center, rad = CalculateCircumcircle(pa, pb, pc)
                    pygame.draw.circle(surf, (255, 0, 255), center, math.sqrt(rad), 2)
                    # pygame.draw.polygon(
                    #     surf,
                    #     col,
                    #     [v[i].draw_pos for i in [a, b, c]])

                pygame.draw.line(surf, (0, 255, 0), v[a].draw_pos, v[b].draw_pos, 3)
                pygame.draw.line(surf, (0, 255, 0), v[b].draw_pos, v[c].draw_pos, 3)
                pygame.draw.line(surf, (0, 255, 0), v[c].draw_pos, v[a].draw_pos, 3)

            for vert in self.triangulation.verts.values():
                pygame.draw.line(surf, (255, 0, 0), v[vert.v].draw_pos, CalculateCircumcircle(*[v[vs].pos for vs in vert.e.t.verts], )[0])

        for (a, b, c) in self.focused_tris:
            pygame.draw.line(surf, (255, 255, 0), v[a].draw_pos, v[b].draw_pos, 3)
            pygame.draw.line(surf, (255, 255, 0), v[b].draw_pos, v[c].draw_pos, 3)
            pygame.draw.line(surf, (255, 255, 0), v[c].draw_pos, v[a].draw_pos, 3)

        # pygame.draw.polygon(
        #     surf,
        #     col,
        #     [v[i].draw_pos for i in [a, b, c]])

        random.setstate(s)

        for (s, e, c) in ((self.vertices[s].draw_pos,  self.vertices[e].draw_pos, c) for s, es in self.edges.items() for e, c in es):
            pygame.draw.line(surf, c, s, e, 3)

        for label, vert in self.vertices.items():
            # draw all vertexes and groups
            vert.Draw(surf, label)

        textsurface = font.render(f"Step {self.step}: {self.status}", False, (255, 255, 255))
        r = textsurface.get_rect()
        surf.blit(textsurface, r)

        pygame.display.flip()


handlers: "list[Callable]" = []


def OnGraphChanged(func: "Callable"):
    handlers.append(func)
    return func


if __name__ == "__main__":

    @OnGraphChanged
    def TestRegions(graph: "Graph"):
        graph.groups = [
            ["a"],
            ["b", "c"],
            ["d", "e", "f"],
        ]
        # graph.focused_tris = []
        # for t in graph.triangulation.TrianglesOfVertex(graph.triangulation.verts[-1]):
        #     # print(t)
        #     graph.focused_tris.append(t.verts)
        pass

    Graph().JitterPoints(3, 2).Run()
