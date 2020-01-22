"""Routines which are useful to create and plot graphs
"""
from typing import Set, Tuple, Optional, List

from numpy import random as rand
import matplotlib.pyplot as plt

import networkx as nx


def generate_graph(
    n_nodes: int, n_edges: int, n_edge_max: int = 5, seed: Optional[int] = None
) -> Set[Tuple[int]]:
    """Creates random edges for graph.

    Node labels are from 0 to n_nodes - 1

    Arguments:
        n_nodes: Number of nodes.
        n_edges: Number of edges.
        n_edge_max: Maximal number of edges for graph.
        seed: Random seed to set before graph is generated.

    Todo:
        What about connected / unconnected graphs?
        Am I creating the right graph?
        Think about ill input conditions.
    """
    assert n_edge_max > 0
    assert n_nodes * n_edge_max // 2 > n_edges

    if seed:
        rand.seed(42)

    nodes = {n: 0 for n in range(n_nodes)}
    edges = set()

    while len(edges) < n_edges:
        choices = [n for n, count in nodes.items() if count <= n_edge_max]
        e1, e2 = (int(e) for e in rand.choice(choices, size=2, replace=False))
        edge = (e2, e1) if e2 < e1 else (e1, e2)
        if edge not in edges:
            edges.add(edge)
            nodes[e1] += 1
            nodes[e2] += 2

    return edges


def get_plot(
    graph: Set[Tuple[int]],
    color_nodes: Optional[List[int]] = None,
    show_plot: bool = False,
    backend="mpl",
):
    """Creates graph plot from edge set.

    Arguments:
        graph: The graph to plot.
        color_nodes: Colors given nodes green, default is white.
        show_plot: Wether or not the plot should be rendered.
        backend: The plotting backend. Either "mpl" or "bokeh".
    """
    if backend == "mpl":
        fig = get_plot_mpl(graph, color_nodes=color_nodes, show_plot=show_plot)
    elif backend == "bokeh":
        fig = get_plot_bokeh(graph, color_nodes=color_nodes, show_plot=show_plot)
    else:
        raise KeyError("Only implemented matplotlib ('mpl') or ")

    return fig


def get_plot_mpl(
    graph: Set[Tuple[int]],
    color_nodes: Optional[List[int]] = None,
    show_plot: bool = False,
    ax: Optional["Axes"] = None,
) -> Optional["Figure"]:
    """Creates matplotlib graph plot from edge set.

    Arguments:
        graph: The graph to plot.
        color_nodes: Colors given nodes green, default is white.
        show_plot: Wether or not the plot should be rendered.
        ax: The axes to plot in
    """
    if not ax:
        fig, ax = plt.subplots()
    else:
        fig = None

    G = nx.Graph(list(graph))
    node_color = (
        ["lightgreen" if node in color_nodes else "white" for node in G.nodes]
        if color_nodes
        else "white"
    )

    nx.draw(
        G,
        with_labels=True,
        node_color=node_color,
        font_size=10,
        node_shape="o",
        linewidths=1.0,
        edgecolors="black",
        ax=ax,
    )

    if show_plot:
        plt.show()

    return fig


def get_plot_bokeh(  # pylint: disable=R0914
    graph: Set[Tuple[int]],
    color_nodes: Optional[List[int]] = None,
    show_plot: bool = False,
) -> "Plot":
    """Creates bokeh graph plot from edge set.

    Arguments:
        graph: The graph to plot.
        color_nodes: Colors given nodes green, default is white.
        show_plot: Wether or not the plot should be rendered.
    """
    from bokeh.io import show
    from bokeh.models import Plot, Range1d, Circle, HoverTool
    from bokeh.models.graphs import from_networkx

    color_nodes = color_nodes or []

    G = nx.Graph(list(graph))
    edge_attrs = {}
    node_attrs = {}

    for start_node, end_node, _ in G.edges(data=True):
        edge_attrs[(start_node, end_node)] = "black"

    for node in G.nodes:
        node_attrs[node] = "green" if node in color_nodes else "white"

    nx.set_edge_attributes(G, edge_attrs, "edge_color")
    nx.set_node_attributes(G, node_attrs, "node_color")

    # Show with Bokeh
    plot = Plot(
        plot_width=400,
        plot_height=400,
        x_range=Range1d(-1.1, 1.1),
        y_range=Range1d(-1.1, 1.1),
    )

    graph_renderer = from_networkx(G, nx.spring_layout, scale=1, center=(0, 0))
    graph_renderer.node_renderer.glyph = Circle(size=15, fill_color="node_color")
    plot.renderers.append(graph_renderer)  # pylint: disable=E1101

    node_hover_tool = HoverTool(tooltips=[("index", "@index")])
    plot.add_tools(node_hover_tool)

    if show_plot:
        show(plot)

    return plot


def main():
    """Creates a random graph (fixed seed) and plots it.
    """

    n_nodes = 5
    n_edges = 5
    n_edge_max = 3

    test_graph = generate_graph(n_nodes, n_edges, n_edge_max=n_edge_max, seed=42)
    get_plot(test_graph, show_plot=True)


if __name__ == "__main__":
    main()
