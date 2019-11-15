"""Plotting setup for example problem
"""
import numpy as np
import plotly.graph_objects as go

from qlp.example import f

LAYOUT = go.Layout(
    paper_bgcolor="white", plot_bgcolor="white", height=600, width=600, showlegend=False
)


def _get_f_data(zmin: float = 6.0) -> go.Heatmap:
    """Plots heatmap for f
    """
    zmin = 6.0

    xx, yy = np.meshgrid(*[np.linspace(0, 4, 100)] * 2)
    zz = f(xx, yy)
    zz = np.where(zz < zmin, np.nan, zz)
    return go.Heatmap(
        x=np.linspace(0, 4, 100),
        y=np.linspace(0, 4, 100),
        z=zz,
        colorscale="Blues",
        zmin=zmin,
        zsmooth="best",
    )


F_DATA = _get_f_data()
