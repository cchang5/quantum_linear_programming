from django.views.generic import DetailView, TemplateView
from django.views.generic.list import ListView

from plotly.graph_objs import Scatter
from plotly.subplots import make_subplots
from plotly.offline import plot

from qlpdb.tdse.models import Tdse

from qlpdb.tdse.plotting.tdse_plots import aggregate


class TdseView(DetailView):
    """View for visualizing tdse
    """

    template_name = "tdse.html"
    model = Tdse


class TdseSummaryView(ListView):
    """
    """

    template_name = "tdse-summary.html"
    model = Tdse
    ordering = ["-tag"]


class ComparisonView(TemplateView):
    """
    """

    template_name = "plot.html"

    def get_context_data(self, **kwargs):
        """
        """
        context = super().get_context_data(**kwargs)

        data = aggregate()

        fig = make_subplots(subplot_titles=["Probability", "Entropy"], cols=2)

        for key, val in data.items():
            fig.add_trace(
                Scatter(x=val.time, y=val.prob, name=key, showlegend=False),
                col=1,
                row=1,
            )
            fig.add_trace(Scatter(x=val.time, y=val.entropy, name=key), col=2, row=1)

        context["graph"] = plot(fig, auto_open=False, output_type="div")

        return context
