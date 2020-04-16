from django.views.generic import DetailView
from django.views.generic.list import ListView

from qlpdb.tdse.models import Tdse


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
