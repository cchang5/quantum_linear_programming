from django.views.generic import DetailView

from qlpdb.experiment.models import Experiment

# Create your views here.


class ExperimentView(DetailView):
    """View for visualizing experiments
    """

    template_name = "experiments.html"
    model = Experiment

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        experiment = context["experiment"]  # this is the instance of experiment 1
        return context
