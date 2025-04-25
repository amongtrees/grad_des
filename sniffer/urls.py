from django.urls import path
from django.views.generic import TemplateView

from .views import ana_test, sniffer

urlpatterns = [
    # path('', ana_test.ana_test, name='index'),
    path('start_sniffer/', sniffer.start_sniffer),
    path('stop_sniffer/', sniffer.stop_sniffer),
    path('get_interfaces/', sniffer.get_interfaces),
    path('', TemplateView.as_view(template_name='new.html')),
]

