# sniffer/routing.py
from django.urls import path
from . import consumers

websocket_urlpatterns = [
    path('ws/sniffer/', consumers.SnifferConsumer.as_asgi()),
]