"""
ASGI config for project project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.2/howto/deployment/asgi/
"""

import os

from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from sniffer import routing

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project.settings')
django_application = get_asgi_application()
application = ProtocolTypeRouter({
    'http': get_asgi_application(),
    'websocket':URLRouter(routing.websocket_urlpatterns),  # WebSocket 路由
})