"""Verified HTTPS using the operating system trust store plus requests' CAs."""

import ssl

import requests
from requests.adapters import HTTPAdapter


class SystemTrustAdapter(HTTPAdapter):
    def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
        # On Windows create_default_context also loads Windows ROOT/CA stores.
        # requests still applies its configured CA bundle. Neither path disables
        # chain or hostname validation, including when a proxy is configured.
        pool_kwargs["ssl_context"] = ssl.create_default_context()
        return super().init_poolmanager(connections, maxsize, block=block, **pool_kwargs)

    def proxy_manager_for(self, proxy, **proxy_kwargs):
        proxy_kwargs.setdefault("ssl_context", ssl.create_default_context())
        return super().proxy_manager_for(proxy, **proxy_kwargs)


def verified_session():
    session = requests.Session()
    session.mount("https://", SystemTrustAdapter())
    return session
