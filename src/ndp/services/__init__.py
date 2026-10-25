"""Thin clients for external services. Every function here degrades gracefully: missing
configuration or a network failure becomes a `ServiceError` the UI can display."""


class ServiceError(RuntimeError):
    pass
