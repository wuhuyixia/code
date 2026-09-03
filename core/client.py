from __future__ import annotations


class Client:
    """Client container for local model, optimizer, data loaders, and state."""

    def __init__(self, client_id: int, model, optimizer=None, train_loader=None):
        self.client_id = int(client_id)
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
