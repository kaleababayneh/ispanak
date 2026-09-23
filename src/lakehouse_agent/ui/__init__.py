"""The web interface: a second, independent consumer of the agent's event
stream (`agent/events.py`), next to the terminal one (`agent/console.py`).

`render.py` is pure and network-free, tested without a browser; `app.py` is
the thin Chainlit adapter that calls it. Neither is imported by the engine.
"""
