"""The agent core: session state, events, the turn loop and its guards.

Nothing in this package imports a UI framework. The loop publishes events and
awaits an approval callback; whatever draws them is the caller's business. That
keeps the engine testable without a browser and lets the interface be replaced
without touching the engine.
"""
