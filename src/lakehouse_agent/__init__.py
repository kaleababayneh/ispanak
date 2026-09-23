"""KKB Hackathon 2026 lakehouse agent."""


def main() -> int:
    """The installed `lakehouse-agent` command.

    A thin forward to the module entry point, so `lakehouse-agent --real
    --chat` and `python -m lakehouse_agent.agent.loop --real --chat` are the
    same program. Imported inside the function: the console script is
    resolved at install time and importing the loop at package import would
    make `import lakehouse_agent` pull in pandas and the whole tool layer.
    """
    from lakehouse_agent.agent.loop import _cli

    return _cli()
