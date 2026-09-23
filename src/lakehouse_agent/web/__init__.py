"""Reading a web address the question names: fetching and extraction.

Split in two on purpose. `fetch` is the only part that touches the network,
so everything about parsing a PDF, a spreadsheet or a page is tested with
bytes on disk and no network at all (CLAUDE.md #5).
"""
