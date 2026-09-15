"""Tools the model can call.

Each tool is a plain synchronous function plus a Pydantic argument model. The
loop validates arguments before calling, runs the function off the event loop
and turns whatever comes back into one tool message.
"""
