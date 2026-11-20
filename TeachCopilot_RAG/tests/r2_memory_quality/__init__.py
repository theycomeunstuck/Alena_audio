"""Round-2 synthetic archive quality package (client A).

Frozen six-session synthetic histories, the immutable source-key mapping, the
loader that inserts them into a disposable Store, and the gold manifest. The
gold manifest is frozen before client B's live run; expected answers and
source keys never enter generation, resolver or extraction prompts.
"""
