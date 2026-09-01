"""Generate the reference testset once using an external API.

External LLM calls are intentionally confined to this module.  Reference
contexts are never synthesized: they are exact chunk texts returned by the
existing ``src.chunking.chunk_text`` implementation.
"""