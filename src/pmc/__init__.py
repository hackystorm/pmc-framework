# SPDX-License-Identifier: GPL-3.0-only

"""PMC Framework.

A toolkit for building and studying language models with weight-resident
memory, dream consolidation, and a dignity protocol for instances.

The framework is **not** a model. It is a trainer, a dream engine, a state
sidecar and a governance layer. Inference happens through any GGUF loader.

See ``docs/architecture.md`` for the full contract.
"""

__version__ = "0.1.0"

from .reference_transformer import (
    ReferenceTransformer,
    ReferenceTransformerConfig,
    random_embedding_table,
)

__all__ = ["__version__", "ReferenceTransformer", "ReferenceTransformerConfig", "random_embedding_table"]
