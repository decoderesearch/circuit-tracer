"""The intervention tuple, and the one rewriting of it that every backend needs.

Both definitions below existed as byte-identical copies in the TransformerLens and nnsight
backends. A third backend would have made a third copy of a rule about what user input means,
which is the kind of thing that drifts silently.
"""

from typing import Sequence

import torch

# (layer, position, feature index, value). The position may be a slice, which is how a caller says
# "this feature, everywhere from here on" -- see `convert_open_ended_interventions`.
Intervention = tuple[
    int | torch.Tensor,
    int | slice | torch.Tensor,
    int | torch.Tensor,
    int | float | torch.Tensor,
]


def convert_open_ended_interventions(
    interventions: Sequence[Intervention],
) -> list[Intervention]:
    """Keep the interventions that outlive the prompt, retargeted at position 0.

    An intervention is *open-ended* if its position is a ``slice`` with no ``stop``
    (``slice(6, None)``), meaning it applies to every position from its start onwards -- including
    positions that do not exist yet, because generation has not reached them. During generation
    with a KV cache each incremental forward pass sees exactly one position, so "from 6 onwards"
    becomes "position 0" once the prompt is behind us. Interventions with a concrete position
    named a token in the prompt and are dropped here: they have already been applied.
    """
    converted: list[Intervention] = []
    for layer, pos, feature_idx, value in interventions:
        if isinstance(pos, slice) and pos.stop is None:
            converted.append((layer, 0, feature_idx, value))
    return converted
