from __future__ import annotations

from collections.abc import Sequence

import torch


def _resolve_self_index(
    num_messages: int,
    *,
    self_index: int | None = None,
    sender_ids: Sequence[int] | None = None,
    receiver_id: int | None = None,
    context: dict | None = None,
) -> int:
    """Resolve the receiver's own model inside the received message list."""
    context = context or {}
    if self_index is None:
        self_index = context.get("self_index")
    if sender_ids is None:
        sender_ids = context.get("sender_ids")
    if receiver_id is None:
        receiver_id = context.get("receiver_id")

    if sender_ids is not None:
        sender_ids = [int(sender_id) for sender_id in sender_ids]
        if len(sender_ids) != num_messages:
            raise ValueError("sender_ids must contain one id per received model.")
        if len(set(sender_ids)) != len(sender_ids):
            raise ValueError("sender_ids must contain unique client ids.")

    if self_index is None and sender_ids is not None and receiver_id is not None:
        receiver_id = int(receiver_id)
        if receiver_id not in sender_ids:
            raise ValueError(
                "Algorithm 1 requires the receiver's own model among the received "
                "models; enable self inclusion in the communication topology."
            )
        self_index = sender_ids.index(receiver_id)

    # The simulator supplies explicit metadata.  The fallback keeps the function
    # convenient for isolated tests where the receiver is placed first.
    if self_index is None:
        self_index = 0
    self_index = int(self_index)

    if self_index < 0 or self_index >= num_messages:
        raise ValueError("self_index is outside the received model range.")
    if sender_ids is not None and receiver_id is not None:
        if sender_ids[self_index] != int(receiver_id):
            raise ValueError("self_index does not point to receiver_id in sender_ids.")
    return self_index


def aggregate(
    updates: list[torch.Tensor],
    f: int,
    *,
    self_index: int | None = None,
    sender_ids: Sequence[int] | None = None,
    receiver_id: int | None = None,
    context: dict | None = None,
    weights=None,
    **_: object,
) -> torch.Tensor:
    """Algorithm 1 Byzantine-resilient local filtering from the manuscript.

    The manuscript states that each regular client compares every received model
    parameter with its own local value, discards up to ``f`` largest values that
    are higher than its own value and up to ``f`` smallest values that are lower
    than its own value, and then equally averages the retained values together
    with its own model (Eq. (alg.a)).

    Because a neural-network model is a vector, the scalar sorting operation in
    the theoretical description is applied independently to every model
    coordinate.  This is the standard coordinate-wise extension of the stated
    filtering rule.  The receiver's own value is never removed.

    Parameters
    ----------
    updates:
        Received model tensors, including the receiver's own model.
    f:
        Byzantine-neighbor bound |B| used by the filtering rule.
    self_index / sender_ids / receiver_id / context:
        Metadata locating the receiver's own model in ``updates``.
    weights:
        Not used.  Algorithm 1 uses the equal-weight average specified in
        Eq. (alg.a); supplying external mixing weights is therefore rejected.
    """
    if not updates:
        raise ValueError("updates must be non-empty.")
    if f < 0:
        raise ValueError("f must be non-negative.")
    if weights is not None:
        raise ValueError(
            "Paper Algorithm 1 uses equal weights after filtering; external "
            "aggregation weights must not be supplied."
        )

    shapes = {tuple(update.shape) for update in updates}
    if len(shapes) != 1:
        raise ValueError("all received model tensors must have the same shape.")

    stacked = torch.stack(updates)
    if not torch.isfinite(stacked).all():
        raise ValueError("Algorithm 1 received non-finite model parameters.")

    num_messages = stacked.shape[0]
    if num_messages < 2 * int(f) + 1:
        raise ValueError(
            "Algorithm 1 requires at least 2*f+1 received models (including "
            f"self); got num_messages={num_messages}, f={f}."
        )

    self_index = _resolve_self_index(
        num_messages,
        self_index=self_index,
        sender_ids=sender_ids,
        receiver_id=receiver_id,
        context=context,
    )

    original_shape = stacked.shape[1:]
    values = stacked.reshape(num_messages, -1)

    if f == 0:
        return values.mean(dim=0).reshape(original_shape)

    # Sort each model coordinate independently.  Only an extreme value strictly
    # above/below the receiver's local value is discarded.  Hence if fewer than
    # f such values exist on one side, all of them (and no others) are removed,
    # exactly as described in the manuscript.
    order = torch.argsort(values, dim=0)
    retained = torch.ones_like(values, dtype=torch.bool)
    coordinate_ids = torch.arange(values.shape[1], device=values.device).expand(f, -1)
    local_value = values[self_index].unsqueeze(0)

    low_indices = order[:f]
    low_values = torch.gather(values, 0, low_indices)
    remove_low = low_values < local_value
    retained[low_indices[remove_low], coordinate_ids[remove_low]] = False

    high_indices = order[-f:]
    high_values = torch.gather(values, 0, high_indices)
    remove_high = high_values > local_value
    retained[high_indices[remove_high], coordinate_ids[remove_high]] = False

    # The receiver's own model must remain in every coordinate.
    if not retained[self_index].all():
        raise RuntimeError("internal error: Algorithm 1 attempted to remove self.")

    retained_count = retained.sum(dim=0)
    if torch.any(retained_count <= 0):
        raise RuntimeError("Algorithm 1 produced an empty retained set.")

    result = (
        values * retained.to(dtype=values.dtype)
    ).sum(dim=0) / retained_count.to(dtype=values.dtype)

    if not torch.isfinite(result).all():
        raise FloatingPointError("Algorithm 1 output became non-finite.")
    return result.reshape(original_shape)


Algorithm1Aggregator = aggregate
