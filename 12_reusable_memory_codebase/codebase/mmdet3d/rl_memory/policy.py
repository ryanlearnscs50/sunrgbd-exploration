"""
Policy network for class-wise memory allocation.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryAllocPolicy(nn.Module):
    """Small MLP that maps an RL state vector to allocation logits.

    Args:
        state_dim: Dimensionality of the flattened state vector.
        num_classes: Number of object classes to allocate memory for.
        hidden_dim: Hidden width of the MLP.
    """

    def __init__(self, state_dim: int, num_classes: int, hidden_dim: int = 128):
        super().__init__()
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")

        self.state_dim = int(state_dim)
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)

        self.fc1 = nn.Linear(self.state_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute allocation logits from a batch of state vectors.

        Args:
            state: Tensor of shape [batch_size, state_dim] or [state_dim].

        Returns:
            Tensor of shape [batch_size, num_classes] containing unnormalized
            allocation logits.
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if state.size(-1) != self.state_dim:
            raise ValueError(
                f"Expected state_dim={self.state_dim}, got {state.size(-1)}"
            )

        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        logits = self.out(x)
        return logits


def allocation_from_logits(
    logits: torch.Tensor,
    total_object_slots: int,
) -> Dict[int, int]:
    """Convert class-wise logits into integer allocation targets.

    Steps:
      1) Apply softmax to get per-class probabilities.
      2) Multiply by total_object_slots.
      3) Round to integers.
      4) Ensure at least 1 slot per class.
      5) Adjust allocations so the total equals total_object_slots.

    Args:
        logits: Tensor of shape [batch_size, num_classes] or [num_classes].
        total_object_slots: Total number of object slots across all classes.

    Returns:
        Dict[int, int]: Mapping class_id -> target object count T_c for a
        single batch element (batch size is assumed to be 1 if present).
    """
    if total_object_slots <= 0:
        raise ValueError(
            f"total_object_slots must be positive, got {total_object_slots}"
        )

    if logits.dim() == 2:
        if logits.size(0) != 1:
            raise ValueError(
                "allocation_from_logits currently supports batch_size == 1, "
                f"got batch_size={logits.size(0)}"
            )
        logits = logits[0]
    elif logits.dim() != 1:
        raise ValueError(
            f"Expected logits with dim 1 or 2, got shape {tuple(logits.shape)}"
        )

    num_classes = logits.numel()
    probs = F.softmax(logits, dim=-1)

    # Initial allocation by rounding expected counts
    expected_counts = probs * float(total_object_slots)
    alloc = torch.round(expected_counts).to(torch.long)

    # Ensure at least 1 slot per class
    alloc = torch.clamp(alloc, min=1)

    # Adjust total to exactly match total_object_slots
    total_alloc = int(alloc.sum().item())
    diff = total_alloc - int(total_object_slots)

    # Convert to Python list for easier deterministic adjustments
    alloc_list = [int(x) for x in alloc.tolist()]
    prob_list = [float(x) for x in probs.tolist()]

    if diff > 0:
        # Need to remove diff slots, prioritizing classes with largest allocation
        while diff > 0:
            # Sorted indices by decreasing allocation, then decreasing probability
            indices = sorted(
                range(num_classes),
                key=lambda i: (alloc_list[i], prob_list[i]),
                reverse=True,
            )
            changed = False
            for idx in indices:
                if alloc_list[idx] > 1:
                    alloc_list[idx] -= 1
                    diff -= 1
                    changed = True
                    if diff == 0:
                        break
            if not changed:
                break
    elif diff < 0:
        # Need to add -diff slots, prioritizing classes with highest probability
        while diff < 0:
            indices = sorted(
                range(num_classes),
                key=lambda i: prob_list[i],
                reverse=True,
            )
            for idx in indices:
                alloc_list[idx] += 1
                diff += 1
                if diff == 0:
                    break

    allocation: Dict[int, int] = {i: int(alloc_list[i]) for i in range(num_classes)}
    return allocation

