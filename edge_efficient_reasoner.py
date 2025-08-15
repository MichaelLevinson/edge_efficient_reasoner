"""
Edge-Efficient Reasoner — Low-Rank Adapter Construction
Implements Addenda A1, A2 & A3 from the formal proof.

Notation (↔ proof):
    X  : stacked hidden states  (A2-R)
    Y  : ±1 label matrix        (A1-T)
    U  : (XᵀX)⁻¹XᵀY            (A1-T separator)
    V  : d×r  basis whose first r rows are I_r   (A3-P requirement)
    δ  : U Vᵀ                   (low-rank adapter)

All tensors live on the device chosen by `choose_device()`.

regarding boolean_decode >=0 for logits iff the following is changed:
    # add a constant 1 column to X before training
    X_aug = torch.cat([X, torch.ones(len(X),1, device=X.device)], dim=1)  # (m,d+1)
    U = build_separator(X_aug, Y)    # U now has a learned bias in its last row

    # when decoding, concatenate the same 1 to h
    logits = torch.cat([h, torch.ones(h.size(0),1, device=h.device)], dim=1) @ (U @ Vt.T)
    return (logits >= 0).int()       # ≥ is fine once you have a bias


"""

from __future__ import annotations
import torch
import numpy as np

# ───────────────────────── device selection (codeEx2 guideline) ──────────────
def choose_device() -> torch.device:
    """
    Prefer Apple-silicon MPS → CUDA → CPU, as recommended by codeEx2.md.
    """
    if torch.backends.mps.is_available():
        return torch.device("cpu") # not functioning on MPS yet
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = choose_device()
print(f"Using device → {DEVICE}")

# ─────────────────────────────── core construction ───────────────────────────
def build_separator(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Compute U = (XᵀX)⁻¹ Xᵀ Y  (Addendum A1-T).

    X : (m,d) full-column-rank matrix
    Y : (m,r) ±1 label matrix   (r ≤ 64)
    Returns
    -------
    U : (d,r) tensor on the same device as X
    """
    gram = X.T @ X                                       # (d,d)
    U = torch.linalg.solve(gram, X.T @ Y)                # (d,r)
    return U


def make_structured_basis(d: int, r: int) -> torch.Tensor:
    """
    Build V that satisfies proof condition V = [I_r ; 0] ⇒ w_i = e_i.

    VᵀV = I_r automatically holds, so rank(V) = r (Addendum A3-P).

    Raises
    ------
    ValueError  if r > d
    """
    if r > d:
        raise ValueError("Adapter rank r cannot exceed hidden size d.")
    V = torch.zeros(d, r, device=DEVICE)
    V[:r, :] = torch.eye(r, device=DEVICE)               # identity block
    return V                                             # (d,r)


def build_adapter(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Pipeline:   1. rank check   2. U   3. structured V   4. δ = U Vᵀ
    """
    # Step 1 — full-rank check (Addendum A2-R)
    rank = np.linalg.matrix_rank(X.cpu().numpy())
    if rank < X.shape[1]:
        raise ValueError("X is not full column-rank; abort per A2-R.")

    # Step 2 — separator
    U = build_separator(X, Y)                            # (d,r)

    # Step 3 — proof-compatible V
    V = make_structured_basis(X.shape[1], Y.shape[1])    # (d,r)

    # Step 4 — adapter
    delta = U @ V.T                                      # (d,d)
    return delta


# ───────────────────────────── Boolean thresholding ──────────────────────────
def boolean_decode(U: torch.Tensor, h: torch.Tensor,
                   Vt: torch.Tensor | None = None) -> torch.Tensor:
    """
    Compute σ(wᵢ·Uᵀh) for every rule i (Addendum A1-T).

    If Vt ( = Vᵀ ) is None we use the proof-mandated Vᵀ = [I_r | 0]ᵀ,
    i.e. weight vectors wᵢ = eᵢ.

    Returns
    -------
    bits : 0/1 tensor of shape (batch, r)
    """
    r = U.shape[1]
    if Vt is None:
        Vt = torch.eye(r, device=U.device)               # rows = e_i
    logits = h @ (U @ Vt.T)                              # (batch,r)
    return (logits > 0).int()


# ───────────────────────────── verification unit test ────────────────────────
def _unit_test() -> None:
    """
    End-to-end verification on the 2-atom truth table (m = 4).

    Checks:
      1. X is full-rank, δ has rank r.          (A2-R, A3-P)
      2. Boolean decoding recovers ground truth (A1-T)
    """
    # Truth table for two propositional atoms
    truth_table = torch.tensor([[0, 0],
                                [0, 1],
                                [1, 0],
                                [1, 1]], dtype=torch.float32)

    X = truth_table.to(DEVICE)                 # (4,2), rank = 2
    Y = truth_table * 2 - 1                    # map {0,1}→{-1,+1}
    r = Y.shape[1]

    # Build adapter
    delta = build_adapter(X, Y)
    assert torch.linalg.matrix_rank(delta).item() == r, "Rank(δ) ≠ r"

    # Boolean recovery
    U = build_separator(X, Y)
    preds = boolean_decode(U, X)               # (4,2) 0/1
    assert torch.equal(preds, truth_table.int()), "Boolean mismatch"

    print("✓ Unit test passed — all proof constraints satisfied.")


if __name__ == "__main__":
    _unit_test()
