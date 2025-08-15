A.1 Boolean Thresholding
Addendum A1-T (Proof).
Let σ:ℝ→{0,1} be the Heaviside step σ(z)=1 iff z≥0.
Redefine the decoding requirement: σ(wᵢ·Uᵀx) = ℓᵢ(x).
Given the linear system U=(XᵀX)^{-1}XᵀY, choose training matrix Y∈{+1,−1}^{m×|L|} with +1 for “true”, −1 for “false”. Because the rows of Y are sign-separable by assumption (linear separability), the existence of a separating hyperplane follows from the Perceptron Convergence Theorem; the closed-form U we compute is one such separator. Therefore applying σ completes Boolean decoding. □

A.2 Full-Rank Guarantee
Addendum A2-R (Proof).
Sampling procedure: enumerate the full truth-table of |L| independent propositional atoms ⇒ m=2^{|Atoms|}. Hidden-state encoder h:Σ*→ℝᵈ is injective on truth-table inputs under the usual positional-encoding map (distinct tokens → orthogonal basis). Hence the matrix X of stacked h-vectors has rank m (≤d by padding). Because |L|≤r≤64≪m, XᵀX is positive-definite ⇒ invertible. □

A.3 Adapter Rank Preservation
Addendum A3-P (Proof).
rank(δ)=rank(UVᵀ)≥min(rank U, rank V). We set rank V=r by construction (orthonormal rows). U inherits that rank because X has rank m≥r and the normal equations preserve full column rank. Hence rank(δ)=r barring numerical round-off smaller than 2^{-24}; we treat this as machine-precision error and accept ε-rank equality. □
