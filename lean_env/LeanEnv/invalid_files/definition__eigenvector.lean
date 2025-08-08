import Mathlib.LinearAlgebra.Eigenspace

open scoped BigOperators

section

variable {K V : Type*} [Field K] [AddCommGroup V] [Module K V]

/-- `v` is an eigenvector of a linear map `f` with eigenvalue `μ` iff
    `v ≠ 0` and `f v = μ • v`.  (We give a standalone definition to make
    the notion explicit, although mathlib already has `LinearMap.eigenspace`
    and `IsEigenvalue`.)
-/
def IsEigenvector (f : V →ₗ[K] V) (μ : K) (v : V) : Prop :=
  v ≠ 0 ∧ f v = μ • v

/-- Bridge between our `IsEigenvector` and mathlib’s `eigenspace`. -/
lemma isEigenvector_iff_mem_eigenspace
    {f : V →ₗ[K] V} {μ : K} {v : V} :
    IsEigenvector (K := K) (V := V) f μ v
      ↔ v ≠ 0 ∧ v ∈ LinearMap.eigenspace f μ := by
  constructor
  · intro h
    refine ⟨h.1, ?_⟩
    -- membership in the eigenspace is exactly the defining equation
    simpa [LinearMap.mem_eigenspace] using h.2
  · rintro ⟨hv, hv_mem⟩
    exact ⟨hv, by simpa [LinearMap.mem_eigenspace] using hv_mem⟩

/-- If `v` is an eigenvector for `μ`, then any nonzero scalar multiple of `v`
    is also an eigenvector for the same `μ`. -/
lemma IsEigenvector.smul {f : V →ₗ[K] V} {μ : K} {v : V} {c : K}
    (h : IsEigenvector (K := K) (V := V) f μ v) (hc : c ≠ 0) :
    IsEigenvector (K := K) (V := V) f μ (c • v) := by
  -- Use the eigenspace formulation to avoid re-proving linearity facts.
  have hv_mem : v ∈ LinearMap.eigenspace f μ := by
    have : v ≠ 0 ∧ v ∈ LinearMap.eigenspace f μ :=
      (isEigenvector_iff_mem_eigenspace (K := K) (V := V)).1 h
    exact this.2
  have hcv_mem :
      c • v ∈ LinearMap.eigenspace f μ := by
    -- eigenspace is a submodule, hence closed under `K`-scalar multiplication
    simpa using (Submodule.smul_mem (LinearMap.eigenspace f μ) c hv_mem)
  -- Show `c • v ≠ 0` using invertibility of `c`.
  have hcv_ne : c • v ≠ 0 := by
    intro hzero
    -- Multiply by `c⁻¹` on the left; over a field this cancels `c`.
    have : v = (c⁻¹) • (c • v) := by
      -- `v = 1 • v = (c⁻¹ * c) • v`
      simpa [smul_smul, inv_mul_cancel hc, one_smul]
    have : v = 0 := by simpa [hzero, smul_zero] using this
    exact h.1 this
  -- Convert back from eigenspace membership.
  exact (isEigenvector_iff_mem_eigenspace (K := K) (V := V)).2 ⟨hcv_ne, hcv_mem⟩

/-- (Optional) A convenience predicate for eigenvalues, matching the usual math definition. -/
def IsEigenvalue (f : V →ₗ[K] V) (μ : K) : Prop :=
  ∃ v : V, IsEigenvector (K := K) (V := V) f μ v

end
