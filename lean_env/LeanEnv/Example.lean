import Mathlib

variable {R : Type*} [Semiring R]

theorem cast_add_rw (n m : ℕ) : ((n + m : ℕ) : R) = (n : R) + (m : R) := by
  -- @TACTICS@
