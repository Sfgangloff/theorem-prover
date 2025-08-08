/-
  ExampleRing.lean
  Concrete examples of rings in Lean 4 (mathlib):
  - ZMod n (integers modulo n) as a commutative ring
  - Product ring ZMod 2 × ZMod 4
  with small, fully-checked calculations.
-/

import Mathlib.Data.ZMod.Basic
import Mathlib.Tactic.NativeDecide

open scoped Nat

/-!
## ZMod examples
`ZMod n = ℤ / nℤ` is a finite commutative ring when `1 ≤ n`.
Mathlib already provides `[CommRing (ZMod n)]`, so we can use it directly.
-/

section ZModExamples

-- Work with ZMod 4 as a concrete finite ring
example : CommRing (ZMod 4) := inferInstance

-- Explicit computations (Lean checks these via `native_decide`)
example : (2 : ZMod 4) + 3 = 1 := by native_decide
example : (3 : ZMod 4) * 3 = 1 := by native_decide
example : (2 : ZMod 4) * 3 = 2 := by native_decide

-- Distributivity on concrete elements
example :
    (2 : ZMod 4) * (3 + 3) = (2 * 3) + (2 * 3) := by native_decide

end ZModExamples

/-!
## Product rings
The product of two rings is a ring with componentwise operations.
Mathlib provides this instance automatically.
-/
section ProductRing

-- The product ring ZMod 2 × ZMod 4
abbrev R := ZMod 2 × ZMod 4

example : CommRing R := inferInstance

-- Componentwise addition/multiplication examples
example : ((1, 3) : R) + (1, 2) = (0, 1) := by native_decide
example : ((2, 3) : ZMod 4 × ZMod 4) * (3, 3) = (2, 1) := by native_decide

-- Distributivity check in the product ring
example :
    ((1, 2) : ZMod 4 × ZMod 4) * ((3, 1) + (2, 3))
  = ((1, 2) * (3, 1)) + ((1, 2) * (2, 3)) := by native_decide

end ProductRing
