import LeanEnv.ExtractionTactic
-- file: examples/IdApply.lean
import Lean
open Lean Meta Elab Tactic

theorem id_apply (α : Type) (x : α) : id x = x := by
    logStepAuto simp

