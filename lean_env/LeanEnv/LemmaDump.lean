/-
  LeanEnv/LemmaDump.lean
  ----------------------
  Defines a command `dumpLemmas "path.jsonl"` that writes Prop-typed
  constants (theorems/axioms) in the environment to JSONL.

  Filters applied:
    - drop names containing "._proof_" or "._match_"
    - drop names ending with ".rec", ".recOn", ".brecOn"
    - drop statements that contain the pretty-printer ellipsis "⋯"

  Each JSON line:
    { "name": "<fully.qualified.name>"
    , "base": "<basename>"
    , "stmt": "<pretty-printed statement>"
    , "module": "<declaring module (best effort)>"
    }

  Do NOT invoke it here; use a separate runner file to avoid heavy work in the editor.
-/

import Mathlib
open Lean Meta Elab Command System

/-- Pretty-print an expression. -/
def ppExprToString (e : Expr) : MetaM String := do
  let fmt ← PrettyPrinter.ppExpr e
  pure fmt.pretty

/-- Best-effort module name for a constant. -/
def moduleOfConst (env : Environment) (nm : Name) : String :=
  match env.getModuleIdxFor? nm with
  | some i => (env.allImportedModuleNames[i]!).toString
  | none   => ""

/-- Simple substring test (Lean's `String.contains` takes a `Char`, not `String`). -/
@[inline] def hasSubstr (s pat : String) : Bool :=
  (s.splitOn pat).length > 1

/-- Heuristic: keep only human-facing lemma names. -/
def goodName (nm : Name) : Bool :=
  let s := nm.toString
  not (hasSubstr s "._proof_"
       || hasSubstr s "._match_"
       || s.endsWith ".rec"
       || s.endsWith ".recOn"
       || s.endsWith ".brecOn")

/-- Dump all (filtered) Prop-typed constants to a JSONL file. -/
elab "dumpLemmas" path:str : command => do
  let outPath : System.FilePath := (path.getString : String)

  -- Ensure parent directory exists (handle Option FilePath properly)
  match outPath.parent with
  | some parent => do
      let dirExists ← parent.pathExists
      if !dirExists then
        IO.FS.createDirAll parent
  | none => pure ()

  -- Open the output file for writing
  let h ← IO.FS.Handle.mk outPath IO.FS.Mode.write

  -- We need Meta to check `isProp`
  runTermElabM fun _ => do
    let env ← getEnv
    for (nm, ci) in env.constants.toList do
      if goodName nm then
        let ty? :=
          match ci with
          | ConstantInfo.thmInfo i   => some i.type
          | ConstantInfo.axiomInfo i => some i.type
          | _                        => none
        match ty? with
        | none => pure ()
        | some ty => do
            if (← isProp ty) then
              let stmt ← ppExprToString ty
              -- drop heavily elided structure equalities / huge terms
              if hasSubstr stmt "⋯" then
                pure ()
              else
                let mod  := moduleOfConst env nm
                let full := nm.toString
                let base := (full.splitOn ".").getLastD full
                let j := Json.mkObj
                  [ ("name",   Json.str full)
                  , ("base",   Json.str base)
                  , ("stmt",   Json.str stmt)
                  , ("module", Json.str mod)
                  ]
                discard <| h.putStr (Json.compress j ++ "\n")
  h.flush
