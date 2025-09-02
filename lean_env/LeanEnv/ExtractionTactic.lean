/-
  LeanEnv/Logger.lean  (strict-gated auto logger)
  ------------------------------------------------
  - `logStep`: original logger (mathlib scraping). Writes to data/goal_tactic_log.jsonl.
  - `logStepAuto`: STRICT version for auto-prover runs.
      * Requires env:
          RUN_JSON_PATH    -> output file (no fallback; required)
          RUN_ID_OVERRIDE  -> run id stored in "run_id" (no fallback; required)
          RUN_SOURCE       -> optional provenance tag (defaults to "auto-prove")
      * If env is missing, it throws and does NOT execute the tactic nor write logs.
  - `logGoalsAuto`: convenience no-op that logs current goals once (first build).
-/

import Lean
open Lean Meta Elab Tactic IO System

/-- Pretty-print an expression to a string. -/
def ppExprToString (e : Expr) : MetaM String := do
  let fmt ← PrettyPrinter.ppExpr e
  return fmt.pretty

/-- One hypothesis (local declaration) → JSON. -/
def formatLocalDeclJson (ldecl : LocalDecl) : MetaM Json := do
  let ty ← ppExprToString ldecl.type
  return Json.mkObj
    [ ("name",  Json.str ldecl.userName.toString)
    , ("type",  Json.str ty)
    ]

/-- One goal (by `MVarId`) → JSON with `context` and `target`. -/
def goalToJson (g : MVarId) : MetaM Json := do
  g.withContext do
    let decl ← g.getDecl
    let mut ctxArr : Array Json := #[]
    for ldecl in decl.lctx do
      if ¬ ldecl.isImplementationDetail then
        ctxArr := ctxArr.push (← formatLocalDeclJson ldecl)
    let tgt ← ppExprToString decl.type
    return Json.mkObj
      [ ("target",  Json.str tgt)
      , ("context", Json.arr ctxArr)
      ]

/-- Collect *all* current goals as a JSON array. -/
def collectAllGoalsJson : TacticM Json := do
  let gs ← getGoals
  let arr ← gs.mapM (fun g => liftMetaM (goalToJson g))
  return Json.arr arr.toArray

/-- Append `data` to file `path`. (Assumes parent dir exists.) -/
def appendFile (path : System.FilePath) (data : String) : IO Unit := do
  let h ← IO.FS.Handle.mk path IO.FS.Mode.append
  h.putStr data
  h.flush

/-- (proof_id, decl, file, line, col) gathered from the current tactic ref. -/
def currentProofId : TacticM (String × String × String × Nat × Nat) := do
  let declName := (← Elab.Term.getDeclName?).getD Name.anonymous
  let file : String := (← getFileName)
  let fm   ← getFileMap
  let (line, col) ←
    match (← getRef).getPos? with
    | none      => pure (0, 0)
    | some bp   =>
      let lc := fm.toPosition bp
      pure (lc.line, lc.column)
  let declStr := s!"{declName}"
  let proofId := s!"{file}::{declStr}::L{line}"
  pure (proofId, declStr, file, line, col)

/-- Lean & logger versions. -/
def leanVersionJson : Json :=
  Json.mkObj
    [ ("lean", Json.str s!"{Lean.versionString}")
    , ("logger", Json.str "logStep-v2")
    ]

/-- Count how many existing log lines already have this `proof_id` in a given file. -/
def nextStepIdxFromFile (proofId : String) (path : System.FilePath) : TacticM Nat := do
  let fileExists ← path.pathExists
  if !fileExists then
    return 0
  let contents ← IO.FS.readFile path
  let needle := s!"\"proof_id\":\"{proofId}\""
  let parts := contents.splitOn needle
  return (if parts.isEmpty then 0 else parts.length - 1)

/-- Original global mathlib-scraping log path. -/
def mathlibLogPath : System.FilePath := "data/goal_tactic_log.jsonl"

/-- STRICT: read per-run log path from env, else throw. -/
def getAutoLogPath! : IO System.FilePath := do
  match (← IO.getEnv "RUN_JSON_PATH") with
  | some p => pure p
  | none   => throw <| IO.userError "RUN_JSON_PATH not set; refusing to run logStepAuto"

/-- STRICT: read run id from env, else throw. -/
def getRunId! : IO String := do
  match (← IO.getEnv "RUN_ID_OVERRIDE") with
  | some v => pure v
  | none   => throw <| IO.userError "RUN_ID_OVERRIDE not set; refusing to run logStepAuto"

/-- Optional provenance tag. -/
def getRunSource? : IO (Option String) := IO.getEnv "RUN_SOURCE"

/-!
`logStep` (original):
  - logs goals **before**, the **tactic**, and goals **after** (or error),
  - adds file/decl/pos, per-proof `step_idx`, status, timing, versions,
  - writes to `data/goal_tactic_log.jsonl` (unchanged behavior).
-/
elab "logStep" tac:tacticSeq : tactic => do
  let t0 ← IO.monoMsNow
  let goalsBefore ← collectAllGoalsJson
  let tacticStr := toString tac
  let (proofId, decl, file, line, col) ← currentProofId
  let stepIdx ← nextStepIdxFromFile proofId mathlibLogPath

  let mut status := "ok"
  let mut goalsAfter : Json := Json.arr #[]
  let mut errMsg := ""
  try
    evalTactic tac
    goalsAfter ← collectAllGoalsJson
  catch e =>
    status := "error"
    errMsg := (← e.toMessageData.toString)

  let timeMs := (← IO.monoMsNow) - t0

  let jsonObj : Json := Json.mkObj
    [ ("run_id",       Json.str "<fill-at-run-level-if-needed>")  -- unchanged
    , ("proof_id",     Json.str proofId)
    , ("file",         Json.str file)
    , ("decl",         Json.str decl)
    , ("pos",          Json.mkObj [("line", Json.num line), ("col", Json.num col)])
    , ("step_idx",     Json.num stepIdx)
    , ("goals_before", goalsBefore)
    , ("tactic",       Json.str tacticStr)
    , ("status",       Json.str status)
    , ("goals_after",  goalsAfter)
    , ("error",        Json.str errMsg)
    , ("time_ms",      Json.num timeMs)
    , ("versions",     leanVersionJson)
    ]

  let _ ← appendFile mathlibLogPath (Json.compress jsonObj ++ "\n")

/-!
`logStepAuto` (STRICT, gated):
  - same logging shape as `logStep`,
  - **requires** env `RUN_ID_OVERRIDE` and `RUN_JSON_PATH`, otherwise throws,
  - `step_idx` counted within that file,
  - writes optional `"source"` (defaults to "auto-prove").
  - Because env is checked **before** `evalTactic`, nothing runs if the gate fails.
-/
elab "logStepAuto" tac:tacticSeq : tactic => do
  -- Strict gate: fail fast if not launched by auto_prove.py (or a run that set the env vars).
  let runId   ← getRunId!
  let outPath ← getAutoLogPath!
  let source  := (← getRunSource?).getD "auto-prove"

  let t0 ← IO.monoMsNow
  let goalsBefore ← collectAllGoalsJson
  let tacticStr := toString tac
  let (proofId, decl, file, line, col) ← currentProofId
  let stepIdx ← nextStepIdxFromFile proofId outPath

  let mut status := "ok"
  let mut goalsAfter : Json := Json.arr #[]
  let mut errMsg := ""
  try
    evalTactic tac
    goalsAfter ← collectAllGoalsJson
  catch e =>
    status := "error"
    errMsg := (← e.toMessageData.toString)

  let timeMs := (← IO.monoMsNow) - t0

  let jsonObj : Json := Json.mkObj
    [ ("run_id",       Json.str runId)
    , ("source",       Json.str source)
    , ("proof_id",     Json.str proofId)
    , ("file",         Json.str file)
    , ("decl",         Json.str decl)
    , ("pos",          Json.mkObj [("line", Json.num line), ("col", Json.num col)])
    , ("step_idx",     Json.num stepIdx)
    , ("goals_before", goalsBefore)
    , ("tactic",       Json.str tacticStr)
    , ("status",       Json.str status)
    , ("goals_after",  goalsAfter)
    , ("error",        Json.str errMsg)
    , ("time_ms",      Json.num timeMs)
    , ("versions",     leanVersionJson)
    ]

  let _ ← appendFile outPath (Json.compress jsonObj ++ "\n")

/-- Log current goals once without changing the state (useful as a first line). -/
elab "logGoalsAuto" : tactic => do
  evalTactic (← `(tactic| logStepAuto (skip)))
