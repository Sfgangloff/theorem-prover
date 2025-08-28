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

/-- Append `data` to file `path`. -/
def appendFile (path : System.FilePath) (data : String) : IO Unit := do
  let h ← IO.FS.Handle.mk path IO.FS.Mode.append
  h.putStr data
  h.flush


/-- Global counter: `proof_id ↦ next step index`. -/
initialize stepCounters : IO.Ref (Std.HashMap String Nat) ← IO.mkRef {}


/-- Count how many existing log lines already have this `proof_id`. -/
def nextStepIdxFromLog (proofId : String) : TacticM Nat := do
  let logPath : System.FilePath := "goal_tactic_log.jsonl"
  let fileExists ← logPath.pathExists
  if !fileExists then
    return 0
  let contents ← IO.FS.readFile logPath
  let needle := s!"\"proof_id\":\"{proofId}\""
  -- Count occurrences of `needle` (works across versions without substring helpers)
  let parts := contents.splitOn needle
  let n := if parts.isEmpty then 0 else parts.length - 1
  return n

/-- (proof_id, decl, file, line, col) gathered from the current tactic ref. -/
def currentProofId : TacticM (String × String × String × Nat × Nat) := do
  -- declaration name if we’re inside one
  let declName := (← Elab.Term.getDeclName?).getD Name.anonymous
  -- file & position from the current ref
  let file : String := (← getFileName)
  let fm   ← getFileMap
  let (line, col) ←
    match (← getRef).getPos? with
    | none      => pure (0, 0)
    | some bp   =>
      let lc := fm.toPosition bp
      pure (lc.line, lc.column)
  let declStr := s!"{declName}"   -- use ToString instance for Name
  let proofId := s!"{file}::{declStr}::L{line}"
  pure (proofId, declStr, file, line, col)

/-- Lean & logger versions. -/
def leanVersionJson : Json :=
  Json.mkObj
    [ ("lean", Json.str s!"{Lean.versionString}")
    , ("logger", Json.str "logStep-v2")
    ]
/--
`logStep`:
  - logs goals **before**, the **tactic**, and goals **after** (or error),
  - adds file/decl/pos, per-proof `step_idx`, status, timing, versions.
-/
elab "logStep" tac:tacticSeq : tactic => do
  let t0 ← IO.monoMsNow
  let goalsBefore ← collectAllGoalsJson
  let tacticStr := toString tac
  let (proofId, decl, file, line, col) ← currentProofId

  -- step index (works in TacticM)
  let stepIdx ← nextStepIdxFromLog proofId

  -- Try to run tactic, capture after or error
  let mut status := "ok"
  let mut goalsAfter : Json := Json.arr #[]
  let mut errMsg := ""

  try
    evalTactic tac
    goalsAfter ← collectAllGoalsJson
  catch e =>
    status := "error"
    errMsg := (← e.toMessageData.toString)

  let t1 ← IO.monoMsNow
  let timeMs := t1 - t0

  let jsonObj : Json := Json.mkObj
    [ ("run_id",       Json.str "<fill-at-run-level-if-needed>")
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

  let _ ← appendFile "goal_tactic_log.jsonl" (Json.compress jsonObj ++ "\n")
