import Lean
open Lean Meta Elab Tactic IO

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

/-- One goal (by `MVarId`) → JSON with `context` (array) and `target` (string). -/
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

/-- Append `data` to file `path` (create if missing). -/
def appendFile (path : System.FilePath) (data : String) : IO Unit := do
  let h ← IO.FS.Handle.mk path IO.FS.Mode.append
  h.putStr data
  h.flush

/--
`logStep`:
  * captures **all current goals (before)**,
  * records the tactic source,
  * executes the tactic,
  * captures **all current goals (after)**,
  * appends one JSON line to `goal_tactic_log.jsonl`.
-/
elab "logStep" tac:tacticSeq : tactic => do
  -- 1) all goals before
  let goalsBefore ← collectAllGoalsJson

  -- 2) tactic as string
  let tacticStr := toString tac

  -- 3) run the tactic
  evalTactic tac

  -- 4) all goals after
  let goalsAfter ← collectAllGoalsJson

  -- 5) assemble a JSON object
  let jsonObj : Json := Json.mkObj
    [ ("goals_before", goalsBefore)
    , ("tactic",       Json.str tacticStr)
    , ("goals_after",  goalsAfter)
    ]

  -- 6) append as one JSON line
  appendFile "goal_tactic_log.jsonl" (Json.compress jsonObj ++ "\n")
