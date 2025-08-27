-- no module declaration needed in Lean 4
import Lean
open Lean Meta Elab Tactic IO

-- Pretty-print an expression
def ppExprToString (e : Expr) : MetaM String := do
  let fmt ← PrettyPrinter.ppExpr e
  return fmt.pretty

-- Format one hypothesis from the context
def formatLocalDecl (l : LocalDecl) : MetaM String := do
  let tyStr ← ppExprToString l.type
  return s!"{l.userName} : {tyStr}"

-- Extract the context and goal target
def collectGoalInfo : TacticM String := do
  let goals ← getGoals
  match goals with
  | [] => return "No goals"
  | g :: _ =>
    let mvarDecl ← g.getDecl
    let lctx := mvarDecl.lctx
    let mut contextStrs := #[]
    for ldecl in lctx do
      if ¬ldecl.isImplementationDetail then
        let s ← formatLocalDecl ldecl
        contextStrs := contextStrs.push s
    let targetStr ← ppExprToString mvarDecl.type
    let header := "Context:\n" ++ String.intercalate "\n" contextStrs.toList
    let footer := "\n\nGoal:\n" ++ targetStr
    return header ++ footer

-- Custom tactic that writes to file
elab "dumpGoalToFile" : tactic => do
  let out ← collectGoalInfo
  let file := "goal_output.txt"
  IO.FS.writeFile file out
  logInfo m!"Goal written to {file}"

/-- Append `data` to a file `path`, creating it if necessary. -/
def appendFile (path : System.FilePath) (data : String) : IO Unit := do
  let h ← IO.FS.Handle.mk path IO.FS.Mode.append
  h.putStr data
  h.flush

/-- Tactic wrapper that logs the goal + tactic to file, then applies the tactic -/
elab "logStep" tac:tacticSeq : tactic => do
  -- 1. Goal string
  let goalStr ← collectGoalInfo

  -- 2. Tactic string
  let tacticStr := toString tac

  -- 3. Format as real JSON
  let jsonObj : Json := Json.mkObj [
    ("goal", Json.str goalStr),
    ("tactic", Json.str tacticStr)
  ]
  let log := Json.compress jsonObj ++ "\n"

  -- 4. Write
  appendFile "goal_tactic_log.jsonl" log

  -- 5. Apply tactic
  evalTactic tac
