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
