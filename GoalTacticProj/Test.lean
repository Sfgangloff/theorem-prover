import GoalTacticProj.GoalDumper

example (n m : Nat) (h : n = m) : m + 0 = n := by
  dumpGoalToFile
  rw [h]
  rw [Nat.add_zero]
