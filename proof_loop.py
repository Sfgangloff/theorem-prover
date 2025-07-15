from lean_dojo import LeanGitRepo, trace, Theorem
from lean_dojo.interaction.dojo import Dojo, ProofFinished, ProofGivenUp, LeanError
from infer import predict  # your trained model


# 1️⃣ Initialize and trace
repo = LeanGitRepo("https://github.com/Sfgangloff/theorem-prover", "21e8cf86c103ae68cbe74aa68b7220ffe6055225")
trace(repo, dst_dir="traced_repo")

# 2️⃣ Load your target theorem
thm = Theorem.from_traced_repo("theorems_to_prove/Test_theorem.lean", "example", repo)

# 3️⃣ Start Dojo (Lean gym)
dojo = Dojo(thm, timeout=300)

# 4️⃣ Proof search loop
state = dojo.run_cmd(None, "")  # initialize
while True:
    if isinstance(state, ProofFinished):
        print("✅ Proof complete!")
        break

    if isinstance(state, ProofGivenUp):
        print("⚠️ Given up:", state)
        break

    if isinstance(state, LeanError):
        print("❌ Lean error:", state.error)
        break

    # Otherwise we have a `TacticState`
    goal = state.pp  # pretty-printed goal + context
    print("📌 Current goal:\n", goal)

    tac = predict(goal)
    print("💡 Predicted tactic:", tac)

    try:
        state = dojo.run_tac(state, tac)
    except LeanError as e:
        print("⚠️ Tactic failed:", e.error)
        break

print("Finished with state:", state)