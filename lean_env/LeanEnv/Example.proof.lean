open Classical

theorem ex_and_comm (p q : Prop) : p ∧ q ↔ q ∧ p := by
  constructor
  intro h
  constructor
  rcases h with ⟨hp, hq⟩
  exact hq
  rcases h with ⟨hp, hq⟩
  exact hp
  intro h
  constructor
  rcases h with ⟨hp, hq⟩
  exact hq
  rcases h with ⟨hp, hq⟩
  exact hp
