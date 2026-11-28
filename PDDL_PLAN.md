# Plan: PDDL → RDDL translator in pyRDDLGym

## Context
pyRDDLGym should accept PDDL (and later PPDDL) by converting it automatically into lifted RDDL. If a domain uses a construct that has no faithful RDDL equivalent, the translator raises an error naming that construct and where it appears.

This is an **RDDL package feature, not a PDDL toolkit**. The deliverable is the minimum needed for PDDL text → RDDL text. Simulation, PDDL pretty-printing, plan tools, a CLI, environment helpers and action encoding/decoding are all **out of scope**. RDDL→PDDL is also out of scope.

No existing paper or tool does PDDL→RDDL. The building blocks are standard: Reiter-style per-fluent update rules for effects, and intermediate fluents for correlated probabilistic effects (from Sanner's RDDL spec and tutorial).

Decisions already made:
- The translator lives inside pyRDDLGym.
- It uses a custom parser with no external dependencies.
- Order of work: parser → v1 (deterministic) → PPDDL.

## Deliverable (new subpackage `pyRDDLGym/pddl/`, 4 files)
```
pyRDDLGym/pddl/
  __init__.py     # re-exports PDDLParser, PDDLToRDDLTranslator and the errors
  errors.py       # PDDLParseError, PDDLUnsupportedError (both carry line/col)
  parser.py       # PDDLParser: lexer + s-expression reader + IR dataclasses + symbol checks (stdlib only)
  translator.py   # PDDLToRDDLTranslator: IR -> RDDL via RDDLBuilder (naming, types, effects, reward, termination)
```
The entry point is the static method `PDDLToRDDLTranslator.translate(domain, problem, *, horizon, noop_cost=1.0) -> (domain_rddl, instance_rddl)`. `domain` and `problem` may each be a file path or PDDL text. There are no module-level functions: helpers are static methods of the class that uses them. No new runtime dependencies.

Reuse:
- `RDDLBuilder` (`pyRDDLGym/core/builder.py`) to emit the RDDL.
- `RDDLlex.reserved` (`pyRDDLGym/core/parser/parser.py`) as the list of reserved words to avoid.

---

## Step 1 — Parser (`parser.py`, stdlib only)

**Lexer and reader**
- `;` starts a comment.
- Everything is lowercased, since PDDL is case-insensitive.
- Tokens: `(`, `)`, names, `?vars`, numbers, `-`.
- Every node keeps its line and column. Errors look like `line:col: expected X, got Y`.
- A file may hold a domain and a problem together, as IPPC files do. Each entry point picks its own `define`, and two definitions of the same kind are an error.
- A 0-ary function can be written without parentheses (PDDL 2.1 allows `<f-head> ::= <function-symbol>`). A type glued to its dash, as in `?x -type`, is read as `- type`.

**What gets built into the IR**
Only the constructs the translator actually translates:
- Domain: `:requirements`, `:types` (multi-level, `either`), `:constants`, `:predicates`, `:functions`, `:derived`, `:action`.
- Problem: `:objects`, `:init`, `:goal`, `:metric`.
- Formulas: `and/or/not/imply/exists/forall/when`, `=`, comparisons, arithmetic, `increase/decrease/assign/scale-up/scale-down`.
- PPDDL (step 3): `probabilistic` (in effects and in `:init`), the `(reward)` fluent, `:goal-reward`, and `(goal-probability)` as the whole metric.
- Entry points: `PDDLParser.parse_domain(text)` and `PDDLParser.parse_problem(text, domain)`. The parser has no module-level functions.

**What is recognized and rejected**
Constructs the translator can't translate are detected by keyword and raise `PDDLUnsupportedError(construct, line, col)` immediately, without building a representation for them:
- `:durative-action`, `:process`, `:event`
- `:constraints`, `preference`
- timed initial literals
- object-valued functions

Any other unknown form raises `PDDLParseError`.

**Symbol checks (done while parsing)**
These raise `PDDLParseError`, because otherwise pyRDDLGym would later fail with a confusing error:
- undeclared predicates, functions, types, objects or variables
- wrong arity
- argument type mismatches (subtype-aware)
- duplicate declarations

**Exit criteria (step 1)**
1. Unit tests check the IR built for small inputs of each construct.
2. Negative tests: each malformed or unsupported input asserts the error class and the line number.
3. Every file in a small benchmark corpus either parses or raises `PDDLUnsupportedError`, never anything else. The corpus is about 20 IPC classical and numeric domains plus IPPC PPDDL domains.
   - **Local only, never committed.** It lives in `tests/pddl/benchmarks/`, which is listed in `.gitignore`, so it is never redistributed. A local `SOURCES.md` there records where each file came from.
   - Tests that use the corpus skip when the directory is missing. Run them locally before each commit. CI runs only the committed hand-written tests.

---

## Step 2 — v1 translator (`translator.py`, deterministic)

**Supported**
`strips, typing, negative/disjunctive/existential/universal preconditions, equality, conditional-effects, adl, derived-predicates (non-recursive), numeric-fluents, action-costs`.

Everything else raises `PDDLUnsupportedError`. That includes recursive axioms and metrics other than those listed below. PPDDL is step 3.

**Names**
- RDDL identifiers must match `[A-Za-z]([A-Za-z0-9_-]*[A-Za-z0-9])?`. Any other character is mangled. `__` is collapsed too, because pyRDDLGym uses it to separate objects in grounded names.
- PDDL keeps predicates, functions, actions and types in separate namespaces, but RDDL shares one namespace for all of them. A name that collides with an RDDL reserved word or an earlier name gets its kind as a prefix (`action-move`), then a number.
- Generated names contain upper-case letters (`IS-t`, `OBJ-o`, `DEFINED-f`, `Object`, `NONE`). The parser lower-cases PDDL, so they can't collide with PDDL names.
- Output is deterministic.

**Types**
- Group PDDL's top-level types with union-find. Two groups merge whenever some parameter or `either` type spans both.
- Each group becomes one RDDL `: object` type.
- Each narrower subtype becomes a non-fluent `IS-<type>(?o)` membership predicate. It guards every parameter and quantifier variable whose type is narrower than its group.
- A parameter typed `object` merges every group into one RDDL type, `Object`. That is also the case for untyped domains.
- pyRDDLGym allows only enum literals in domain expressions, so `@c` is not available for object types. Each object referenced in an expression (domain constants and goal objects) gets a non-fluent `OBJ-o(?x)` that is true only for `o`. A read of `p(?x, o)` becomes `(exists_{?k : T}[OBJ-o(?k) ^ p(?x, ?k)])`, or a `sum` for numbers.
- RDDL has no empty types, so a group with no objects gets the object `NONE`, and every type in that group is guarded, which excludes it.
- RDDL cannot compare objects of different types with `==`, so an equality `(= x y)` merges the groups of the two types it compares.
- **Untyped domains.** Without types, everything would share one RDDL type, and an action with k parameters would have |objects|^k groundings (zenotravel took over a minute to load). So when a domain declares no types, the IR is first rewritten as if it were typed:
  - An object's type is its signature: the set of static unary predicates (never changed by an effect or a random `:init`, not derived) that hold for it in `:init`. Objects with none get the type `untyped`.
  - An action parameter may have the signatures that satisfy every static unary atom that the precondition requires of it as a top-level conjunct. With no such atom, it may have any signature.
  - Each predicate and function position gets every signature that some argument there can have: from actions, derived rules (whose parameters take their predicate's positions, solved by a fixpoint), `:init`, the goal and the metric. So every atom that can hold fits its declaration.
  - Quantified variables may have any signature. That is sound, but it merges every type that reaches them.
  - The usual grouping then applies, and the precondition's unary atoms become redundant with the `IS-t` guards. zenotravel's `zoom` goes from 14⁶ groundings to 4·3·3·7³.

**Fluents**
- Predicates or functions that some effect modifies, or that a probabilistic `:init` sets in any grounding, become `state-fluent`. All others become `non-fluent`. A state-fluent that is never modified gets the identity CPF `p'(x̄) = p(x̄)`. The reason: RDDL init-state entries can be random expressions, but the non-fluents block accepts only constants.
- Derived predicates become state-fluents, not `derived-fluent`. pyRDDLGym deprecates `derived-fluent`, and its `VALID_DEPENDENCIES` forbids derived and interm fluents in preconditions and termination. Each derived predicate's CPF evaluates its rules on the next state (`p'(x̄) = body` with the state-fluents in `body` primed). A lifted init-state entry `p(x̄) = body` evaluates them on the initial state. So `p` always equals its rules on the current state. Recursive rules raise, and so do effects on derived predicates and derived atoms in `:init`.
- Numeric functions the problem only partly initializes get a non-fluent `DEFINED-f(x̄)`. It guards every comparison that reads `f`. For an effect that reads or changes `f`, the precondition requires `DEFINED` wherever the effect fires, because PDDL cannot apply such an effect to an undefined value. Definedness stays fixed only if `f` is never assigned, so a partly initialized function that some effect assigns raises.
- Functions in the metric that nothing reads, such as `total-cost`, are not declared as fluents; they become reward terms.

**Actions**
- Each schema becomes a `bool` action-fluent.
- Preconditions become `forall_{?ȳ}[ α(?ȳ) => guards ^ pre ]`.
- `max-nondef-actions = 1`.

**Effect update rules (core)**
For each fluent `p(x̄)`, every effect occurrence of `p(t̄)` inside some action `α(ȳ)`, under some `forall z̄` and `when c`, becomes a trigger:
`exists_{free vars}[ α(ȳ) ^ c ^ guards ^ ⋀ x_i == t_i ]`.
An argument that is a variable appearing only once is substituted directly instead of being matched with `==`.
- Boolean fluents: `p'(x̄) = OR(add) | (p(x̄) ^ ~OR(del))`. This keeps PDDL's rule that adds win over deletes. Every condition is read from the current state, as PDDL requires.
- Numeric fluents: `f'(x̄) = if (assign) then v else f(x̄) × Πscale-up ÷ Πscale-down + Σinc − Σdec`. Increases under `forall` add up over every grounding that fires. An assign wins over every other effect on the same fluent; PDDL forbids that combination anyway. If several assigns of one grounding fire in the same step, their values are summed (marked `ponytail:` in the code); PDDL calls such plans invalid.
- RDDL aggregations extend as far right as possible: `forall_{?x : t}[a] | b` means `forall_{?x : t}([a] | b)`. So every emitted `exists`, `forall`, `sum` and `prod` is parenthesized.

**Goal, reward and horizon**
- `termination { goal }`.
- `reward = −Σ_α cost_α·α − noop_cost·[no action]`.
  - Without a metric, each action costs 1.
  - With `(:metric minimize M)`, `M` must be a linear combination, with constant coefficients, of 0-ary functions that nothing reads and that change only through `increase`/`decrease`. A step's reward is minus the change in `M`, so the total reward is minus the final value of `M` (plus its initial value). `maximize` flips the sign. Anything else raises.
- The entry point is `PDDLToRDDLTranslator.translate(domain, problem, *, horizon, noop_cost=1.0)`. Each argument may be a file path or PDDL text, and passing the same text twice handles combined files.
- Taking no action is allowed. It doesn't change which goals can be reached, and it avoids having no legal action in dead-end states.
- `horizon` is required. `discount = 1.0`. RDDL also accepts `horizon = pos-inf;`, but it isn't offered as a default: with dead ends and discount 1, an episode would never end.

**`RDDLBuilder` fixes (done, 2026-11-28)**
- `object_types` is now an insertion-ordered list, so the output order is deterministic.
- `build_instance` raises `ValueError` naming any object type that has no objects.
- `build_instance` leaves out `max-nondef-actions` when it is unset, and pyRDDLGym then defaults to `pos-inf`. It raises `ValueError` when `horizon` or `discount` is unset, since pyRDDLGym has no default for either.

**Exit criteria (step 2)** — status 2026-11-28: criteria 1, 4 and 5 are met. Criteria 2 and 3 are deferred: no Linux machine for Fast Downward and VAL, and WSL, Docker or pyperplan (STRIPS-only) are the options.
1. The emitted RDDL parses and compiles in pyRDDLGym, in both vectorized and non-vectorized modes, for every supported benchmark. This is `tests/pddl/test_benchmarks.py`: all 48 supported corpus cases load in both modes, and a no-op step keeps the state and costs `noop_cost`.
2. **Plan replay:** plans for the benchmark problems are produced once with Fast Downward, validated with VAL, and stored as plain text next to their problems in the local corpus, so they are not committed either. Neither tool is a dependency. Each plan, run in `RDDLEnv`:
   - has every step pass the action preconditions
   - terminates exactly on the last step
   - gives total reward = −plan cost
3. **Negative steps:** for each replayed plan, a mutated step (wrong arguments, or an action whose preconditions fail) must violate the RDDL preconditions.
4. **Hand-written transition tests** on tiny domains cover each tricky rule against an expected next state:
   - add-wins-over-delete
   - conditional and `forall` effects
   - repeated variables and constants in effects
   - subtype guards
   - numeric effects
   - derived predicates
5. **Rejection:** for each unsupported construct, a small domain raises `PDDLUnsupportedError` at the correct location.

---

## Step 3 — PPDDL
- One enum type `Outcome : {@O0 … @OK}` serves every block, where K is the largest number of branches. `@O0` is the remainder with no effect.
- Each `(probabilistic p1 e1 … pk ek)` in an action becomes an `interm-fluent` `OUT-<α>-<n>(ȳ, z̄)`: one sample per grounding of the action parameters and the enclosing `forall` variables. It is sampled with `Discrete(Outcome, @O0 : 1−Σp, @O1 : p1, …)`, with 0 for unused values, and every trigger inside branch i also requires `OUT-… == @Oi`. Probabilities may be expressions over the current state; one that reads a partly initialized function raises.
- A block with a single branch uses a `bool` interm-fluent sampled with `Bernoulli(p)` instead.
- Nested blocks each get their own outcome fluent, and their triggers carry every enclosing outcome test. `when` conditions are not part of sampling: every block is sampled every step, which gives the same distribution.
- Preconditions cannot read interm-fluents, so the `DEFINED-f` precondition of a numeric effect inside a branch applies whatever the outcome.
- Rewards: `(reward)` is an ordinary metric function, so `(:metric maximize (reward))` turns each `(increase (reward) e)` into a reward term through the step 2 rules. `(:goal-reward n)` adds `n·coef(reward)·goal'` (the goal on the next state), and `(:metric maximize (goal-probability))` adds `1·goal'`. Goals still map to `termination`, so the bonus is paid once. If the initial state already satisfies the goal, the episode still takes one step.

**Random initial states (`(probabilistic ...)` in `:init`)**

The parser stores the deterministic part in `Problem.init_atoms` and `init_numeric`, and the blocks in `Problem.init_effects`. Semantics follow mdpsim: all blocks are applied to the deterministic part as one simultaneous effect, deletes first and then adds, so an add wins and a branch may delete a deterministic atom. Each block is sampled once per reset, independently of the other blocks.

The parser already rejects these cases:
- a probability that isn't a number;
- probabilities that sum to more than 1;
- two different values for one numeric fluent that can occur in the same outcome;
- a numeric fluent that is set only in some outcomes and has no deterministic value (`PDDLUnsupportedError`).

Translation into RDDL `init-state` expressions:
- **Outcome fluents.** Each block `n` becomes a 0-ary state-fluent `INIT-OUT-n : { state-fluent, Outcome, default = @O0 }`, numbered in source order with nested blocks after their parent. Its CPF is the identity, and its init-state entry is `INIT-OUT-n = Discrete(Outcome, @O0 : 1-Σp, @O1 : p1, …)`.
  - A single-branch block uses a `bool` fluent with `Bernoulli(p)` instead.
  - Nested blocks get their own outcome fluent. Sampling every block, even inside a branch that wasn't chosen, gives the same distribution.
- **Conditions.** An effect in branch `i` of block `n` fires when `cond = (INIT-OUT-n == @Oi)`, ANDed with the conditions of the enclosing branches.
- **Atoms.** An atom becomes `p(ō) = OR(add conds) | (det(ō) ^ ~OR(del conds))`, where `det(ō)` is whether the atom is in `init_atoms`. `det` is a constant, so this is `OR(adds)` or `OR(adds) | ~OR(dels)`, and an atom whose value can't change gets no entry.
- **Numeric fluents.** `f(ō) = if (c1) then v1 else if (c2) then v2 … else det(ō)`. The parser guarantees that conditions with different values never both fire, so the order of the chain doesn't matter. When the fluent has no deterministic value, the parser guarantees the conditions cover every outcome, and the last condition becomes the `else`.
- **Referencing other fluents.** pyRDDLGym resolves init-state entries that refer to other state fluents in dependency order, per grounding, so the order of the entries doesn't matter. This includes the lifted entries of derived predicates, which read the sampled atoms.
- **Visibility.** The outcome fluents are part of the observed state. They don't change the dynamics, because no CPF other than their own identity reads them.

**Exit criteria (step 3)** — status 2026-11-28: criteria 1, 2, 3 and 5 are met; 4 is deferred with step 2's plan replay.
1. The emitted RDDL compiles for IPPC 2004/2006/2008 PPDDL domains. The corpus has IPPC 2008 and the Little–Thiébaux domains: all 24 PPDDL cases load in both modes (rectangle-tireworld stays malformed).
2. On tiny hand-written domains, sampled outcome frequencies from `RDDLEnv` match the probabilities written in the domain, using a seeded chi-square test. One case must change several fluents together, so correlated effects are checked.
3. The same seeded test runs over `RDDLEnv.reset()` for random initial states. It must cover:
   - a block that sets several atoms together, checking the joint frequencies;
   - a nested block;
   - a branch that deletes a deterministic atom;
   - an atom set by two blocks;
   - numeric fluents.

   Run it in both vectorized and non-vectorized modes.
4. Plan replay (step 2) runs only on problems with a deterministic `:init`. For a problem with a random `:init`, replay the plan only after a seeded reset whose sampled initial state matches the state the plan assumes.
5. Rejection tests for PPDDL constructs the translator doesn't support: a probability that reads a partly initialized function, and the step 2 metric rules (assign on `(reward)`, `(reward)` read by the domain).

---

## Ancillary
- `pyproject.toml`: `[project.optional-dependencies] test = ["pytest"]`.
- Docs: one short page with a supported/unsupported table, the meaning of goal, reward and no-op, and a `PDDLToRDDLTranslator.translate(...)` example.
