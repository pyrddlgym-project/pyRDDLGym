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
- PPDDL (step 3): `probabilistic` (in effects and in `:init`), the `(reward)` fluent, `:goal-reward`, `:goal-probability`.
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

Everything else raises `PDDLUnsupportedError`. That includes recursive axioms, metrics other than those listed below, and PPDDL until step 3.

**Names**
- RDDL identifiers must match `[A-Za-z]([A-Za-z0-9_-]*[A-Za-z0-9])?`. Any other character is mangled.
- Names that collide with RDDL reserved words are mangled.
- PDDL keeps predicates, functions, actions and types in separate namespaces, but RDDL shares one namespace for all of them. On a collision we prefix the name.
- Output is deterministic.

**Types**
- Group PDDL's top-level types with union-find. Two groups merge whenever some parameter or `either` type spans both.
- Each group becomes one RDDL `: object` type.
- Each narrower subtype becomes a non-fluent `IS-<type>(?o)` membership predicate. It guards every parameter and quantifier variable whose type is narrower than its group.
- Domain constants are added to every instance and referenced as `@c`. Check that pyRDDLGym accepts `@obj` for object types; if it doesn't, use a singleton non-fluent per constant.

**Fluents**
- Predicates or functions that some effect modifies, or that a probabilistic `:init` sets in any grounding, become `state-fluent`. All others become `non-fluent`. A state-fluent that is never modified gets the identity CPF `p'(x̄) = p(x̄)`. The reason: RDDL init-state entries can be random expressions, but the non-fluents block accepts only constants.
- Derived predicates become `derived-fluent bool`. A cycle in their dependency graph (found with strongly connected components) raises.
- Numeric functions the problem only partly initializes get a non-fluent `DEFINED-f(x̄)` guard on reads. Assigning to an undefined fluent raises.

**Actions**
- Each schema becomes a `bool` action-fluent.
- Preconditions become `forall_{?ȳ}[ α(?ȳ) => guards ^ pre ]`.
- `max-nondef-actions = 1`.

**Effect update rules (core)**
For each fluent `p(x̄)`, every effect occurrence of `p(t̄)` inside some action `α(ȳ)`, under some `forall z̄` and `when c`, becomes a trigger:
`exists_{free vars}[ α(ȳ) ^ c ^ guards ^ ⋀ x_i == t_i ]`.
An argument that is a variable appearing only once is substituted directly instead of being matched with `==`.
- Boolean fluents: `p'(x̄) = OR(add) | (p(x̄) ^ ~OR(del))`. This keeps PDDL's rule that adds win over deletes. Every condition is read from the current state, as PDDL requires.
- Numeric fluents: `f'(x̄) = if (assign) then v else f(x̄) + Σinc − Σdec`, with scale effects as multiplicative factors. When two effects of one action conflict, the order of the `if` chain decides and is documented.

**Goal, reward and horizon**
- `termination { goal }`.
- `reward = −Σ_α cost_α·α − noop_cost·[no action]`.
  - `cost_α` is the action's increment to `total-cost`, or 1 if the domain has no costs.
  - Allowed metrics: `minimize (total-cost)`, or a linear metric over fluents that change only through `increase`/`decrease`. Anything else raises.
- Taking no action is allowed. It doesn't change which goals can be reached, and it avoids having no legal action in dead-end states.
- `horizon` is required. `discount = 1.0`. RDDL also accepts `horizon = pos-inf;`, but it isn't offered as a default: with dead ends and discount 1, an episode would never end.

**`RDDLBuilder` fixes (done, 2026-11-28)**
- `object_types` is now an insertion-ordered list, so the output order is deterministic.
- `build_instance` raises `ValueError` naming any object type that has no objects.
- `build_instance` leaves out `max-nondef-actions` when it is unset, and pyRDDLGym then defaults to `pos-inf`. It raises `ValueError` when `horizon` or `discount` is unset, since pyRDDLGym has no default for either.

**Exit criteria (step 2)**
1. The emitted RDDL parses and compiles in pyRDDLGym, in both vectorized and non-vectorized modes, for every supported benchmark.
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
- Each `(probabilistic p1 e1 … pk ek)` becomes an `interm-fluent` `OUT-<α>-<n>(ȳ, z̄)`. Its value is an enum `{@o0 … @ok}`, where `@o0` is the remainder with no effect, and it is sampled with `Discrete(...)`. Every trigger inside branch i also requires `OUT-… == @oi`.
- Nested blocks each get their own outcome fluent.
- A block that affects a single literal compiles to an inline `Bernoulli`.
- Rewards: `(increase (reward) e)` becomes a term in the reward; `:goal-reward` and `:goal-probability` become a goal bonus. Goals still map to `termination`.

**Random initial states (`(probabilistic ...)` in `:init`)**

The parser stores the deterministic part in `Problem.init_atoms` and `init_numeric`, and the blocks in `Problem.init_effects`. Semantics follow mdpsim: all blocks are applied to the deterministic part as one simultaneous effect, deletes first and then adds, so an add wins and a branch may delete a deterministic atom. Each block is sampled once per reset, independently of the other blocks.

The parser already rejects these cases:
- a probability that isn't a number;
- probabilities that sum to more than 1;
- two different values for one numeric fluent that can occur in the same outcome;
- a numeric fluent that is set only in some outcomes and has no deterministic value (`PDDLUnsupportedError`).

Translation into RDDL `init-state` expressions:
- **Outcome fluents.** Each block `n` with `k` branches becomes a state-fluent `INIT-OUT-n : { state-fluent, init-out-n, default = @o0 }`. Its enum is `{@o0 … @ok}`, where `@o0` is the remainder with no effect. Its CPF is the identity, and its init-state entry is `INIT-OUT-n = Discrete(init-out-n, @o0 : 1-Σp, @o1 : p1, …)`.
  - A block whose only effect is a single literal in a single branch skips the fluent and inlines `Bernoulli(p)` (or `~Bernoulli(p)` for a delete).
  - Nested blocks get their own outcome fluent. A nested block's condition is ANDed with the condition of the branch it sits in. Sampling every block, even inside a branch that wasn't chosen, gives the same distribution.
- **Conditions.** An effect in branch `i` of block `n` fires when `cond = (INIT-OUT-n == @oi)`, ANDed with the conditions of the enclosing branches.
- **Atoms.** An atom becomes `p(ō) = OR(add conds) | (det(ō) ^ ~OR(del conds))`, where `det(ō)` is whether the atom is in `init_atoms`. Simplify when `det` is constant, and emit an entry only when it differs from the default.
- **Numeric fluents.** `f(ō) = if (c1) then v1 else if (c2) then v2 … else det(ō)`. The parser guarantees that conditions with different values never both fire, so the order of the chain doesn't matter. When the fluent has no deterministic value, the parser guarantees the conditions cover every outcome, and the last condition becomes the `else`.
- **Referencing other fluents.** pyRDDLGym resolves init-state entries that refer to other state fluents in dependency order, per grounding, so the order of the entries doesn't matter.
- **Visibility.** The outcome fluents are part of the observed state. They don't change the dynamics, because no CPF other than their own identity reads them.

**Exit criteria (step 3)**
1. The emitted RDDL compiles for IPPC 2004/2006/2008 PPDDL domains.
2. On tiny hand-written domains, sampled outcome frequencies from `RDDLEnv` match the probabilities written in the domain, using a seeded chi-square test. One case must change several fluents together, so correlated effects are checked.
3. The same seeded test runs over `RDDLEnv.reset()` for random initial states. It must cover:
   - a block that sets several atoms together, checking the joint frequencies;
   - a nested block;
   - a branch that deletes a deterministic atom;
   - an atom set by two blocks;
   - numeric fluents.

   Run it in both vectorized and non-vectorized modes.
4. Plan replay (step 2) runs only on problems with a deterministic `:init`. For a problem with a random `:init`, replay the plan only after a seeded reset whose sampled initial state matches the state the plan assumes.
5. Rejection tests for PPDDL constructs the translator doesn't support.

---

## Ancillary
- `pyproject.toml`: `[project.optional-dependencies] test = ["pytest"]`.
- Docs: one short page with a supported/unsupported table, the meaning of goal, reward and no-op, and a `PDDLToRDDLTranslator.translate(...)` example.
