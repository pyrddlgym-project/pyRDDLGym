# Plan: PDDL → RDDL converter in pyRDDLGym

## Context
pyRDDLGym should accept PDDL (and later PPDDL) by converting it automatically into lifted RDDL. If a domain uses a construct that has no faithful RDDL equivalent, the converter raises an error naming that construct and where it appears.

This is an **RDDL package feature, not a PDDL toolkit**. The deliverable is the minimum needed for PDDL text → RDDL text. Simulation, PDDL pretty-printing, plan tools, a CLI, environment helpers and action encoding/decoding are all **out of scope**. RDDL→PDDL is also out of scope.

No existing paper or tool does PDDL→RDDL. The building blocks are standard: Reiter-style per-fluent update rules for effects, and intermediate fluents for correlated probabilistic effects (from Sanner's RDDL spec and tutorial).

Decisions already made:
- The converter lives inside pyRDDLGym.
- It uses a custom parser with no external dependencies.
- Order of work: parser → v1 (deterministic) → PPDDL.

## Deliverable (new subpackage `pyRDDLGym/pddl/`, 4 files)
```
pyRDDLGym/pddl/
  __init__.py     # convert(domain, problem, *, horizon, noop_cost=1.0) -> (domain_rddl, instance_rddl)
  errors.py       # PDDLParseError, PDDLUnsupportedError (both carry line/col)
  parser.py       # lexer + s-expression reader + IR dataclasses + symbol checks (stdlib only)
  compiler.py     # IR -> RDDL via RDDLBuilder (naming, types, effects, reward, termination)
```
`domain` and `problem` may each be a file path or PDDL text. No new runtime dependencies.

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

**What gets built into the IR**
Only the constructs the converter actually translates:
- Domain: `:requirements`, `:types` (multi-level, `either`), `:constants`, `:predicates`, `:functions`, `:derived`, `:action`.
- Problem: `:objects`, `:init`, `:goal`, `:metric`.
- Formulas: `and/or/not/imply/exists/forall/when`, `=`, comparisons, arithmetic, `increase/decrease/assign/scale-up/scale-down`.
- PPDDL (step 3): `probabilistic`, the `(reward)` fluent, `:goal-reward`, `:goal-probability`.

**What is recognized and rejected**
Constructs the converter can't translate are detected by keyword and raise `PDDLUnsupportedError(construct, line, col)` immediately, without building a representation for them:
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
3. Every file in a small benchmark corpus in `tests/pddl/benchmarks/` either parses or raises `PDDLUnsupportedError`, never anything else. The corpus is about 20 IPC classical and numeric domains plus IPPC PPDDL domains, with the source and license of each noted.

---

## Step 2 — v1 converter (`compiler.py`, deterministic)

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
- Predicates or functions that some effect modifies become `state-fluent`. All others become `non-fluent`.
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
- `horizon` is required. `discount = 1.0`.

**`RDDLBuilder` fixes (these are bugs in their own right)**
- `object_types` is a `set`, so the output order is nondeterministic. Make it insertion-ordered.
- `build_instance` raises `KeyError` for an object type with no objects. Decide the behaviour for empty types and raise a clear error.
- `build_instance` prints `None` when `maxnondef`, `horizon` or `discount` were never set. Validate them, or leave them out.

**Exit criteria (step 2)**
1. The emitted RDDL parses and compiles in pyRDDLGym, in both vectorized and non-vectorized modes, for every supported benchmark.
2. **Plan replay:** plans for the benchmark problems are produced once with Fast Downward, validated with VAL, and checked in as plain text fixtures. Neither tool is a dependency. Each plan, run in `RDDLEnv`:
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

**Exit criteria (step 3)**
1. The emitted RDDL compiles for IPPC 2004/2006/2008 PPDDL domains.
2. On tiny hand-written domains, sampled outcome frequencies from `RDDLEnv` match the probabilities written in the domain, using a seeded chi-square test. One case must change several fluents together, so correlated effects are checked.
3. Rejection tests for PPDDL constructs the converter doesn't support.

---

## Ancillary
- `pyproject.toml`: `[project.optional-dependencies] test = ["pytest"]`.
- Docs: one short page with a supported/unsupported table, the meaning of goal, reward and no-op, and a `convert(...)` example.
