# Count-domain consistency check

Inspection after the spatial audit identifies another modeling restriction:
`ConstrainedDQN.min_count` and `rate_search_command` both start at `K+1` modules.
The implemented physical precoder instead has one row per active antenna and
uses all `M * Ns` independently weighted branches. With K=3 and Ns=8, four modules
are not necessary merely to obtain three spatial degrees of freedom. An
orthogonal-channel unit test will verify that a single eight-antenna module can
serve all three streams under this exact rate function. It is a constructive
counterexample to a rank-based explanation, not a claim of universal feasibility.

The paper is fully digital at the rate-model level. If the intended hardware is
one shared RF chain per module, that would require a different constrained hybrid
precoder, not silently using a module count as the number of digital dimensions.
No hardware assumption or old physical source is changed in this diagnostic.

Before observing domain-extension outcomes, specify exactly two causal search
domains, {4,...,16} and {1,...,16}, using the same predictor, margin, true QoS,
recovery and accounting. Replay the first four already-viewed development
trajectories per scenario. Check that domain {4,...,16} exactly reproduces the
archived PRS records; count all decisions selecting one to three modules in the
expanded domain. Save every trajectory and outage, including infeasible outcomes.
Do not train a new network yet or call a domain-expansion gain an RL gain.

This determines whether the existing four-module floor materially limits the
reported energy optimization. Any future main-model domain change must apply to
all learning and non-learning controls and regenerate the affected evidence;
old 13-output networks cannot be reinterpreted as 16-output models.
