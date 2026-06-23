# Experimental Backends

RoboCasa and Genesis are tracked as experiments, not stable v0.1 backends.

The experiment configs live in `configs/experiments/` and describe the minimum integration contract required before these engines should be treated as supported:

- task spec to scene asset mapping
- success predicate mapping
- randomization support
- replay or event export
- failure taxonomy mapping

The adapters currently fail with explicit messages so users do not mistake them for production-ready backends.
