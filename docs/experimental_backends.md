# Experimental Backends

RoboCasa and Genesis are tracked as experiments, not stable v0.1 backends.

The experiment configs live in `configs/experiments/` and describe the minimum integration contract required before these engines should be treated as supported:

- task spec to scene asset mapping
- success predicate mapping
- randomization support
- replay or event export
- failure taxonomy mapping

Run contract validation with:

```bash
uv run python scripts/validate_backend.py robocasa
uv run python scripts/validate_backend.py genesis
```

The adapters can run real environments once a task provides a concrete mapping:

- RoboCasa: `success.engine_env_ids.robocasa` for a robosuite/RoboCasa environment, or `success.engine_factory.robocasa` as `module:function`.
- Genesis: `success.engine_factory.genesis` as `module:function`.

The factory receives a `TaskSpec` and must return an environment with `reset`, `step`, `render`, and `close` methods. Until those mappings and upstream assets exist, the adapters fail with explicit setup guidance instead of pretending the backend is supported.
