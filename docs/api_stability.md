# API Stability

NyssaBench is pre-1.0. The following APIs are treated as stable candidates:

- `nyssa_bench.Suite`
- `nyssa_bench.PolicyRunner`
- `nyssa_bench.core.task.TaskSpec`
- `nyssa_bench.engines.base.NyssaEngine`
- `nyssa_bench.policies.base.PolicyLike`

Before v1.0, breaking changes may still occur. The v1.0 target is to freeze:

- task spec schema
- suite spec schema
- engine adapter API
- policy adapter API
- metrics summary shape
- run artifact layout
- plugin registration API
