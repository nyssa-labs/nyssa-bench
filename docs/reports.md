# Reports

NyssaBench writes an HTML report for each run. Reports include:

- suite, policy, engine, and episode count
- success rate
- prototype reliability score
- primary failure mode
- public-claim validation status
- unsupported stressors
- top failure episodes and replay links when available
- aggregate metrics
- failure counts
- raw summary JSON

Use `nyssa report <run>` to regenerate `report.html` from a run directory. Use `nyssa compare` for multi-policy reports.
