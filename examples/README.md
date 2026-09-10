# agentsim examples

- `banking_agent.py` — deterministic tool-using agent + registry (no LLM)
- `campaign.json` — sample fault campaign
- `contracts.json` — sample contract suite
- `fixtures/` — recorded trajectories for CI replay

```bash
# clean run (no faults)
agentsim run --seed 0 --contracts examples/contracts.json --record /tmp/t.json

# chaos campaign sweep
agentsim campaign --seed-start 0 --seed-count 25 --campaign examples/campaign.json \
  --contracts examples/contracts.json --replay /tmp/t.json

# shrink a failing trace
agentsim shrink /tmp/fail.json --campaign examples/campaign.json \
  --contracts examples/contracts.json --out /tmp/fail.min.json
```
