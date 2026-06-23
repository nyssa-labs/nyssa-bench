# Adding New Policies

Policy files can be passed directly to `nyssa run --policy path/to/policy.py`.

The file must expose either `create_policy()` or `PolicyAdapter`. The resulting object must implement:

```python
def act(observation):
    ...
```

Optional methods:

```python
def reset(task=None, seed=None):
    ...

def close():
    ...
```
