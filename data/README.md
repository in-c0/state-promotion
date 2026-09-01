# Generated benchmark data

PALS files are generated deterministically rather than committed as opaque data blobs.

```bash
python scripts/generate_pals.py --seed 20260901
```

For paper runs, each manifest must record the generator commit SHA and seed.
