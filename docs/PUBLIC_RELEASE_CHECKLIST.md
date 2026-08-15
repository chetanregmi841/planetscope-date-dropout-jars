# Public release checklist

Before creating a public GitHub repository:

- [ ] Confirm institutional/advisor permission to release project code.
- [ ] Confirm no Planet API keys, credentials, tokens, cookies, or secrets are present.
- [ ] Confirm no restricted row-level yield data are present.
- [ ] Confirm no PlanetScope imagery or restricted derived row-level model inputs are present.
- [ ] Confirm no local network-drive paths or user-machine-specific paths are present.
- [ ] Choose a software license only after confirming project/institution permissions.
- [ ] Add the final journal DOI to `CITATION.cff` after publication.
- [ ] If the original final training script is recovered, compare it against `reference_model.py` and replace reference choices with archival code before claiming full retraining reproducibility.
