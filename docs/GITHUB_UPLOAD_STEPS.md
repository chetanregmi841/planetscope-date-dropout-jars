# GitHub upload steps

1. Create a new GitHub repository, suggested name: `planetscope-date-dropout-jars`.
2. Keep the repository **private initially** until institutional/project permission for public code release is confirmed.
3. Upload the contents of this repository root (not the outer ZIP itself).
4. Do not add restricted data, imagery, model checkpoints, credentials, or local project paths.
5. Run the repository tests locally: `PYTHONPATH=src pytest -q`.
6. After the GitHub repository exists, replace `ADD_GITHUB_REPOSITORY_URL_AFTER_CREATION` in `CITATION.cff` with the repository URL.
7. Choose a software license only after confirming institutional/project permissions.
8. If the journal form requires a code-sharing platform before the repository can be made public, provide the private-repository platform choice now and update the Code and Data Availability statement with the final public/archived URL at revision or acceptance, subject to journal policy.
