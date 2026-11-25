# Checkpoints

Trained weights are not included in this repository.

Training (`python main.py train ...`) writes, per fold, to `<run-dir>/fold{k}/`:

- `best_stress.pt`: weights with the best validation stress score (HR and RMSSD heads, stage head)
- `best_dis.pt`: weights with the best subject-level validation disorder loss (disorder head)
- `last.pt`: full training state, used to resume an interrupted run
- `history.csv`, `test_pred.csv`

Those files are ignored by git (`.gitignore`).
