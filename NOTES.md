# Notes

## 2026-10-27 hardware and sizing

- Machine: i7-10750H (6 cores / 12 threads), 21 GB RAM, but only about 2 GB was free when I checked because other programs were open. GTX 1660 Ti with 6 GB, CUDA available (driver reports CUDA 13.1, I installed the torch cu126 wheel).
- Elliptic is small: 203,769 nodes x 166 float32 features is about 135 MB. Full-batch GCN/GraphSAGE fits on the GPU or the CPU without mini-batching or neighbour sampling, so I went with full-batch training.
- LightGBM uses 6 threads (one per physical core).

## 2026-10-27 data

- `torch_geometric.datasets.EllipticBitcoinDataset` downloaded fine, so I didn't need the Kaggle CSVs.
- PyG's processed `x` has 165 columns: it drops the time step. Weber et al. count 94 local features (time step + 93) and 166 in total, so `01_load_data.py` reads the raw CSVs PyG downloaded and keeps the time step as column 0. "Local" = columns 0-93, "all" = 0-165.
- Counts: 4,545 illicit, 42,019 licit, 157,205 unknown. About 2.2% of all nodes are illicit, but 9.8% of the labelled ones.
- **No edge crosses a time step** (0 of 234,355), and there are no self loops. Each time step is its own disconnected graph. So under the temporal split, a 2-layer GNN trained on the full graph never aggregates anything from steps 35-49 into a training node. Full-graph training doesn't leak across the split, and I don't need to cut the graph.
- Under the random split, train and test nodes share time steps and are often direct neighbours. That's part of what the random split rewards.
- The illicit share swings a lot from step to step: 0.4% at step 5, 36% at step 13. Validation (30-34) is 16.8% illicit and test (35-49) is 6.5%, so a threshold picked on validation is being moved to a period with a lower base rate.
- Steps 43-46 have 24, 24, 5 and 2 illicit labels; then 47-49 have 22, 36 and 56. More on this once the models are scored.

## 2026-10-27 split choices

- Temporal: fit on labelled nodes in steps 1-29 (26,381), validate on 30-34 (3,513), test on 35-49 (16,670). Every model fits on 1-29 only, including logistic regression, so all models see the same data. I don't refit on 1-34 afterwards.
- Random: for each seed, a stratified split of all labelled nodes with the same sizes (26,381 / 3,513 / 16,670). The seed changes both the split and the model's randomness.

## 2026-10-27 training setup

- GNNs run full-batch on the GPU: about 65-90 ms per epoch, against about 2.3 s on the CPU while LightGBM was using the cores. Two repeat runs on CUDA with the same seed gave identical predictions.
- Loss weight 0.7 illicit / 0.3 licit as in Weber et al.; early stopping on validation AUPRC, patience 100, max 1000 epochs.
- First temporal seeds: GCN with all features around 0.55 F1, GraphSAGE with all features around 0.63. GCN with only local features is unstable (0.29 to 0.48 across seeds).
