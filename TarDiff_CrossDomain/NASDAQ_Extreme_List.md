# NASDAQ Extreme Events Pipeline
# Three-class classification: Loss (0), Neutral (1), Gain (2)
# Creates class imbalance suitable for TarDiff Influence Guidance

## Step 1: Setup
```bash
conda activate tardiff
cd TarDiff_CrossDomain
```

## Step 2: Preprocess Data (Three-class)
```bash
python scripts/preprocess_nasdaq_for_tardiff.py \
    --input_path data/raw/NASDAQ \
    --output_path data/processed/nasdaq_extreme \
    --label_mode extreme \
    --extreme_percentile 5
```

## Step 3: Train Base Diffusion Model
```bash
cd ../TarDiff
python train_main.py --base configs/base/nasdaq_extreme_base.yaml --name nasdaq_extreme_exp --logdir ./outputs
```

## Step 4: Generate Synthetic Data with Influence Guidance
```bash
python guidance_generation.py \
    --config configs/base/nasdaq_extreme_base.yaml \
    --ckpt outputs/nasdaq_extreme_exp/checkpoints/last.ckpt \
    --outdir ../TarDiff_CrossDomain/data/generated/nasdaq_extreme \
    --guidance_data ../TarDiff_CrossDomain/data/processed/nasdaq_extreme/guidance_tuple.pkl \
    --n_samples 10000 \
    --alpha 0.5
```

## Step 5: Evaluate
```bash
cd ../TarDiff_CrossDomain
python evaluation/run_experiment.py \
    --real_data data/processed/nasdaq_extreme/train_tuple.pkl \
    --synthetic_data data/generated/nasdaq_extreme/generated.pkl \
    --test_data data/processed/nasdaq_extreme/test_tuple.pkl \
    --n_classes 3
```

---

## Notes
- `--extreme_percentile 5`: Top/bottom 5% returns → Gain/Loss classes (~5% each, ~90% Neutral)
- This creates class imbalance where TarDiff's Influence Guidance can be more effective
- Adjust `--extreme_percentile` to control imbalance ratio (e.g., 10 for ~10% each class)

