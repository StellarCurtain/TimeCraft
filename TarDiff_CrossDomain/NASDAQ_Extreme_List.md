# TarDiff Cross-Domain Application - NASDAQ Extreme Events (Three-class)

## 1. Environment Setup
```bash
cd TarDiff
conda env create -f environment.yaml
conda activate tardiff
```

## 2. Download Data
Download from Kaggle: https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset
Extract to `TarDiff_CrossDomain/data/raw/NASDAQ`

## 3. Data Preprocessing (Three-class: Loss/Neutral/Gain)
```bash
python ../TarDiff_CrossDomain/scripts/preprocess_nasdaq_for_tardiff.py --input_path ../TarDiff_CrossDomain/data/raw/NASDAQ --output_path ../TarDiff_CrossDomain/data/processed/nasdaq_extreme --seq_len 24 --pred_horizon 5 --label_mode extreme --extreme_percentile 5
```

## 4. Train Diffusion Model
```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python train_main.py --base configs/base/nasdaq_extreme_base.yaml --name nasdaq_extreme_exp --logdir ../TarDiff_CrossDomain/models --max_steps 50000
```

## 5. Train Downstream Classifier
```bash
python classifier/classifier_train.py --num_classes 3 --rnn_type gru --hidden_dim 256 --train_data ../TarDiff_CrossDomain/data/processed/nasdaq_extreme/train_tuple.pkl --val_data ../TarDiff_CrossDomain/data/processed/nasdaq_extreme/val_tuple.pkl --ckpt_dir ../TarDiff_CrossDomain/models/nasdaq_extreme_base --input_dim 5
```

## 6. Compute Gradient Norm Statistics (Analyze class imbalance)
```bash
python ../TarDiff_CrossDomain/tools/compute_grad_norm.py \
    --data_path ../TarDiff_CrossDomain/data/processed/nasdaq_extreme/train_tuple.pkl \
    --model_path ../TarDiff_CrossDomain/models/nasdaq_extreme_base/classifier_best.pt \
    --input_dim 5 --num_classes 3 --max_samples 10000
```

## 7. Create Data Subset (Optional, to reduce generation time)
```bash
python ../TarDiff_CrossDomain/tools/create_subset.py --ratio 0.5 --input ../TarDiff_CrossDomain/data/processed/nasdaq_extreme/train_tuple.pkl --output ../TarDiff_CrossDomain/data/processed/nasdaq_extreme/train_tuple_50.pkl
```

## 8. Generate Samples with Influence Guidance
```bash
for alpha in 0.5 0.25 0.1 0; do
    python guidance_generation.py --base configs/base/nasdaq_extreme_base.yaml --gen_ckpt_path ../TarDiff_CrossDomain/models/nasdaq_extreme_base/nasdaq_extreme_exp_24_nl_16_lr5.0e-05_bs256_ms50k_centered_pit_seed23/checkpoints/last.ckpt --downstream_pth_path ../TarDiff_CrossDomain/models/nasdaq_extreme_base/classifier_best.pt --origin_data_path ../TarDiff_CrossDomain/data/processed/nasdaq_extreme/train_tuple_50.pkl --save_path ../TarDiff_CrossDomain/data/processed/nasdaq_extreme --input_dim 5 --num_latents 1 --num_classes 3 --alpha $alpha
done
```

## 9. Evaluate Generation Quality
```bash
chmod +x ../TarDiff_CrossDomain/evaluation/run_quality_evaluation.sh
for alpha in 0.5 0.25 0.1 0; do ../TarDiff_CrossDomain/evaluation/run_quality_evaluation.sh nasdaq_extreme 50 $alpha; done
```

## 10. Run Downstream Task Experiments
```bash
chmod +x ../TarDiff_CrossDomain/evaluation/run_evaluation.sh
for alpha in 0.5 0.25 0.1 0; do ../TarDiff_CrossDomain/evaluation/run_evaluation.sh nasdaq_extreme 50 $alpha; done
```

---

## Notes
- `--label_mode extreme --extreme_percentile 5`: Top/bottom 5% returns → Gain/Loss classes (~5% each, ~90% Neutral)
- This creates class imbalance where TarDiff's Influence Guidance can be more effective
- Adjust `--extreme_percentile` to control imbalance ratio (e.g., 10 for ~10% each class)
