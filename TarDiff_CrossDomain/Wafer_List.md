# TarDiff Cross-Domain Application - Wafer Semiconductor Data

## 1. Environment Setup
```bash
cd TarDiff
conda env create -f environment.yaml
conda activate tardiff
cd ../TarDiff_CrossDomain
```

## 2. Download Data
Download from UCR Archive: https://www.timeseriesclassification.com/description.php?Dataset=Wafer
Extract to `TarDiff_CrossDomain/data/raw/Wafer`

## 3. Data Preprocessing
```bash
python scripts/preprocess_wafer_for_tardiff.py --input_path data/raw/Wafer --output_path data/processed/wafer --seq_len 24 --val_ratio 0.1
```

## 4. Train Diffusion Model
```bash
cd ../TarDiff
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python train_main.py --base configs/base/wafer_base.yaml --name wafer_exp --logdir ../TarDiff_CrossDomain/models --max_steps 20000
```

## 5. Train Downstream Classifier
```bash
python classifier/classifier_train.py --num_classes 1 --rnn_type gru --hidden_dim 256 --train_data ../TarDiff_CrossDomain/data/processed/wafer/train_tuple.pkl --val_data ../TarDiff_CrossDomain/data/processed/wafer/val_tuple.pkl --ckpt_dir ../TarDiff_CrossDomain/models/wafer_base --input_dim 1
```

## 6. Compute Gradient Norm Statistics (Analyze class imbalance)
```bash
python ../TarDiff_CrossDomain/tools/compute_grad_norm.py \
    --data_path ../TarDiff_CrossDomain/data/processed/wafer/train_tuple.pkl \
    --model_path ../TarDiff_CrossDomain/models/wafer_base/classifier_best.pt \
    --input_dim 1 --num_classes 2
```

## 7. Generate Samples with Influence Guidance
```bash
for alpha in 0.5 0.25 0.1 0; do
    python guidance_generation.py --base configs/base/wafer_base.yaml --gen_ckpt_path ../TarDiff_CrossDomain/models/wafer_base/wafer_exp_24_nl_16_lr5.0e-05_bs256_ms20k_centered_pit_seed23/checkpoints/last.ckpt --downstream_pth_path ../TarDiff_CrossDomain/models/wafer_base/classifier_best.pt --origin_data_path ../TarDiff_CrossDomain/data/processed/wafer/train_tuple.pkl --save_path ../TarDiff_CrossDomain/data/processed/wafer --input_dim 1 --num_latents 1 --alpha $alpha
done
```

## 8. Evaluate Generation Quality
```bash
chmod +x ../TarDiff_CrossDomain/evaluation/run_quality_evaluation.sh
for alpha in 0.5 0.25 0.1 0; do ../TarDiff_CrossDomain/evaluation/run_quality_evaluation.sh wafer 20 $alpha; done
```

## 9. Run Downstream Task Experiments
```bash
chmod +x ../TarDiff_CrossDomain/evaluation/run_evaluation.sh
for alpha in 0.5 0.25 0.1 0; do ../TarDiff_CrossDomain/evaluation/run_evaluation.sh wafer 20 $alpha; done
```
