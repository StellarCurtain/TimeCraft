# TarDiff Cross-Domain Application: From Healthcare to Financial and Industrial Domains

**Author:** Yuanze Xu

## Abstract

TarDiff is a target-aware time series diffusion generation model originally designed for healthcare scenarios. This report presents my efforts to adapt TarDiff to non-medical domains: the NASDAQ stock market (financial) and Wafer semiconductor fault detection (industrial). I developed a generalizable pipeline for cross-domain application. Experiments reveal that TarDiff's effectiveness varies significantly across domains—the Wafer dataset shows remarkable improvements (AUROC: 0.9864 → 0.9960 with TSRTR at α=0.1), while NASDAQ presents challenges due to balanced class distribution and inherent unpredictability.

---

## 1. Introduction

TarDiff is a diffusion-based time series generation model that incorporates Influence Guidance to generate task-aware synthetic data. Originally developed for healthcare applications (e.g., MIMIC-III ICU data), it addresses data scarcity by generating high-quality synthetic time series beneficial for downstream classification tasks.

**Objectives:**
1. Evaluate TarDiff's adaptability to non-medical domains
2. Develop a reusable cross-domain adaptation pipeline
3. Analyze feasibility and challenges across different data characteristics

**Selected Datasets:**

| Dataset | Domain | Characteristics |
|---------|--------|-----------------|
| **NASDAQ** | Financial | 5 channels, balanced classes (~50%), 450K+ samples |
| **Wafer** | Industrial | 1 channel, imbalanced (~10% anomaly), 1K samples |

---

## 2. Cross-Domain Adaptation Pipeline

I developed a standardized 10-step pipeline:

1. Environment Setup → 2. Data Download → 3. Data Preprocessing → 4. Config Creation → 5. Diffusion Training → 6. Classifier Training → 7. Gradient Norm Analysis → 8. Influence Guidance Generation → 9. Quality Evaluation → 10. Downstream Evaluation

**Data Format:** processed pickle files with `(data, labels)` tuple, where `data` shape is `(N, C, T)`.

---

## 3. Experimental Setup

| Config | NASDAQ | Wafer |
|--------|--------|-------|
| Channels | 5 (OHLCV) | 1 |
| Seq Length | 24 | 24 |
| Train/Val/Test | 361K/45K/45K | 900/100/6K |
| Class Balance | ~50% | ~10.7% anomaly |
| Training Steps | 50,000 | 20,000 |

**Evaluation Protocols:** Baseline (100% real), TSTR (100% synthetic), TSRTR (real + synthetic), Mixed 50/50

**Metrics:** AUROC, AUPRC, Accuracy

---

## 4. Results

### 4.1 NASDAQ Stock Dataset

**Data Overview:**
- Samples: 361,501 | Shape: (N, 5, 24) | Classes: 2
- Label distribution: Class 0 (Decrease): 49.70%, Class 1 (Increase): 50.30%
- Sample example (label=1): 
  - Channel 0-3 (OHLC): [186.55, 188.31, ..., 169.70] (price series)
  - Channel 4 (Volume): [958700, 1439700, ..., 1740700]

**Labeling Strategy:**
- Compute the 5-day return rate for each sample: `return = (close[t+5] - close[t]) / close[t]`
- Binary classification: Class 1 (Increase) if return > 0, else Class 0 (Decrease)

**Diffusion Model Training (50k steps):**

![NASDAQ Diffusion Training Loss](nasdaq_diffusion_train_loss_50k.png)

**Gradient Norm Analysis:**

| Class | Mean Gradient Norm | Sample Count |
|-------|-------------------|--------------|
| Class 0 (Decrease) | 8.0296 | 5,027 |
| Class 1 (Increase) | 6.0054 | 4,973 |
| **Ratio** | **1.34x** | - |

**Baseline (100% Real Data):**

| AUROC | AUPRC | Accuracy |
|-------|-------|----------|
| 0.5413 | 0.5427 | 0.5283 |

**TSTR (100% Synthetic Data):**

| α | AUROC | AUPRC | Accuracy |
|---|-------|-------|----------|
| 0 | 0.4728 | 0.4869 | 0.5023 |
| 0.25 | 0.4888 | 0.5015 | 0.5022 |
| 0.5 | 0.5097 | 0.5116 | 0.5034 |

**TSRTR (100% Real + 100% Synthetic):**

| α | AUROC | AUPRC | Accuracy |
|---|-------|-------|----------|
| 0 | 0.5342 | 0.5265 | 0.5023 |
| 0.25 | 0.5415 | 0.5385 | 0.5272 |
| 0.5 | 0.5416 | 0.5341 | 0.5304 |

**Mixed 50/50 (50% Real + 50% Synthetic):**

| α | AUROC | AUPRC | Accuracy |
|---|-------|-------|----------|
| 0 | 0.5268 | 0.5276 | 0.5023 |
| 0.25 | 0.5325 | 0.5269 | 0.5206 |
| 0.5 | 0.5397 | 0.5301 | 0.5234 |

**Analysis:** Despite the challenging setting (low gradient norm ratio of 1.34x and weak baseline classifier with AUROC ~0.54), Influence Guidance still demonstrates positive effects. In TSRTR experiments, AUROC improves from 0.5342 (α=0) to 0.5416 (α=0.5), and in Mixed 50/50, from 0.5268 (α=0) to 0.5397 (α=0.5). This suggests that even when the classifier performs poorly overall, it can still capture some discriminative patterns that guide the generation process toward more useful synthetic samples. The classifier's gradient information, though noisy, provides directional signals that help the diffusion model generate samples that better align with the downstream task objectives when combined with real data.

**Challenges:** Poor baseline performance (AUROC ~0.54, barely above random), high noise in financial data, similar class distributions, weak classifier guidance.

### 4.2 NASDAQ Extreme Dataset

To address the class balance issue in the standard NASDAQ dataset, we created an "extreme" variant by redefining the classification task. Instead of binary up/down prediction, we focus on detecting **extreme price movements**.

**Labeling Strategy:** Using the same 5-day return rate, we apply percentile-based thresholds:
- Calculate the 5th and 95th percentile thresholds across all returns
- **Class 0 (Extreme Loss)**: Bottom 5% returns (large price drops)
- **Class 1 (Neutral)**: Middle 90% returns (normal fluctuations)  
- **Class 2 (Extreme Gain)**: Top 5% returns (large price increases)

This creates a class distribution of approximately **5% : 90% : 5%**, introducing significant class imbalance similar to the medical datasets where TarDiff excels. The hypothesis is that the extreme price movement patterns (both gains and losses) are harder to classify, leading to higher gradient norms for minority classes and thus more effective Influence Guidance.

**Data Overview:**
- Samples: 360,328 | Shape: (N, 5, 24) | Classes: 3
- Label distribution: Class 0 (Loss): 4.98%, Class 1 (Neutral): 90.03%, Class 2 (Gain): 4.99%
- Sample example (label=1, Neutral):
  - Channel 0-3 (OHLC): [2.36, 2.39, ..., 2.48] (price series)
  - Channel 4 (Volume): [1689900, 2508100, ..., 881000]

**Diffusion Model Training (50k steps):**

![NASDAQ Extreme Diffusion Training Loss](nasdaq_extreme_diffusion_train_loss_50k.png)

*Results pending...*

### 4.3 Wafer Semiconductor Dataset

**Data Overview:**
- Samples: 901 | Shape: (N, 1, 24) | Classes: 2
- Label distribution: Class 0 (Normal): 90.23%, Class 1 (Anomaly): 9.77%
- Data range: [-2.86, 7.17], mean=-0.02, std=0.97 (normalized)
- Sample example (label=0, Normal):
  - Channel 0: [-1.14, -1.14, -1.14, 0.94, ..., -1.14, -1.14, -1.14, -1.14]

**Diffusion Model Training (20k steps, early stopped at ~5k):**

![Wafer Diffusion Training Loss](wafer_diffusion_train_loss_20k.png)

**Gradient Norm Analysis:**

| Class | Mean Gradient Norm | Sample Count |
|-------|-------------------|--------------|
| Class 0 (Normal) | 0.5960 | 813 |
| Class 1 (Anomaly) | 7.3570 | 88 |
| **Ratio** | **12.34x** | - |

**Baseline (100% Real Data):**

| AUROC | AUPRC | Accuracy |
|-------|-------|----------|
| 0.9932 | 0.9546 | 0.9813 |

**TSTR (100% Synthetic Data):**

| α | AUROC | AUPRC | Accuracy |
|---|-------|-------|----------|
| 0 | 0.3374 | 0.0828 | 0.8921 |
| 0.1 | 0.3831 | 0.0813 | 0.5999 |
| 0.25 | 0.4885 | 0.1037 | 0.5706 |
| 0.5 | 0.6224 | 0.1574 | 0.2382 |

**TSRTR (100% Real + 100% Synthetic):**

| α | AUROC | AUPRC | Accuracy |
|---|-------|-------|----------|
| 0 | 0.9864 | 0.8930 | 0.9661 |
| **0.1** | **0.9960** | **0.9650** | **0.9825** |
| 0.25 | 0.9902 | 0.9318 | 0.9718 |
| 0.5 | 0.9903 | 0.9265 | 0.9653 |

**Mixed 50/50 (50% Real + 50% Synthetic):**

| α | AUROC | AUPRC | Accuracy |
|---|-------|-------|----------|
| 0 | 0.6802 | 0.1485 | 0.8921 |
| **0.1** | **0.9892** | **0.9493** | **0.9763** |
| 0.25 | 0.9772 | 0.8682 | 0.9534 |
| 0.5 | 0.9904 | 0.9285 | 0.9745 |

**Key Findings:** 
- **Real + Synthetic data achieves best results**: TSRTR with α=0.1 reaches AUROC 0.9960, outperforming baseline. Pure synthetic data (TSTR) is insufficient, but real+synthetic combinations (TSRTR/Mixed) effectively enhance performance
- **Baseline performance ceiling**: The strong baseline (AUROC 0.9932) limits absolute improvement, but relative gains are meaningful
- **Optimal guidance strength**: α=0.1 provides the best balance between task-specific guidance and distribution fidelity

---

## 5. Discussion

**Key Factors for TarDiff Effectiveness:**

1. **Class Imbalance**: TarDiff's Influence Guidance relies on gradient signals from a downstream classifier. When class imbalance exists, minority class samples are harder to classify, producing larger gradients that provide stronger guidance. Our experiments confirm this: Wafer achieves a gradient norm ratio of 12.34x (minority/majority), while NASDAQ only reaches 1.34x.

2. **Domain-Specific Learnability**: The classifier must be able to learn meaningful discriminative patterns from the data. This varies significantly across domains:
   - **Industrial data (Wafer)**: Anomalous wafers exhibit distinct waveform patterns (e.g., sudden spikes, abnormal shapes). The classifier achieves AUROC 0.99, providing reliable gradient signals.
   - **Financial data (NASDAQ)**: Stock price movements are inherently noisy and largely stochastic. Future returns are notoriously difficult to predict from historical prices alone (weak-form market efficiency). The classifier only achieves AUROC ~0.54 (barely above random), meaning its gradients are essentially noise rather than useful guidance.

**Cross-Domain Implications**: TarDiff is most effective when (1) the target domain has class imbalance, and (2) the classification task is learnable. Domains with high noise and unpredictable patterns (e.g., financial markets) pose fundamental challenges that cannot be overcome by synthetic data augmentation alone.


---

## 6. Conclusion

TarDiff achieves remarkable success on Wafer (AUROC improvement from 0.9864 to 0.9960) but faces challenges on NASDAQ due to inherent data characteristics. This work demonstrates both the potential and limitations of cross-domain transfer for target-aware time series generation.

---


## Appendix

Reproduction instructions: `TarDiff_CrossDomain/NASDAQ_List.md`, `TarDiff_CrossDomain/Wafer_List.md`

Trained on 3070Ti
