## Machine Learning Portfolio

My machine learning portfolio linking to individual project repositories.

### [Diffusion Audio Restoration (Low-Bitwidth Speech Dequantization)](https://github.com/kirti8192/diffusion-model-audio-restoration.git)

Conditional diffusion for **speech restoration from severe bit-depth reduction (4-bit)**, focusing on perceptual quality improvement while remaining faithful to the quantized observation.

- Domain: Complex STFT (2-channel real/imag) with 256×256 patches (`n_fft=512`, `hop=128`)
- Formulations: conditioned clean diffusion `p(x_clean | x_q)` and residual diffusion `p(r | x_q)` where `r = x_clean - x_q`, `x_hat = x_q + r_hat`
- Objective: `ε`-prediction with magnitude-weighted loss; timeline truncated to low-noise regime (`~0–10% of T`)
- Inference: SDEdit-style DDIM from `x_init = add_noise(x_q, ε, t_start)`
- Long-form: patch-wise restoration + overlap-averaged STFT stitching + ISTFT
- Evaluation: perceptual listening (`x_q -> x_hat -> x_clean`)
- Key insight: STFT conditioning outperforms time-domain diffusion; residual diffusion yields speech-correlated enhancement despite imperfect spectra

### [Neural Machine Translation (Multilingual NMT with Back-Translation)](https://github.com/kirti8192/neural-machine-translation/tree/main)

Multilingual neural machine translation across English, Tamil, and Bengali, focusing on zero-shot transfer and back-translation for zero-resource language pairs.

- Models: mBART-50, M2M-100, NLLB-200  
- Data: ai4bharat/samanantar (`en`↔`ta`, `en`↔`bn`) with *synthetic* `ta`↔`bn` via back-translation  
- Training: English-centric fine-tuning followed by back-translation  
- Evaluation: BLEU (n-gram analysis) and COMET (semantic, human-aligned metric)  
- Key insight: Back-translation improves cross-lingual transfer even when validation loss does not decrease; MT-specific models remain stable while mBART-50 exhibits target-language drift under heavy back-translation fine-tuning

### [Fantasy League Time-Series Forecasting](https://github.com/kirti8192/fantasy-league-time-series)

Time-series forecasting of fantasy-league player points with an explicit focus on **ranking quality** under heavy zero inflation and temporal dependence.

- Models: Hurdle-based XGBoost (two-stage hurdle classifier + regressor), Temporal Fusion Transformer (TFT), Custom Two-Head TFT (classification + regression)
- Data: Player-gameweek level historical data across multiple seasons with strict time-based splits
- Framing: Expected points prediction optimized for ranking rather than point-wise accuracy
- Techniques: Hurdle modeling for zero inflation, rolling backtests, Optuna-based hyperparameter optimization, custom-built TFT with hurdle-based output head
- Evaluation: MAE/RMSE for accuracy and Spearman correlation for ranking quality
- Key insight: Explicitly modeling zero inflation consistently improves player ranking quality; increased model complexity alone does not guarantee better performance though


### [Diabetes Prediction Challenge (Kaggle Playground)](https://github.com/kirti8192/kaggle-playground-diabetes-prediction)

Binary classification using gradient boosting models with proper cross-validation and out-of-fold ensembling.

- Models: XGBoost, LightGBM, CatBoost  
- Validation: Stratified 5-fold CV with OOF predictions 
- Submission: Ensemble model with weighted predictions
- Metric: ROC-AUC (private LB: 0.69496)
