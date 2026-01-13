## Machine Learning Portfolio

My machine learning portfolio linking to individual project repositories.

### [Diabetes Prediction Challenge (Kaggle Playground)](https://github.com/kirti8192/kaggle-playground-diabetes-prediction)

Binary classification using gradient boosting models with proper cross-validation and out-of-fold ensembling.

- Models: XGBoost, LightGBM, CatBoost  
- Validation: Stratified 5-fold CV with OOF predictions 
- Submission: Ensemble model with weighted predictions
- Metric: ROC-AUC (private LB: 0.69496)

### [Neural Machine Translation (Multilingual NMT with Back-Translation)](https://github.com/kirti8192/neural-machine-translation/tree/main)

Multilingual neural machine translation across English, Tamil, and Bengali, focusing on zero-shot transfer and back-translation for zero-resource language pairs.

- Models: mBART-50, M2M-100, NLLB-200  
- Data: ai4bharat/samanantar (`en`↔`ta`, `en`↔`bn`) with *synthetic* `ta`↔`bn` via back-translation  
- Training: English-centric fine-tuning followed by back-translation  
- Evaluation: BLEU (n-gram analysis) and COMET (semantic, human-aligned metric)  
- Key insight: Back-translation improves cross-lingual transfer even when validation loss does not decrease; MT-specific models remain stable while mBART-50 exhibits target-language drift under heavy back-translation fine-tuning