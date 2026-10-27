# DASE7506_Miniproject2

Complete Reproduction Commands and Compliance Statement:
Environment and Dependency Pinning: Python 3.11 / 3.12 with only numpy>=2.1,<2.3 (pinned in
requirements.txt) and fixed seed --seed 2026. All training uses exclusively the train split, and
hyperparameter selection uses exclusively the validation split; no test reference answers are used in
training or tuning. Standard Training and Evaluation Commands (executed in code/):

(1) Data check: python check_data.py;  

(2) Training: python train.py --data-dir data --checkpoint runs/model_bundle.npz --seed 2026;

(3) Test prediction: python predict.py --method submission.py --checkpoint
runs/model_bundle.npz --split test --output runs/test-predictions.json;  

(4) Scoring: python score_predictions.py --split test --predictions runs/test-predictions.json -- output runs/test-score.json; 

(5) Ablations: python evaluation.py --data-dir data --checkpoint runs/model_bundle.npz --output runs/ablation_results.json.
