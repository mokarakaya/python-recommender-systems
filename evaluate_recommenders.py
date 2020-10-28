import sys,os
sys.path.append(os.getcwd())
from evaluation.evaluate_cf import evaluate_cf_models
from evaluation.evaluate_cbf import evaluate_cbf_models
from evaluation.evaluate_hybrid import evaluate_hybrid_models
from evaluation.evaluate_baselines import evaluate_baseline_models


evaluate_baseline_models()
evaluate_cf_models()
evaluate_cbf_models()
evaluate_hybrid_models()
