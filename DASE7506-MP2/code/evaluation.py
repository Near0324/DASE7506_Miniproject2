from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from common import DATA, load_inputs, read_jsonl
from submission import (
    IDX_ANS,
    IDX_EMB_STAT,
    IDX_REAS,
    IDX_SURF,
    NUM_TAB_FEATURES,
    _safe_matmul,
    build_model_features,
    compute_normalized_embeddings,
    compute_step_completion_signals,
    compute_step_structural_prior,
    extract_tabular_and_meta,
    predict_with_scores,
    select_with_cluster_aggregation,
)
from train import (
    compute_correctness_matrix,
    evaluate_accuracy,
    fit_pca_basis,
    identify_hard_groups,
    train_and_save,
    train_listwise_linear,
)


def baseline_index_zero(n: int) -> list[int]:
    return [0] * n


def baseline_majority_vote(
    parsed_answers: list[list[str | None]],
    valid_mask: np.ndarray,
) -> list[int]:
    """Standard unweighted Self-Consistency (Majority Vote) with first-valid tie-break."""
    choices: list[int] = []
    for i, ans_list in enumerate(parsed_answers):
        valid_ans = [a for a in ans_list if a is not None]
        if not valid_ans:
            choices.append(0)
            continue
        counts = Counter(valid_ans)
        best_c = 0
        best_cnt = -1
        for c_idx in range(4):
            a = ans_list[c_idx]
            if a is not None and counts[a] > best_cnt:
                best_cnt = counts[a]
                best_c = c_idx
        choices.append(best_c)
    return choices


def train_pointwise_isolated(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    valid_tr: np.ndarray,
    l2: float = 0.25,
    steps: int = 300,
    lr: float = 0.1,
) -> tuple[np.ndarray, float]:
    """Train a pointwise binary logistic regression classifier judging each candidate in isolation."""
    x_flat = x_tr.reshape(-1, x_tr.shape[-1])
    y_flat = y_tr.reshape(-1)
    v_flat = valid_tr.reshape(-1)

    x_v = np.ascontiguousarray(x_flat[v_flat], dtype=np.float32)
    y_v = y_flat[v_flat]
    m, d = x_v.shape

    w = np.zeros(d, dtype=np.float32)
    b = 0.0
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        for _ in range(steps):
            z = np.clip(np.matmul(x_v, w) + b, -25.0, 25.0)
            p = 1.0 / (1.0 + np.exp(-z))
            err = (p - y_v) / float(m)
            grad_w = np.matmul(x_v.T, err) + l2 * w
            grad_b = float(np.sum(err))
            w -= lr * grad_w
            b -= lr * grad_b
    return w, b


def compute_ece_and_quartiles(
    cluster_posteriors: np.ndarray,
    choices: list[int],
    y_true: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute Expected Calibration Error (ECE) and accuracy across confidence quartiles."""
    conf = np.array([cluster_posteriors[i, c] for i, c in enumerate(choices)], dtype=np.float32)
    corr = np.array([y_true[i, c] for i, c in enumerate(choices)], dtype=np.float32)

    order = np.argsort(conf, kind='mergesort')
    quartiles = np.array_split(order, 4)
    q_stats = {}
    for q_idx, idx_arr in enumerate(quartiles, start=1):
        if len(idx_arr) > 0:
            q_stats[f'Q{q_idx}'] = {
                'count': int(len(idx_arr)),
                'mean_confidence': round(float(100.0 * np.mean(conf[idx_arr])), 2),
                'accuracy': round(float(100.0 * np.mean(corr[idx_arr])), 2),
            }

    bins = np.linspace(0.25, 1.0, n_bins + 1)
    ece = 0.0
    for b_idx in range(n_bins):
        in_bin = (conf >= bins[b_idx]) & (
            conf < bins[b_idx + 1] if b_idx < n_bins - 1 else conf <= bins[b_idx + 1]
        )
        if np.any(in_bin):
            prop = float(np.mean(in_bin))
            acc_bin = float(np.mean(corr[in_bin]))
            conf_bin = float(np.mean(conf[in_bin]))
            ece += prop * abs(acc_bin - conf_bin)

    return {
        'ece_percent': round(100.0 * ece, 3),
        'confidence_quartiles': q_stats,
    }


def run_all_ablations(
    data_dir: Path = DATA,
    checkpoint: Path = Path('runs/model_bundle.npz'),
    output_json: Path = Path('runs/ablation_results.json'),
) -> dict[str, Any]:
    """Execute the full experimental suite and write structured results to JSON."""
    needs_train = not checkpoint.exists()
    if not needs_train:
        with np.load(checkpoint, allow_pickle=False) as b:
            if 'version_tag' not in b.files or int(b['version_tag'][0]) != 3:
                needs_train = True
    if needs_train:
        train_and_save(data_dir=data_dir, output_ckpt=checkpoint)

    tr_bank, tr_arr = load_inputs('train', data_dir)
    va_bank, va_arr = load_inputs('validation', data_dir)
    te_bank, te_arr = load_inputs('test', data_dir)

    tr_refs = read_jsonl(Path(data_dir) / 'gsm8k_train.jsonl')
    va_refs = read_jsonl(Path(data_dir) / 'gsm8k_validation.jsonl')
    te_refs = read_jsonl(Path(data_dir) / 'gsm8k_test.jsonl')

    tab_tr, valid_tr, ans_tr = extract_tabular_and_meta(tr_bank, tr_arr['embeddings'])
    tab_va, valid_va, ans_va = extract_tabular_and_meta(va_bank, va_arr['embeddings'])
    tab_te, valid_te, ans_te = extract_tabular_and_meta(te_bank, te_arr['embeddings'])

    se_tr, cf_tr, _ = compute_step_completion_signals(tr_bank, tab_tr, ans_tr)
    se_va, cf_va, _ = compute_step_completion_signals(va_bank, tab_va, ans_va)
    se_te, cf_te, _ = compute_step_completion_signals(te_bank, tab_te, ans_te)

    prior_va = compute_step_structural_prior(tab_va, se_va, cf_va, valid_va)
    prior_te = compute_step_structural_prior(tab_te, se_te, cf_te, valid_te)

    y_tr = compute_correctness_matrix(tr_bank, tr_refs, ans_tr)
    y_va = compute_correctness_matrix(va_bank, va_refs, ans_va)
    y_te = compute_correctness_matrix(te_bank, te_refs, ans_te)

    hard_va = identify_hard_groups(ans_va)
    hard_te = identify_hard_groups(ans_te)

    diff_tr_u, raw_tr_u = compute_normalized_embeddings(tr_arr['embeddings'])
    pca_diff_mean, pca_diff_basis = fit_pca_basis(diff_tr_u.reshape(-1, 1024), 96)
    pca_raw_mean, pca_raw_basis = fit_pca_basis(raw_tr_u.reshape(-1, 1024), 32)

    x_tr_full = build_model_features(
        tab_tr, tr_arr['embeddings'], ans_tr,
        pca_diff_mean, pca_diff_basis, pca_raw_mean, pca_raw_basis, cluster_dim=32
    )
    x_va_full = build_model_features(
        tab_va, va_arr['embeddings'], ans_va,
        pca_diff_mean, pca_diff_basis, pca_raw_mean, pca_raw_basis, cluster_dim=32
    )
    x_te_full = build_model_features(
        tab_te, te_arr['embeddings'], ans_te,
        pca_diff_mean, pca_diff_basis, pca_raw_mean, pca_raw_basis, cluster_dim=32
    )

    f_mean = np.mean(x_tr_full, axis=(0, 1), keepdims=True).astype(np.float32)
    f_std = np.maximum(np.std(x_tr_full, axis=(0, 1), keepdims=True), 1e-6).astype(np.float32)
    x_tr_norm = np.clip((x_tr_full - f_mean) / f_std, -8.0, 8.0)
    x_va_norm = np.clip((x_va_full - f_mean) / f_std, -8.0, 8.0)
    x_te_norm = np.clip((x_te_full - f_mean) / f_std, -8.0, 8.0)

    def eval_subset(
        cols: list[int],
        alpha: float = 0.0,
        name: str = '',
        l2_vote: float | None = None,
        add_step_prior: bool = False,
    ) -> dict[str, Any]:
        w, b = train_listwise_linear(
            x_tr_norm[:, :, cols], y_tr, valid_tr,
            l2_tab=0.06, l2_emb=0.35, n_tab=min(len(cols), NUM_TAB_FEATURES), steps=300,
            l2_vote=l2_vote,
        )
        t_infer = time.perf_counter()
        va_logits = _safe_matmul(x_va_norm[:, :, cols], w) + b
        te_logits = _safe_matmul(x_te_norm[:, :, cols], w) + b
        if add_step_prior:
            va_logits = va_logits + prior_va
            te_logits = te_logits + prior_te
        va_ch, _ = select_with_cluster_aggregation(va_logits, valid_va, ans_va, cluster_alpha=alpha)
        te_ch, _ = select_with_cluster_aggregation(te_logits, valid_te, ans_te, cluster_alpha=alpha)
        infer_ms = 1000.0 * (time.perf_counter() - t_infer)
        return {
            'method': name,
            'num_features': len(cols),
            'val_all_acc': round(evaluate_accuracy(va_ch, y_va), 2),
            'val_hard_acc': round(evaluate_accuracy(va_ch, y_va, hard_va), 2),
            'test_acc': round(evaluate_accuracy(te_ch, y_te), 2),
            'infer_ms': round(infer_ms, 2),
            'va_choices': va_ch,
            'te_choices': te_ch,
        }

    c0_va = baseline_index_zero(len(va_bank))
    c0_te = baseline_index_zero(len(te_bank))
    mv_va = baseline_majority_vote(ans_va, valid_va)
    mv_te = baseline_majority_vote(ans_te, valid_te)

    oracle_va = float(100.0 * np.mean(np.max(y_va, axis=1)))
    oracle_va_hard = float(100.0 * np.mean(np.max(y_va[hard_va], axis=1)))
    oracle_te = float(100.0 * np.mean(np.max(y_te, axis=1)))

    idx_emb_all = IDX_EMB_STAT + list(range(NUM_TAB_FEATURES, x_tr_norm.shape[-1]))
    idx_raw_pca_start = NUM_TAB_FEATURES + 96 + 32
    idx_raw_emb_only = [46] + list(range(idx_raw_pca_start, x_tr_norm.shape[-1]))
    idx_pointwise_features = (
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 20, 22, 24, 26, 27, 28, 29,
         30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 42]
        + idx_raw_emb_only
    )

    res_emb_only = eval_subset(idx_emb_all, alpha=0.0, name='Embeddings Only (Groupwise Centered PCA)')
    res_ans_only = eval_subset(IDX_ANS, alpha=0.35, name='Question + Final Answer Only (Plausibility + Consensus)')
    res_surf_only = eval_subset([0] + IDX_SURF, alpha=0.0, name='Surface & Formatting Features Only')
    res_reas_only = eval_subset([0] + IDX_REAS, alpha=0.0, name='Intrinsic Reasoning & Multi-Term Arithmetic Only', add_step_prior=True)
    res_text_all = eval_subset(IDX_ANS + IDX_SURF + IDX_REAS, alpha=0.45, name='Full Text & Answer Features (No Embeddings)', add_step_prior=True)

    w_pt, b_pt = train_pointwise_isolated(x_tr_norm[:, :, idx_pointwise_features], y_tr, valid_tr)
    va_pt_logits = _safe_matmul(x_va_norm[:, :, idx_pointwise_features], w_pt) + b_pt
    te_pt_logits = _safe_matmul(x_te_norm[:, :, idx_pointwise_features], w_pt) + b_pt
    va_pt_ch, _ = select_with_cluster_aggregation(va_pt_logits, valid_va, ans_va, cluster_alpha=0.0)
    te_pt_ch, _ = select_with_cluster_aggregation(te_pt_logits, valid_te, ans_te, cluster_alpha=0.0)

    res_group_linear = eval_subset(
        list(range(x_tr_norm.shape[-1])),
        alpha=0.55,
        name='Groupwise Comparative Listwise Linear',
        l2_vote=1.5,
        add_step_prior=True,
    )

    t_full0 = time.perf_counter()
    va_full_ch, va_full_post, _ = predict_with_scores(va_bank, va_arr, checkpoint)
    te_full_ch, te_full_post, _ = predict_with_scores(te_bank, te_arr, checkpoint)
    full_infer_ms = 1000.0 * (time.perf_counter() - t_full0)

    emb_te_ch = res_emb_only['te_choices']
    ans_agree_mask = np.array(
        [ans_te[i][mv_te[i]] == ans_te[i][emb_te_ch[i]] for i in range(len(te_bank))],
        dtype=bool,
    )
    ans_disagree_mask = ~ans_agree_mask

    results = {
        'dataset_summary': {
            'train_questions': len(tr_bank),
            'val_questions': len(va_bank),
            'val_hard_questions': int(np.sum(hard_va)),
            'test_questions': len(te_bank),
            'test_hard_questions': int(np.sum(hard_te)),
            'oracle_upper_bound': {
                'val_all': round(oracle_va, 2),
                'val_hard': round(oracle_va_hard, 2),
                'test': round(oracle_te, 2),
            },
        },
        'rq1_modality_comparison': [
            {
                'method': 'Candidate Index 0 (First Candidate Baseline)',
                'val_all_acc': round(evaluate_accuracy(c0_va, y_va), 2),
                'val_hard_acc': round(evaluate_accuracy(c0_va, y_va, hard_va), 2),
                'test_acc': round(evaluate_accuracy(c0_te, y_te), 2),
            },
            {
                'method': 'Unweighted Majority Vote (Self-Consistency)',
                'val_all_acc': round(evaluate_accuracy(mv_va, y_va), 2),
                'val_hard_acc': round(evaluate_accuracy(mv_va, y_va, hard_va), 2),
                'test_acc': round(evaluate_accuracy(mv_te, y_te), 2),
            },
            {k: v for k, v in res_emb_only.items() if not k.endswith('_choices')},
            {k: v for k, v in res_ans_only.items() if not k.endswith('_choices')},
            {k: v for k, v in res_text_all.items() if not k.endswith('_choices')},
            {
                'method': 'Full Hybrid Ensemble (ReliMath-Verifier)',
                'val_all_acc': round(evaluate_accuracy(va_full_ch, y_va), 2),
                'val_hard_acc': round(evaluate_accuracy(va_full_ch, y_va, hard_va), 2),
                'test_acc': round(evaluate_accuracy(te_full_ch, y_te), 2),
                'infer_ms': round(full_infer_ms, 2),
            },
        ],
        'rq2_surface_vs_reasoning': [
            {k: v for k, v in res_surf_only.items() if not k.endswith('_choices')},
            {k: v for k, v in res_reas_only.items() if not k.endswith('_choices')},
            {
                **{k: v for k, v in eval_subset(IDX_ANS + IDX_SURF, alpha=0.35, name='Answer Consensus + Surface Formatting').items() if not k.endswith('_choices')},
            },
            {
                **{k: v for k, v in eval_subset(IDX_ANS + IDX_REAS, alpha=0.35, name='Answer Consensus + Intrinsic Reasoning & Arithmetic', add_step_prior=True).items() if not k.endswith('_choices')},
            },
        ],
        'rq3_pointwise_vs_groupwise': [
            {
                'method': 'Pointwise Isolated Classifier (Judged on Its Own)',
                'val_all_acc': round(evaluate_accuracy(va_pt_ch, y_va), 2),
                'val_hard_acc': round(evaluate_accuracy(va_pt_ch, y_va, hard_va), 2),
                'test_acc': round(evaluate_accuracy(te_pt_ch, y_te), 2),
            },
            {k: v for k, v in res_group_linear.items() if not k.endswith('_choices')},
            {
                'method': 'Full Group-Interaction MLP + Answer-Cluster Aggregation',
                'val_all_acc': round(evaluate_accuracy(va_full_ch, y_va), 2),
                'val_hard_acc': round(evaluate_accuracy(va_full_ch, y_va, hard_va), 2),
                'test_acc': round(evaluate_accuracy(te_full_ch, y_te), 2),
            },
        ],
        'rq4_uncertainty_and_disagreement': {
            'validation_hard_calibration': compute_ece_and_quartiles(
                va_full_post[hard_va],
                [va_full_ch[i] for i in range(len(va_bank)) if hard_va[i]],
                y_va[hard_va],
            ),
            'test_calibration': compute_ece_and_quartiles(te_full_post, te_full_ch, y_te),
            'method_agreement_analysis_test': {
                'agree_count': int(np.sum(ans_agree_mask)),
                'disagree_count': int(np.sum(ans_disagree_mask)),
                'majority_vote_acc_when_agree': round(evaluate_accuracy(mv_te, y_te, ans_agree_mask), 2),
                'majority_vote_acc_when_disagree': round(evaluate_accuracy(mv_te, y_te, ans_disagree_mask), 2),
                'hybrid_verifier_acc_when_disagree': round(evaluate_accuracy(te_full_ch, y_te, ans_disagree_mask), 2),
            },
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))
    return results


def main():
    parser = argparse.ArgumentParser(description='Run DASE7506 MP2 Ablation Experiments')
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--checkpoint', type=Path, default=Path('runs/model_bundle.npz'))
    parser.add_argument('--output', type=Path, default=Path('runs/ablation_results.json'))
    args = parser.parse_args()
    run_all_ablations(data_dir=args.data_dir, checkpoint=args.checkpoint, output_json=args.output)


if __name__ == '__main__':
    main()