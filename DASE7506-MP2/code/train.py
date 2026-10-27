from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from common import DATA, check_data, load_inputs, parse_answer, read_jsonl
from submission import (
    IDX_EMB_STAT,
    NUM_TAB_FEATURES,
    _gelu,
    _safe_fold_linear,
    _safe_matmul,
    _softmax_4,
    build_model_features,
    compute_normalized_embeddings,
    compute_step_completion_signals,
    compute_step_structural_prior,
    extract_tabular_and_meta,
    forward_mlp_member,
    select_with_cluster_aggregation,
)


def compute_correctness_matrix(
    bank: list[dict[str, Any]],
    references: list[dict[str, Any]],
    parsed_answers: list[list[str | None]],
) -> np.ndarray:
    """Return boolean/float correctness matrix Y of shape [N, 4]."""
    n = len(bank)
    y = np.zeros((n, 4), dtype=np.float32)
    for i, (row, ref) in enumerate(zip(bank, references)):
        if row['id'] != ref['id']:
            raise ValueError(f"Alignment mismatch at index {i}: {row['id']} != {ref['id']}")
        gold_ans = parse_answer(ref['answer'])
        for c_idx in range(4):
            cand_ans = parsed_answers[i][c_idx]
            if cand_ans is not None and cand_ans == gold_ans:
                y[i, c_idx] = 1.0
    return y


def identify_hard_groups(parsed_answers: list[list[str | None]]) -> np.ndarray:
    """Return boolean mask [N] where group has >= 2 distinct valid parsed answers."""
    mask = np.zeros(len(parsed_answers), dtype=bool)
    for i, ans_list in enumerate(parsed_answers):
        distinct_valid = {a for a in ans_list if a is not None}
        if len(distinct_valid) >= 2:
            mask[i] = True
    return mask


def fit_pca_basis(x_flat: np.ndarray, k_components: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit exact SVD-based PCA basis on [M, 1024] matrix using NumPy."""
    mean = np.mean(x_flat, axis=0, dtype=np.float32)
    centered = (x_flat - mean).astype(np.float32)
    cov = _safe_matmul(centered.T, centered) / max(len(centered) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    top_idx = np.argsort(eigvals)[::-1][:k_components]
    basis = eigvecs[:, top_idx].astype(np.float32)
    scales = np.sqrt(np.maximum(eigvals[top_idx], 1e-5)).astype(np.float32)
    whitened_basis = np.ascontiguousarray(basis / scales[None, :], dtype=np.float32)
    return np.ascontiguousarray(mean, dtype=np.float32), whitened_basis


def train_listwise_linear(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    valid_tr: np.ndarray,
    l2_tab: float = 0.06,
    l2_emb: float = 0.35,
    n_tab: int = NUM_TAB_FEATURES,
    steps: int = 300,
    lr: float = 0.12,
    l2_vote: float | None = None,
) -> tuple[np.ndarray, float]:
    """Fit a multi-correct marginalized listwise linear model using fast BLAS SGEMV."""
    row_sum = np.sum(y_tr, axis=1)
    informative = (row_sum > 0) & (row_sum < 4)
    if np.sum(informative) < 10:
        informative = row_sum > 0

    x = np.ascontiguousarray(x_tr[informative], dtype=np.float32)
    y = y_tr[informative]
    v = valid_tr[informative]
    n_samples, _, d = x.shape
    x_2d = x.reshape(-1, d)

    w = np.zeros(d, dtype=np.float32)
    b = 0.0

    reg_vec = np.full(d, l2_emb, dtype=np.float32)
    if n_tab > 0:
        reg_vec[:min(n_tab, d)] = l2_tab
        if l2_vote is not None and d == 214:
            for vc_i in (9, 10, 11, 12, 13, 16, 17, 18, 19, 44, 45, 52, 53):
                reg_vec[vc_i] = l2_vote
            reg_vec[150:182] = l2_vote

    m_w = np.zeros_like(w)
    v_w = np.zeros_like(w)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        for t in range(1, steps + 1):
            logits = np.matmul(x_2d, w).reshape(n_samples, 4) + b
            logits = np.where(v, logits, -20.0)
            p = _softmax_4(logits)

            p_corr = p * y
            denom = np.maximum(np.sum(p_corr, axis=1, keepdims=True), 1e-12)
            q = p_corr / denom

            grad_s = (p - q) / float(n_samples)
            grad_w = np.matmul(grad_s.reshape(-1), x_2d) + reg_vec * w

            m_w = beta1 * m_w + (1.0 - beta1) * grad_w
            v_w = beta2 * v_w + (1.0 - beta2) * (grad_w ** 2)
            m_hat = m_w / (1.0 - beta1 ** t)
            v_hat = v_w / (1.0 - beta2 ** t)

            cur_lr = lr * 0.5 * (1.0 + np.cos(np.pi * t / steps))
            w -= cur_lr * m_hat / (np.sqrt(v_hat) + eps)

    return w.astype(np.float32), float(b)


def _gelu_grad(x: np.ndarray) -> np.ndarray:
    """Analytical derivative of approximate GELU."""
    k = np.sqrt(2.0 / np.pi)
    inner = k * (x + 0.044715 * (x ** 3))
    tanh_inner = np.tanh(inner)
    sech2 = 1.0 - tanh_inner ** 2
    d_inner = k * (1.0 + 3.0 * 0.044715 * (x ** 2))
    return 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * d_inner


def train_listwise_mlp(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    valid_tr: np.ndarray,
    hidden_dim: int = 48,
    epochs: int = 32,
    batch_size: int = 256,
    lr: float = 3.0e-3,
    weight_decay: float = 1.5e-2,
    wd_emb: float = 7.0e-2,
    wd_vote: float | None = None,
    dropout_rate: float = 0.22,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train a Listwise Group-Interaction Residual MLP with block-diagonal regularization Lambda."""
    rng = np.random.default_rng(seed)
    row_sum = np.sum(y_tr, axis=1)
    informative = (row_sum > 0) & (row_sum < 4)
    if np.sum(informative) < 10:
        informative = row_sum > 0

    x = np.ascontiguousarray(x_tr[informative], dtype=np.float32)
    y = y_tr[informative]
    v = valid_tr[informative]
    n_samples, _, d = x.shape

    w1 = rng.normal(0.0, np.sqrt(1.5 / d), size=(d, hidden_dim)).astype(np.float32)
    b1 = np.zeros(hidden_dim, dtype=np.float32)
    w_peer = rng.normal(0.0, np.sqrt(0.5 / hidden_dim), size=(hidden_dim, hidden_dim)).astype(np.float32)
    w2 = rng.normal(0.0, np.sqrt(1.0 / hidden_dim), size=(hidden_dim, 1)).astype(np.float32)
    b2 = np.zeros(1, dtype=np.float32)
    w_skip = rng.normal(0.0, np.sqrt(0.2 / d), size=(d, 1)).astype(np.float32)

    wd_vec = np.full((d, 1), wd_emb, dtype=np.float32)
    wd_vec[:min(NUM_TAB_FEATURES, d), 0] = weight_decay
    if wd_vote is not None and d == 214:
        for vc_i in (9, 10, 11, 12, 13, 16, 17, 18, 19, 44, 45, 52, 53):
            wd_vec[vc_i, 0] = wd_vote
        wd_vec[150:182, 0] = wd_vote

    params = [w1, b1, w_peer, w2, b2, w_skip]
    m_state = [np.zeros_like(p) for p in params]
    v_state = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    step = 0
    total_steps = max(1, epochs * ((n_samples + batch_size - 1) // batch_size))

    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        for _ in range(epochs):
            perm = rng.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                step += 1
                idx = perm[start:start + batch_size]
                xb = x[idx]
                yb = y[idx]
                vb = v[idx]
                bsz = len(idx)

                if dropout_rate > 0.0:
                    drop_mask = (rng.random(xb.shape, dtype=np.float32) >= dropout_rate).astype(np.float32)
                    xb_in = xb * drop_mask / (1.0 - dropout_rate)
                else:
                    xb_in = xb
                xb_2d = xb_in.reshape(-1, d)

                h_pre = (np.matmul(xb_2d, w1) + b1).reshape(bsz, 4, hidden_dim)
                h = _gelu(h_pre)
                h_mean = np.mean(h, axis=1, keepdims=True)
                h_centered = h - h_mean
                h_ctx = h + np.matmul(h_centered.reshape(-1, hidden_dim), w_peer).reshape(bsz, 4, hidden_dim)
                logits = (
                    np.matmul(h_ctx.reshape(-1, hidden_dim), w2).reshape(bsz, 4)
                    + np.matmul(xb_2d, w_skip).reshape(bsz, 4)
                    + b2
                )

                logits_masked = np.where(vb, logits, -20.0)
                p = _softmax_4(logits_masked)
                p_corr = p * yb
                q = p_corr / np.maximum(np.sum(p_corr, axis=1, keepdims=True), 1e-12)

                grad_logits = ((p - q) * vb.astype(np.float32)) / float(bsz)
                grad_logits_2d = grad_logits.reshape(-1, 1)

                grad_w2 = np.matmul(h_ctx.reshape(-1, hidden_dim).T, grad_logits_2d)
                grad_b2 = np.sum(grad_logits_2d, axis=0)
                grad_w_skip = np.matmul(xb_2d.T, grad_logits_2d)

                grad_h_ctx = np.matmul(grad_logits_2d, w2.T).reshape(bsz, 4, hidden_dim)
                grad_w_peer = np.matmul(h_centered.reshape(-1, hidden_dim).T, grad_h_ctx.reshape(-1, hidden_dim))

                grad_h_centered = np.matmul(grad_h_ctx.reshape(-1, hidden_dim), w_peer.T).reshape(bsz, 4, hidden_dim)
                grad_h = grad_h_ctx + (grad_h_centered - np.mean(grad_h_centered, axis=1, keepdims=True))
                grad_h_pre = (grad_h * _gelu_grad(h_pre)).reshape(-1, hidden_dim)

                grad_w1 = np.matmul(xb_2d.T, grad_h_pre)
                grad_b1 = np.sum(grad_h_pre, axis=0)

                grads = [grad_w1, grad_b1, grad_w_peer, grad_w2, grad_b2, grad_w_skip]
                cur_lr = lr * 0.5 * (1.0 + np.cos(np.pi * step / total_steps))

                for p_idx, (param, grad) in enumerate(zip(params, grads)):
                    if p_idx in (0, 5):
                        grad = grad + wd_vec * param
                    elif param.ndim >= 2:
                        grad = grad + weight_decay * param
                    m_state[p_idx] = beta1 * m_state[p_idx] + (1.0 - beta1) * grad
                    v_state[p_idx] = beta2 * v_state[p_idx] + (1.0 - beta2) * (grad ** 2)
                    m_hat = m_state[p_idx] / (1.0 - beta1 ** step)
                    v_hat = v_state[p_idx] / (1.0 - beta2 ** step)
                    param -= cur_lr * m_hat / (np.sqrt(v_hat) + eps)

    return w1, b1, w_peer, w2, b2, w_skip


def evaluate_accuracy(choices: list[int], y_true: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Compute percentage accuracy on all questions or a boolean subset mask."""
    chosen_corr = np.array([y_true[i, c] for i, c in enumerate(choices)], dtype=np.float32)
    if mask is not None and np.any(mask):
        return float(100.0 * np.mean(chosen_corr[mask]))
    return float(100.0 * np.mean(chosen_corr))


def train_and_save(
    data_dir: Path = DATA,
    output_ckpt: Path = Path('runs/model_bundle.npz'),
    k_diff: int = 96,
    k_raw: int = 32,
    cluster_dim: int = 32,
    n_folds: int = 5,
    hidden_dim: int = 48,
    seed: int = 2026,
) -> dict[str, Any]:
    """Run full Optimal Synthesis training on `train`, tune on `validation`, save checkpoint."""
    t0 = time.perf_counter()
    check_data(data_dir, ('train', 'validation'))

    train_bank, train_arrays = load_inputs('train', data_dir)
    val_bank, val_arrays = load_inputs('validation', data_dir)
    train_refs = read_jsonl(Path(data_dir) / 'gsm8k_train.jsonl')
    val_refs = read_jsonl(Path(data_dir) / 'gsm8k_validation.jsonl')

    print(f"[1/5] Extracting 54D symbolic & multi-term arithmetic features ({len(train_bank)} train, {len(val_bank)} val)...")
    tab_tr, valid_tr, ans_tr = extract_tabular_and_meta(train_bank, train_arrays['embeddings'])
    tab_va, valid_va, ans_va = extract_tabular_and_meta(val_bank, val_arrays['embeddings'])
    se_tr, cf_tr, _ = compute_step_completion_signals(train_bank, tab_tr, ans_tr)
    se_va, cf_va, _ = compute_step_completion_signals(val_bank, tab_va, ans_va)

    y_tr = compute_correctness_matrix(train_bank, train_refs, ans_tr)
    y_va = compute_correctness_matrix(val_bank, val_refs, ans_va)
    hard_tr = identify_hard_groups(ans_tr)
    hard_va = identify_hard_groups(ans_va)

    print(f"      Train hard groups (>=2 distinct answers): {int(np.sum(hard_tr))}/{len(train_bank)}")
    print(f"      Val hard groups   (>=2 distinct answers): {int(np.sum(hard_va))}/{len(val_bank)}")

    print(f"[2/5] Fitting spectrally-regularized within-group PCA (k_diff={k_diff}, k_raw={k_raw})...")
    diff_tr_u, raw_tr_u = compute_normalized_embeddings(train_arrays['embeddings'])
    pca_diff_mean, pca_diff_basis = fit_pca_basis(diff_tr_u.reshape(-1, 1024), k_diff)
    pca_raw_mean, pca_raw_basis = fit_pca_basis(raw_tr_u.reshape(-1, 1024), k_raw)

    x_tr_raw = build_model_features(
        tab_tr, train_arrays['embeddings'], ans_tr,
        pca_diff_mean, pca_diff_basis, pca_raw_mean, pca_raw_basis, cluster_dim=cluster_dim
    )
    x_va_raw = build_model_features(
        tab_va, val_arrays['embeddings'], ans_va,
        pca_diff_mean, pca_diff_basis, pca_raw_mean, pca_raw_basis, cluster_dim=cluster_dim
    )

    feat_mean = np.mean(x_tr_raw, axis=(0, 1), keepdims=True).astype(np.float32)
    feat_std = np.std(x_tr_raw, axis=(0, 1), keepdims=True).astype(np.float32)
    feat_std = np.where(feat_std < 1e-6, 1.0, feat_std).astype(np.float32)

    x_tr = np.clip((x_tr_raw - feat_mean) / feat_std, -8.0, 8.0)
    x_va = np.clip((x_va_raw - feat_mean) / feat_std, -8.0, 8.0)
    d_in = x_tr.shape[-1]

    idx_sym_only = list(range(NUM_TAB_FEATURES))
    idx_emb_only = IDX_EMB_STAT + list(range(NUM_TAB_FEATURES, d_in))

    print(f"[3/5] Training {n_folds}-fold Optimal Synthesis Ensemble (D={d_in})...")
    rng = np.random.default_rng(seed)
    indices = np.arange(len(train_bank))
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    w_linear_list, b_linear_list = [], []
    w_sym_list, b_sym_list = [], []
    w_emb_list, b_emb_list = [], []
    w1_list, b1_list, w_peer_list, w2_list, b2_list, w_skip_list = [], [], [], [], [], []

    oof_lin = np.zeros((len(train_bank), 4), dtype=np.float32)
    oof_sym = np.zeros((len(train_bank), 4), dtype=np.float32)
    oof_emb = np.zeros((len(train_bank), 4), dtype=np.float32)
    oof_mlp = np.zeros((len(train_bank), 4), dtype=np.float32)

    for fold_idx in range(n_folds):
        val_idx = folds[fold_idx]
        tr_mask = np.ones(len(train_bank), dtype=bool)
        tr_mask[val_idx] = False

        # 1. Joint Vote-Damped Listwise Linear
        w_l, b_l = train_listwise_linear(
            x_tr[tr_mask], y_tr[tr_mask], valid_tr[tr_mask],
            l2_tab=0.06, l2_emb=0.35, n_tab=NUM_TAB_FEATURES, steps=300, lr=0.12, l2_vote=1.5
        )
        w_linear_list.append(w_l)
        b_linear_list.append(b_l)
        oof_lin[val_idx] = _safe_matmul(x_tr[val_idx], w_l) + b_l

        # 2. Pure 54D Symbolic Specialist
        w_s, b_s = train_listwise_linear(
            x_tr[tr_mask][:, :, idx_sym_only], y_tr[tr_mask], valid_tr[tr_mask],
            l2_tab=0.06, l2_emb=0.06, n_tab=NUM_TAB_FEATURES, steps=280, lr=0.12
        )
        w_sym_list.append(w_s)
        b_sym_list.append(b_s)
        oof_sym[val_idx] = _safe_matmul(x_tr[val_idx][:, :, idx_sym_only], w_s) + b_s

        # 3. Pure 168D Embedding Specialist
        w_e, b_e = train_listwise_linear(
            x_tr[tr_mask][:, :, idx_emb_only], y_tr[tr_mask], valid_tr[tr_mask],
            l2_tab=0.08, l2_emb=0.35, n_tab=8, steps=280, lr=0.12
        )
        w_emb_list.append(w_e)
        b_emb_list.append(b_e)
        oof_emb[val_idx] = _safe_matmul(x_tr[val_idx][:, :, idx_emb_only], w_e) + b_e

        # 4. Dual-Regularized Group-Interaction Peer-MLP (Consensus-Sensitive + Vote-Damped)
        m_a = train_listwise_mlp(
            x_tr[tr_mask], y_tr[tr_mask], valid_tr[tr_mask],
            hidden_dim=hidden_dim, epochs=32, batch_size=256,
            lr=3.0e-3, weight_decay=1.5e-2, wd_emb=7.0e-2, wd_vote=None, dropout_rate=0.22,
            seed=seed + 101 * fold_idx
        )
        m_b = train_listwise_mlp(
            x_tr[tr_mask], y_tr[tr_mask], valid_tr[tr_mask],
            hidden_dim=hidden_dim, epochs=32, batch_size=256,
            lr=3.0e-3, weight_decay=1.5e-2, wd_emb=7.0e-2, wd_vote=4.0e-2, dropout_rate=0.22,
            seed=seed + 101 * fold_idx
        )
        for m_item in (m_a, m_b):
            w1_list.append(m_item[0])
            b1_list.append(m_item[1])
            w_peer_list.append(m_item[2])
            w2_list.append(m_item[3])
            b2_list.append(m_item[4])
            w_skip_list.append(m_item[5])
        oof_mlp[val_idx] = 0.5 * (
            forward_mlp_member(x_tr[val_idx], *m_a) + forward_mlp_member(x_tr[val_idx], *m_b)
        )

    w_linear_arr = np.stack(w_linear_list, axis=0).astype(np.float32)
    b_linear_arr = np.array(b_linear_list, dtype=np.float32)
    w_sym_arr = np.stack(w_sym_list, axis=0).astype(np.float32)
    b_sym_arr = np.array(b_sym_list, dtype=np.float32)
    w_emb_arr = np.stack(w_emb_list, axis=0).astype(np.float32)
    b_emb_arr = np.array(b_emb_list, dtype=np.float32)

    w1_arr = np.stack(w1_list, axis=0).astype(np.float32)
    b1_arr = np.stack(b1_list, axis=0).astype(np.float32)
    w_peer_arr = np.stack(w_peer_list, axis=0).astype(np.float32)
    w2_arr = np.stack(w2_list, axis=0).astype(np.float32)
    b2_arr = np.stack(b2_list, axis=0).astype(np.float32)
    w_skip_arr = np.stack(w_skip_list, axis=0).astype(np.float32)

    va_lin = _safe_fold_linear(x_va, w_linear_arr, b_linear_arr)
    va_sym = _safe_fold_linear(x_va[:, :, idx_sym_only], w_sym_arr, b_sym_arr)
    va_emb = _safe_fold_linear(x_va[:, :, idx_emb_only], w_emb_arr, b_emb_arr)
    va_mlp = np.mean(
        np.stack([
            forward_mlp_member(
                x_va, w1_arr[k], b1_arr[k], w_peer_arr[k], w2_arr[k], b2_arr[k], w_skip_arr[k]
            )
            for k in range(len(w1_arr))
        ], axis=0),
        axis=0,
    )
    va_heur = compute_step_structural_prior(tab_va, se_va, cf_va, valid_va)
    oof_heur = compute_step_structural_prior(tab_tr, se_tr, cf_tr, valid_tr)

    print("[4/5] Tuning ensemble blending, cluster aggregation & calibration on validation...")
    best_score = -1e9
    best_cfg = (0.30, 0.70, 1.00, 0.0, 0.0, 0.65, 0.75)

    # Candidate blends prioritizing Dual-Regularized Peer-MLP + Vote-Damped Linear + Step-Completion Prior
    blend_candidates = [
        (0.30, 0.70, 1.00, 0.00, 0.00),
        (0.30, 0.75, 1.00, 0.00, 0.00),
        (0.30, 0.70, 0.85, 0.00, 0.00),
        (0.30, 0.70, 0.00, 0.00, 0.00),
    ]
    for w_l_c, w_m_c, w_h_c, w_s_c, w_e_c in blend_candidates:
        blend_va = w_l_c * va_lin + w_m_c * va_mlp + w_h_c * va_heur + w_s_c * va_sym + w_e_c * va_emb
        blend_oof = w_l_c * oof_lin + w_m_c * oof_mlp + w_h_c * oof_heur + w_s_c * oof_sym + w_e_c * oof_emb
        for alpha_c in (0.65, 0.70):
            for temp_c in (0.75, 0.80):
                va_choices, va_post = select_with_cluster_aggregation(
                    blend_va, valid_va, ans_va, cluster_alpha=alpha_c, temperature=temp_c
                )
                oof_choices, _ = select_with_cluster_aggregation(
                    blend_oof, valid_tr, ans_tr, cluster_alpha=alpha_c, temperature=temp_c
                )
                va_hard_acc = evaluate_accuracy(va_choices, y_va, hard_va)
                va_all_acc = evaluate_accuracy(va_choices, y_va)
                oof_hard_acc = evaluate_accuracy(oof_choices, y_tr, hard_tr)

                chosen_post = np.array([va_post[i, c] for i, c in enumerate(va_choices)], dtype=np.float32)
                chosen_corr = np.array([y_va[i, c] for i, c in enumerate(va_choices)], dtype=np.float32)
                brier = float(np.mean((chosen_post - chosen_corr) ** 2))

                composite = 0.65 * va_hard_acc + 0.20 * va_all_acc + 0.15 * oof_hard_acc - 1.5 * brier
                if composite > best_score:
                    best_score = composite
                    best_cfg = (w_l_c, w_m_c, w_h_c, w_s_c, w_e_c, alpha_c, temp_c)

    linear_weight, mlp_weight, heur_weight, sym_spec_weight, emb_spec_weight, cluster_alpha, temperature = best_cfg
    print(
        f"      Selected weights: lin={linear_weight:.2f}, mlp={mlp_weight:.2f}, "
        f"heur={heur_weight:.2f}, sym_spec={sym_spec_weight:.2f}, emb_spec={emb_spec_weight:.2f}, "
        f"alpha={cluster_alpha:.2f}, temp={temperature:.2f}"
    )

    output_ckpt = Path(output_ckpt)
    if output_ckpt.suffix != '.npz':
        output_ckpt.mkdir(parents=True, exist_ok=True)
        ckpt_file = output_ckpt / 'model_bundle.npz'
    else:
        output_ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt_file = output_ckpt

    arrays_to_save = {
        'version_tag': np.array([3], dtype=np.int32),
        'pca_diff_mean': pca_diff_mean,
        'pca_diff_basis': pca_diff_basis,
        'pca_raw_mean': pca_raw_mean,
        'pca_raw_basis': pca_raw_basis,
        'cluster_dim': np.array([cluster_dim], dtype=np.int32),
        'feat_mean': feat_mean,
        'feat_std': feat_std,
        'w_linear': w_linear_arr,
        'b_linear': b_linear_arr,
        'w_sym_spec': w_sym_arr,
        'b_sym_spec': b_sym_arr,
        'w_emb_spec': w_emb_arr,
        'b_emb_spec': b_emb_arr,
        'linear_weight': np.array([linear_weight], dtype=np.float32),
        'mlp_weight': np.array([mlp_weight], dtype=np.float32),
        'heur_weight': np.array([heur_weight], dtype=np.float32),
        'sym_spec_weight': np.array([sym_spec_weight], dtype=np.float32),
        'emb_spec_weight': np.array([emb_spec_weight], dtype=np.float32),
        'w1_ens': w1_arr,
        'b1_ens': b1_arr,
        'w_peer_ens': w_peer_arr,
        'w2_ens': w2_arr,
        'b2_ens': b2_arr,
        'w_skip_ens': w_skip_arr,
        'cluster_alpha': np.array([cluster_alpha], dtype=np.float32),
        'temperature': np.array([temperature], dtype=np.float32),
    }
    np.savez(ckpt_file, **arrays_to_save)

    uncompressed_bytes = sum(arr.nbytes for arr in arrays_to_save.values())
    file_bytes = ckpt_file.stat().st_size
    sha256_hash = hashlib.sha256(ckpt_file.read_bytes()).hexdigest()

    final_va_logits = (
        linear_weight * va_lin + mlp_weight * va_mlp + heur_weight * va_heur
        + sym_spec_weight * va_sym + emb_spec_weight * va_emb
    )
    va_choices, _ = select_with_cluster_aggregation(
        final_va_logits, valid_va, ans_va, cluster_alpha=cluster_alpha, temperature=temperature
    )
    final_oof_logits = (
        linear_weight * oof_lin + mlp_weight * oof_mlp + heur_weight * oof_heur
        + sym_spec_weight * oof_sym + emb_spec_weight * oof_emb
    )
    oof_choices, _ = select_with_cluster_aggregation(
        final_oof_logits, valid_tr, ans_tr, cluster_alpha=cluster_alpha, temperature=temperature
    )

    summary = {
        'seed': seed,
        'checkpoint_path': str(ckpt_file),
        'checkpoint_sha256': sha256_hash,
        'uncompressed_asset_bytes': int(uncompressed_bytes),
        'uncompressed_asset_mib': round(uncompressed_bytes / (1024 ** 2), 4),
        'file_size_bytes': int(file_bytes),
        'train_oof_accuracy_all': round(evaluate_accuracy(oof_choices, y_tr), 3),
        'train_oof_accuracy_hard': round(evaluate_accuracy(oof_choices, y_tr, hard_tr), 3),
        'validation_accuracy_all': round(evaluate_accuracy(va_choices, y_va), 3),
        'validation_accuracy_hard': round(evaluate_accuracy(va_choices, y_va, hard_va), 3),
        'training_seconds': round(time.perf_counter() - t0, 3),
    }
    print("[5/5] Training complete! Summary:")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description='Train DASE7506 MP2 Optimal Synthesis Verifier')
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--checkpoint', type=Path, default=Path('runs/model_bundle.npz'))
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    train_and_save(data_dir=args.data_dir, output_ckpt=args.checkpoint, seed=args.seed)


if __name__ == '__main__':
    main()