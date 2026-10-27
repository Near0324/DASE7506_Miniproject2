from __future__ import annotations

from collections import Counter
from fractions import Fraction
import math
from pathlib import Path
import re
from typing import Any

import numpy as np

from common import parse_answer

# ---------------------------------------------------------------------------
# Feature Slice Indices (Total Handcrafted Tabular Features = 54)
# ---------------------------------------------------------------------------
IDX_ANS = list(range(0, 20))       # 20 Answer, Plausibility & Split-Topology features
IDX_SURF = list(range(20, 30))     # 10 Surface & Formatting features
IDX_REAS = list(range(30, 46))     # 16 Intrinsic Reasoning & Multi-Term Arithmetic features
IDX_EMB_STAT = list(range(46, 54)) # 8 Embedding Geometry summary statistics
NUM_TAB_FEATURES = 54

_NUM_RE = re.compile(r'(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?')
_WORD_RE = re.compile(r'[A-Za-z0-9_]+')
_CALC_TAG_RE = re.compile(r'<<\s*([^<>]+?)\s*=\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*>>')
_TAG_STRIP_RE = re.compile(r'<<[^<>]*>>')
_LATEX_FRAC_RE = re.compile(r'\\d?frac\s*\{\s*([^{}]+?)\s*\}\s*\{\s*([^{}]+?)\s*\}')
_BOXED_UNWRAP_RE = re.compile(r'\\boxed\s*\{\s*([^{}]+?)\s*\}')
_TEXT_STRIP_RE = re.compile(r'\\(?:text|mathrm|mathbf|textit)\s*\{[^{}]*\}')
_FULL_EQ_RE = re.compile(
    r'(?<![A-Za-z\d.])((?:(?:\([ \t]*)*[A-Za-z\d][\w,.]*(?:[ \t]*\))*[ \t]*[+\-*/×÷][ \t]*)+'
    r'(?:\([ \t]*)*[A-Za-z\d][\w,.]*(?:[ \t]*\))*)'
    r'[ \t]*=[ \t]*([-+]?\d[\d,]*(?:\.\d+)?)(?![A-Za-z\d,.]|[ \t]*[+\-*/×÷=\d,.])'
)
_REV_EQ_RE = re.compile(
    r'(?<![A-Za-z\d.+-])([-+]?\d[\d,]*(?:\.\d+)?)[ \t]*=[ \t]*'
    r'((?:(?:\([ \t]*)*\d[\d,]*(?:\.\d+)?(?:[ \t]*\))*[ \t]*[+\-*/×÷][ \t]*)+'
    r'(?:\([ \t]*)*\d[\d,]*(?:\.\d+)?(?:[ \t]*\))*)(?![A-Za-z\d,.]|[ \t]*[+\-*/×÷=\d,.])'
)
_BACKTRACK_RE = re.compile(
    r'\b(?:wait|mistake|oops|actually|wrong|recalculate|re-calculate|hmm|error|correction|but\s+wait)\b',
    re.IGNORECASE,
)
_STEP_RE = re.compile(
    r'\b(?:step|first|second|third|then|next|finally|therefore|thus|total|hence|so)\b',
    re.IGNORECASE,
)


def _safe_float(s: str) -> float | None:
    try:
        return float(Fraction(s.replace(',', '').strip()))
    except (ValueError, ZeroDivisionError):
        return None


def _extract_numbers(text: str) -> list[float]:
    out = []
    for m in _NUM_RE.findall(text):
        v = _safe_float(m)
        if v is not None and math.isfinite(v):
            out.append(v)
    return out


def _eval_simple_lhs(expr: str) -> float | None:
    """Safely evaluate multi-term arithmetic expressions with standard precedence."""
    expr = expr.replace(',', '').replace('×', '*').replace('÷', '/').replace('$', '').strip()
    if not expr or len(expr) > 80:
        return None
    if not re.fullmatch(r'[\d.+\-*/()\s]+', expr):
        return None
    if expr.count('(') != expr.count(')'):
        return None
    try:
        val = eval(expr, {'__builtins__': None}, {})  # noqa: S307 (strictly numeric charset)
        if isinstance(val, (int, float)) and math.isfinite(val):
            return float(val)
    except Exception:
        return None
    return None


def _verify_arithmetic(text: str) -> tuple[int, int]:
    """Return (total_equations_checked, num_errors) in a candidate reasoning trace."""
    total_eq = 0
    errors = 0
    seen_pairs: set[tuple[float, float]] = set()

    for lhs_str, rhs_str in _CALC_TAG_RE.findall(text):
        lhs_val = _eval_simple_lhs(lhs_str)
        rhs_val = _safe_float(rhs_str)
        if lhs_val is not None and rhs_val is not None:
            total_eq += 1
            seen_pairs.add((round(lhs_val, 3), round(rhs_val, 3)))
            tol = max(1.5e-2, 1.5e-2 * abs(rhs_val))
            if abs(lhs_val - rhs_val) > tol:
                errors += 1

    s = _TAG_STRIP_RE.sub('', text)
    for _ in range(2):
        s = _BOXED_UNWRAP_RE.sub(r' \1 ', s)
        s = _LATEX_FRAC_RE.sub(r'((\1)/(\2))', s)
    s = _TEXT_STRIP_RE.sub(' ', s)
    clean_text = (
        s.replace('\\times', ' * ')
        .replace('\\cdot', ' * ')
        .replace('\\div', ' / ')
        .replace('\\left', '')
        .replace('\\right', '')
        .replace('\\$', '')
        .replace('$', '')
        .replace('**', '')
        .replace('\\,', '')
        .replace('\\ ', ' ')
    )
    for lhs_str, rhs_str in _FULL_EQ_RE.findall(clean_text):
        lhs_val = _eval_simple_lhs(lhs_str)
        rhs_val = _safe_float(rhs_str)
        if lhs_val is None or rhs_val is None:
            continue
        key = (round(lhs_val, 3), round(rhs_val, 3))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        total_eq += 1
        tol = max(1.5e-2, 1.5e-2 * abs(rhs_val))
        if abs(lhs_val - rhs_val) > tol:
            errors += 1

    for lhs_num, rhs_expr in _REV_EQ_RE.findall(clean_text):
        rhs_val = _eval_simple_lhs(rhs_expr)
        lhs_val = _safe_float(lhs_num)
        if lhs_val is None or rhs_val is None:
            continue
        key = (round(rhs_val, 3), round(lhs_val, 3))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        total_eq += 1
        tol = max(1.5e-2, 1.5e-2 * abs(lhs_val))
        if abs(lhs_val - rhs_val) > tol:
            errors += 1

    return total_eq, errors


def _safe_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Numerically stable C-contiguous float64 GEMM immune to macOS Accelerate float32 AMX warnings."""
    a64 = np.ascontiguousarray(np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float64)
    b64 = np.ascontiguousarray(np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float64)
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        if a64.ndim == 3 and b64.ndim == 2:
            n0, n1, d = a64.shape
            out = np.matmul(a64.reshape(n0 * n1, d), b64).reshape(n0, n1, b64.shape[1])
        else:
            out = np.matmul(a64, b64)
    return np.ascontiguousarray(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float32)


def _safe_fold_linear(x_3d: np.ndarray, w_2d: np.ndarray, b_1d: np.ndarray) -> np.ndarray:
    """Compute mean over K folds of (x_3d @ w_2d.T + b_1d) using float64 GEMM."""
    proj = _safe_matmul(x_3d, w_2d.T)  # [N, 4, K]
    return np.mean(np.transpose(proj, (2, 0, 1)) + b_1d[:, None, None], axis=0)


def _zscore_group(arr: np.ndarray) -> np.ndarray:
    """Z-score a 1D array of length 4 within a single question group."""
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if std < 1e-7:
        return np.zeros_like(arr)
    return (arr - mean) / std


def extract_tabular_and_meta(
    bank: list[dict[str, Any]],
    embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[list[str | None]]]:
    """Extract [N, 4, 54] handcrafted features, [N, 4] valid masks, and parsed answers."""
    n = len(bank)
    feats = np.zeros((n, 4, NUM_TAB_FEATURES), dtype=np.float32)
    valid_mask = np.zeros((n, 4), dtype=bool)
    parsed_answers: list[list[str | None]] = []

    emb = embeddings.astype(np.float32, copy=False)
    emb_norms = np.linalg.norm(emb, axis=-1)  # [N, 4]
    emb_unit = emb / np.maximum(emb_norms[..., None], 1e-7)

    group_mean = np.mean(emb, axis=1, keepdims=True)  # [N, 1, 1024]
    group_mean_norm = np.linalg.norm(group_mean, axis=-1)  # [N, 1]
    group_mean_unit = group_mean / np.maximum(group_mean_norm[..., None], 1e-7)
    cos_to_mean = np.sum(emb_unit * group_mean_unit, axis=-1)  # [N, 4]

    diff = emb - group_mean  # [N, 4, 1024]
    diff_norms = np.linalg.norm(diff, axis=-1)  # [N, 4]
    diff_unit = diff / np.maximum(diff_norms[..., None], 1e-7)
    diff_sim_matrix = _safe_matmul(diff_unit, np.transpose(diff_unit, (0, 2, 1)))  # [N, 4, 4]

    for i, row in enumerate(bank):
        q_text = str(row.get('question', ''))
        q_nums = _extract_numbers(q_text)
        q_nums_pos = [abs(x) for x in q_nums if abs(x) > 0]
        q_max_log = math.log10(max(q_nums_pos) + 1.0) if q_nums_pos else 1.5
        q_min_log = math.log10(min(q_nums_pos) + 1.0) if q_nums_pos else 0.5
        q_num_set = {round(x, 4) for x in q_nums}

        cands = row['candidates']
        ans_list: list[str | None] = []
        val_list: list[float | None] = []
        frac_list: list[Fraction | None] = []

        char_lens = np.zeros(4, dtype=np.float32)
        word_counts = np.zeros(4, dtype=np.float32)
        line_counts = np.zeros(4, dtype=np.float32)
        q_covs = np.zeros(4, dtype=np.float32)
        eq_acc_scores = np.zeros(4, dtype=np.float32)

        for c_idx in range(4):
            text = str(cands[c_idx].get('text', ''))
            ans_str = parse_answer(text)
            ans_list.append(ans_str)
            if ans_str is not None:
                valid_mask[i, c_idx] = True
                try:
                    fr = Fraction(ans_str)
                    fl = float(fr)
                    if not math.isfinite(fl):
                        fr, fl = None, None
                except (ValueError, ZeroDivisionError, OverflowError):
                    fr, fl = None, None
            else:
                fr, fl = None, None
            frac_list.append(fr)
            val_list.append(fl)

            # Surface features (20..29)
            c_len = len(text)
            words = _WORD_RE.findall(text.lower())
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            char_lens[c_idx] = float(c_len)
            word_counts[c_idx] = float(len(words))
            line_counts[c_idx] = float(len(lines))

            feats[i, c_idx, 20] = min(c_len, 8000) / 1000.0
            feats[i, c_idx, 22] = min(len(words), 1500) / 200.0
            feats[i, c_idx, 24] = min(len(lines), 100) / 15.0
            has_hash = '####' in text
            has_boxed = '\\boxed{' in text
            feats[i, c_idx, 26] = 1.0 if has_hash else 0.0
            feats[i, c_idx, 27] = 1.0 if has_boxed else 0.0

            if has_hash:
                body, tail = text.rsplit('####', 1)
                feats[i, c_idx, 28] = 1.0 if len(tail.strip()) < 35 else 0.0
            elif has_boxed:
                parts = text.rsplit('\\boxed{', 1)
                body, tail = parts[0], parts[-1]
                feats[i, c_idx, 28] = 1.0 if len(tail.strip()) < 45 else 0.0
            else:
                body = text
                feats[i, c_idx, 28] = 0.0

            digit_cnt = sum(ch.isdigit() for ch in text)
            feats[i, c_idx, 29] = digit_cnt / max(c_len, 1)

            # Intrinsic Reasoning & Multi-Term Arithmetic Verification (30..45)
            tot_eq, err_eq = _verify_arithmetic(body)
            eq_acc = (tot_eq - err_eq) / max(tot_eq, 1) if tot_eq > 0 else 0.85
            feats[i, c_idx, 30] = min(tot_eq, 15) / 10.0
            feats[i, c_idx, 31] = float(eq_acc)
            feats[i, c_idx, 32] = 1.0 if err_eq > 0 else 0.0
            feats[i, c_idx, 33] = min(err_eq, 5) / 5.0
            feats[i, c_idx, 34] = min(max(tot_eq - err_eq, 0), 12) / 8.0
            eq_acc_scores[c_idx] = float(eq_acc) * (0.3 if err_eq > 0 else 1.0)

            body_nums = _extract_numbers(body)
            body_num_set = {round(x, 4) for x in body_nums}
            if q_num_set:
                cov = sum(1 for qn in q_num_set if qn in body_num_set) / len(q_num_set)
            else:
                cov = 1.0
            q_covs[c_idx] = float(cov)
            feats[i, c_idx, 35] = float(cov)

            if fl is not None and body_nums:
                tol = max(1e-2, 1e-3 * abs(fl))
                feats[i, c_idx, 37] = 1.0 if any(abs(bn - fl) <= tol for bn in body_nums) else 0.0
                feats[i, c_idx, 38] = 1.0 if abs(body_nums[-1] - fl) <= tol else 0.0

            if lines:
                feats[i, c_idx, 39] = len(set(lines)) / len(lines)
            else:
                feats[i, c_idx, 39] = 0.0

            if len(words) >= 4:
                ngrams = [tuple(words[j:j + 4]) for j in range(len(words) - 3)]
                feats[i, c_idx, 40] = len(set(ngrams)) / len(ngrams)
            else:
                feats[i, c_idx, 40] = 1.0

            bt_cnt = len(_BACKTRACK_RE.findall(text))
            feats[i, c_idx, 41] = min(bt_cnt, 8) / 5.0
            feats[i, c_idx, 42] = min(len(_STEP_RE.findall(text)), 15) / 10.0

        parsed_answers.append(ans_list)

        feats[i, :, 21] = _zscore_group(char_lens)
        feats[i, :, 23] = _zscore_group(word_counts)
        feats[i, :, 25] = _zscore_group(line_counts)
        feats[i, :, 36] = q_covs - np.mean(q_covs)
        feats[i, :, 43] = eq_acc_scores - np.mean(eq_acc_scores)

        valid_ans = [a for a in ans_list if a is not None]
        counts = Counter(valid_ans)
        num_valid = len(valid_ans)
        sorted_counts = sorted(counts.values(), reverse=True)
        max_vote = sorted_counts[0] if sorted_counts else 0
        num_top_answers = sum(1 for k, v in counts.items() if v == max_vote) if max_vote > 0 else 0

        is_3_1 = 1.0 if sorted_counts == [3, 1] else 0.0
        is_2_2 = 1.0 if sorted_counts == [2, 2] else 0.0
        is_2_1_1 = 1.0 if sorted_counts == [2, 1, 1] else 0.0
        is_1_1_1_1 = 1.0 if sorted_counts == [1, 1, 1, 1] else 0.0

        if num_valid > 1:
            ent = -sum((v / num_valid) * math.log(v / num_valid + 1e-12) for v in counts.values())
            norm_ent = ent / math.log(4.0)
        else:
            norm_ent = 0.0

        valid_logs = [
            math.log10(min(abs(v), 1e12) + 1.0)
            for v in val_list
            if v is not None
        ]
        median_log = float(np.median(valid_logs)) if valid_logs else q_max_log

        z_emb_norm = _zscore_group(emb_norms[i])
        z_diff_norm = _zscore_group(diff_norms[i])
        z_cos_mean = _zscore_group(cos_to_mean[i])

        for c_idx in range(4):
            a_str = ans_list[c_idx]
            fr = frac_list[c_idx]
            fl = val_list[c_idx]

            feats[i, c_idx, 14] = norm_ent
            if a_str is not None and fr is not None and fl is not None:
                v_cnt = counts[a_str]
                rival_max = max((cnt for ans_k, cnt in counts.items() if ans_k != a_str), default=0)
                vote_margin = (v_cnt - rival_max) / 4.0

                log_v = math.log10(min(abs(fl), 1e12) + 1.0)
                feats[i, c_idx, 0] = 1.0
                feats[i, c_idx, 1] = 1.0 if fr.denominator == 1 else 0.0
                feats[i, c_idx, 2] = 1.0 if fl > 0 else 0.0
                feats[i, c_idx, 3] = 1.0 if fl >= 0 else 0.0
                feats[i, c_idx, 4] = 1.0 if fl == 0 else 0.0
                feats[i, c_idx, 5] = 1.0 if round(fl, 4) in q_num_set else 0.0
                feats[i, c_idx, 6] = log_v / 10.0
                feats[i, c_idx, 7] = min(abs(log_v - q_max_log) / 5.0, 3.0)
                feats[i, c_idx, 8] = min(abs(log_v - q_min_log) / 5.0, 3.0)
                feats[i, c_idx, 9] = v_cnt / 4.0
                feats[i, c_idx, 10] = v_cnt / max(num_valid, 1)
                feats[i, c_idx, 11] = 1.0 if (v_cnt == max_vote and num_top_answers == 1) else 0.0
                feats[i, c_idx, 12] = 1.0 if (v_cnt == max_vote and num_top_answers > 1) else 0.0
                feats[i, c_idx, 13] = 1.0 if v_cnt < max_vote else 0.0
                feats[i, c_idx, 15] = min(abs(log_v - median_log) / 5.0, 3.0)
                feats[i, c_idx, 16] = vote_margin
                feats[i, c_idx, 17] = is_3_1 * (1.0 if v_cnt == 3 else -1.0)
                feats[i, c_idx, 18] = is_2_1_1 * (1.0 if v_cnt == 2 else -0.5)
                feats[i, c_idx, 19] = 1.0 if (is_2_2 > 0.5 or is_1_1_1_1 > 0.5) else 0.0

                same_idxs = [k for k in range(4) if ans_list[k] == a_str]
                feats[i, c_idx, 44] = float(np.mean(eq_acc_scores[same_idxs]))
                feats[i, c_idx, 45] = float(np.mean(q_covs[same_idxs]))
                feats[i, c_idx, 53] = float(np.mean(-z_diff_norm[same_idxs]))

                other_same = [k for k in same_idxs if k != c_idx]
                if other_same:
                    feats[i, c_idx, 52] = float(np.mean(diff_sim_matrix[i, c_idx, other_same]))
                else:
                    feats[i, c_idx, 52] = 0.0

            feats[i, c_idx, 46] = float(emb_norms[i, c_idx]) / 60.0
            feats[i, c_idx, 47] = float(z_emb_norm[c_idx])
            feats[i, c_idx, 48] = float(diff_norms[i, c_idx]) / 10.0
            feats[i, c_idx, 49] = float(z_diff_norm[c_idx])
            feats[i, c_idx, 50] = float(z_cos_mean[c_idx])
            peers = [k for k in range(4) if k != c_idx]
            feats[i, c_idx, 51] = float(np.mean(diff_sim_matrix[i, c_idx, peers]))

    return feats, valid_mask, parsed_answers


def compute_step_completion_signals(
    bank: list[dict[str, Any]],
    tab_feats: np.ndarray,
    parsed_answers: list[list[str | None]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute cluster-pooled step-completion asymmetry (se_cl, cf_cl) and process consensus (ps_arr).

    Detects when Candidate c stopped at an intermediate calculation step whose result `my_a`
    appears inside an error-free peer k's derivation (`my_a in derived_sets[k]`) while peer k
    continued to a different final answer (`ans_vals[k] not in derived_sets[c]`).
    """
    n = len(bank)
    se_arr = np.zeros((n, 4), dtype=np.float32)
    cf_arr = np.zeros((n, 4), dtype=np.float32)
    ps_arr = np.zeros((n, 4), dtype=np.float32)

    for i, row in enumerate(bank):
        q_nums = {round(x, 2) for x in _extract_numbers(str(row.get('question', '')))}
        derived_sets: list[set[float]] = []
        ans_vals: list[float | None] = []
        qual_ok = [
            (tab_feats[i, c_idx, 32] < 0.5)
            and (tab_feats[i, c_idx, 1] > 0.5)
            and (tab_feats[i, c_idx, 2] > 0.5)
            for c_idx in range(4)
        ]
        for c_idx in range(4):
            txt = str(row['candidates'][c_idx].get('text', ''))
            body = txt.rsplit('####', 1)[0] if '####' in txt else txt
            bnums = {round(x, 2) for x in _extract_numbers(body)}
            derived_sets.append(bnums - q_nums)
            try:
                ans_vals.append(
                    round(float(parsed_answers[i][c_idx]), 2)
                    if parsed_answers[i][c_idx] is not None
                    else None
                )
            except Exception:
                ans_vals.append(None)

        ps = np.zeros(4, dtype=np.float32)
        for c_idx in range(4):
            d_c = derived_sets[c_idx]
            if d_c:
                ps[c_idx] = (
                    sum(
                        sum(1 for k in range(4) if k != c_idx and num in derived_sets[k]) / 3.0
                        for num in d_c
                    )
                    / len(d_c)
                )
            my_a = ans_vals[c_idx]
            if my_a is not None and my_a not in q_nums and abs(my_a) > 3.0:
                for k in range(4):
                    if k != c_idx and ans_vals[k] is not None and ans_vals[k] != my_a:
                        if qual_ok[k] and my_a in derived_sets[k] and ans_vals[k] not in derived_sets[c_idx]:
                            se_arr[i, c_idx] += 1.0 / 3.0
                        if (
                            qual_ok[c_idx]
                            and abs(ans_vals[k]) > 3.0
                            and ans_vals[k] in derived_sets[c_idx]
                            and my_a not in derived_sets[k]
                        ):
                            cf_arr[i, c_idx] += 1.0 / 3.0
        ps_arr[i] = ps - np.mean(ps)

    se_cl = np.zeros_like(se_arr)
    cf_cl = np.zeros_like(cf_arr)
    for i in range(n):
        for c_idx in range(4):
            a_s = parsed_answers[i][c_idx]
            if a_s is not None:
                same = [k for k in range(4) if parsed_answers[i][k] == a_s]
                se_cl[i, c_idx] = float(np.mean(se_arr[i, same]))
                cf_cl[i, c_idx] = float(np.mean(cf_arr[i, same]))
            else:
                se_cl[i, c_idx] = se_arr[i, c_idx]
                cf_cl[i, c_idx] = cf_arr[i, c_idx]
    return se_cl, cf_cl, ps_arr


def compute_step_structural_prior(
    tab_feats: np.ndarray,
    se_cl: np.ndarray,
    cf_cl: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Compute Step-Completion Asymmetry & Positive-Integer Structural Invariant logits."""
    is_pos_int = (tab_feats[:, :, 1] > 0.5) & (tab_feats[:, :, 2] > 0.5)
    pos_int_logit = np.where(is_pos_int, 0.0, -4.0).astype(np.float32)
    non31 = (np.abs(tab_feats[:, :, 17]) < 0.5).astype(np.float32)
    step_logit = 1.05 * cf_cl - (0.15 + 1.05 * non31) * se_cl
    return np.where(valid_mask, pos_int_logit + step_logit, -1e9)


def compute_normalized_embeddings(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return unit-normalized within-group centered residuals and raw embeddings."""
    emb = embeddings.astype(np.float32, copy=False)
    group_mean = np.mean(emb, axis=1, keepdims=True)
    diff = emb - group_mean
    diff_norm = np.linalg.norm(diff, axis=-1, keepdims=True)
    diff_unit = diff / np.maximum(diff_norm, 1e-6)

    raw_norm = np.linalg.norm(emb, axis=-1, keepdims=True)
    raw_unit = emb / np.maximum(raw_norm, 1e-6)
    return diff_unit, raw_unit


def build_model_features(
    tab_feats: np.ndarray,
    embeddings: np.ndarray,
    parsed_answers: list[list[str | None]],
    pca_diff_mean: np.ndarray,
    pca_diff_basis: np.ndarray,
    pca_raw_mean: np.ndarray,
    pca_raw_basis: np.ndarray,
    cluster_dim: int = 32,
) -> np.ndarray:
    """Combine 54 tabular features with 160 spectrally-regularized PCA embedding features (D=214)."""
    n = tab_feats.shape[0]
    diff_unit, raw_unit = compute_normalized_embeddings(embeddings)

    z_diff = _safe_matmul(diff_unit - pca_diff_mean, pca_diff_basis)
    z_raw = _safe_matmul(raw_unit - pca_raw_mean, pca_raw_basis)

    c_dim = min(cluster_dim, z_diff.shape[-1])
    z_cluster = np.zeros((n, 4, c_dim), dtype=np.float32)
    for i in range(n):
        ans_i = parsed_answers[i]
        for c_idx in range(4):
            a_str = ans_i[c_idx]
            if a_str is None:
                z_cluster[i, c_idx] = z_diff[i, c_idx, :c_dim]
            else:
                same = [k for k in range(4) if ans_i[k] == a_str]
                z_cluster[i, c_idx] = np.mean(z_diff[i, same, :c_dim], axis=0)

    return np.concatenate([tab_feats, z_diff, z_cluster, z_raw], axis=-1)


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * (x ** 3))))


def _softmax_4(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_x = np.exp(np.clip(shifted, -50.0, 50.0))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def forward_mlp_member(
    x_norm: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w_peer: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    w_skip: np.ndarray,
) -> np.ndarray:
    """Forward pass of a single Listwise Group-Interaction MLP member on [N, 4, D]."""
    h_pre = _safe_matmul(x_norm, w1) + b1
    h = _gelu(h_pre)
    h_group_mean = np.mean(h, axis=1, keepdims=True)
    h_ctx = h + _safe_matmul(h - h_group_mean, w_peer)
    logits = (
        _safe_matmul(h_ctx, w2).squeeze(-1)
        + _safe_matmul(x_norm, w_skip).squeeze(-1)
        + b2
    )
    return logits


def compute_answer_cluster_posteriors(
    probs: np.ndarray,
    parsed_answers: list[list[str | None]],
) -> np.ndarray:
    """Return [N, 4] answer-cluster posterior P(ans = a_c | x_i) = sum_{k: a_k=a_c} p_{i,k}."""
    n = probs.shape[0]
    cluster_post = np.zeros_like(probs)
    for i in range(n):
        ans_i = parsed_answers[i]
        p_i = probs[i]
        totals: dict[str, float] = {}
        for c_idx in range(4):
            a_str = ans_i[c_idx]
            if a_str is not None:
                totals[a_str] = totals.get(a_str, 0.0) + float(p_i[c_idx])
        for c_idx in range(4):
            a_str = ans_i[c_idx]
            if a_str is not None:
                cluster_post[i, c_idx] = min(1.0, totals[a_str])
            else:
                cluster_post[i, c_idx] = float(p_i[c_idx])
    return cluster_post


def select_with_cluster_aggregation(
    logits: np.ndarray,
    valid_mask: np.ndarray,
    parsed_answers: list[list[str | None]],
    cluster_alpha: float = 0.55,
    temperature: float = 0.85,
) -> tuple[list[int], np.ndarray]:
    """Select 1 candidate index in 0..3 per question using confidence-weighted clustering."""
    n = logits.shape[0]
    masked_logits = logits.copy()
    for i in range(n):
        if np.any(valid_mask[i]):
            masked_logits[i, ~valid_mask[i]] = -1e9

    probs = _softmax_4(masked_logits / max(temperature, 1e-3))
    choices: list[int] = []
    final_scores = np.zeros((n, 4), dtype=np.float32)

    for i in range(n):
        ans_i = parsed_answers[i]
        p_i = probs[i]
        cluster_prob: dict[str, float] = {}
        for c_idx in range(4):
            a_str = ans_i[c_idx]
            if a_str is not None:
                cluster_prob[a_str] = cluster_prob.get(a_str, 0.0) + float(p_i[c_idx])

        for c_idx in range(4):
            if valid_mask[i, c_idx] and ans_i[c_idx] is not None:
                c_p = max(cluster_prob[ans_i[c_idx]], 1e-12)
                ind_p = max(float(p_i[c_idx]), 1e-12)
                final_scores[i, c_idx] = (
                    (1.0 - cluster_alpha) * math.log(ind_p)
                    + cluster_alpha * math.log(c_p)
                )
            else:
                final_scores[i, c_idx] = -1e9 + float(logits[i, c_idx])

        choices.append(int(np.argmax(final_scores[i])))

    cal_probs = _softmax_4(masked_logits)
    cluster_posteriors = compute_answer_cluster_posteriors(cal_probs, parsed_answers)
    return choices, cluster_posteriors


def _heuristic_fallback_logits(
    tab_feats: np.ndarray,
    valid_mask: np.ndarray,
    se_cl: np.ndarray | None = None,
    cf_cl: np.ndarray | None = None,
) -> np.ndarray:
    """Symbolic + embedding consensus prior (when checkpoint is omitted)."""
    logits = (
        3.2 * tab_feats[:, :, 9]     # vote_frac (self-consistency)
        + 0.8 * tab_feats[:, :, 11]  # is_unique_plurality
        + 0.5 * tab_feats[:, :, 16]  # vote_margin over runner-up answer
        + 1.8 * tab_feats[:, :, 1]   # is_int (100% of GSM8K gold answers are integers)
        + 1.2 * tab_feats[:, :, 2]   # is_pos (99.94% of GSM8K gold answers are positive)
        - 0.7 * tab_feats[:, :, 4]   # is_zero penalty
        - 0.6 * tab_feats[:, :, 5]   # literal echo of question number penalty
        - 0.5 * tab_feats[:, :, 7]   # scale deviation from question max
        - 0.6 * tab_feats[:, :, 15]  # outlier deviation from group median
        + 0.5 * tab_feats[:, :, 26]  # has_hash_marker
        + 0.4 * tab_feats[:, :, 28]  # clean_ending
        + 0.9 * tab_feats[:, :, 31]  # multi-term arithmetic accuracy
        - 1.1 * tab_feats[:, :, 32]  # explicit arithmetic error flag
        + 0.6 * tab_feats[:, :, 35]  # question number coverage
        + 0.5 * tab_feats[:, :, 37]  # final answer appears in reasoning body
        + 0.4 * tab_feats[:, :, 38]  # last calculation matches final answer
        + 0.5 * tab_feats[:, :, 40]  # 4-gram uniqueness (non-repetition)
        - 0.3 * tab_feats[:, :, 41]  # backtracking penalty
        + 0.7 * tab_feats[:, :, 44]  # cluster mean arithmetic accuracy
        + 0.4 * tab_feats[:, :, 45]  # cluster mean question coverage
        - 0.35 * tab_feats[:, :, 49] # relative distance from question mean embedding
        + 0.35 * tab_feats[:, :, 50] # cosine similarity to question mean embedding
        + 0.4 * tab_feats[:, :, 52]  # embedding similarity within same answer cluster
    )
    if se_cl is not None and cf_cl is not None:
        is_pos_int = (tab_feats[:, :, 1] > 0.5) & (tab_feats[:, :, 2] > 0.5)
        pi = np.where(is_pos_int, 0.0, -4.0).astype(np.float32)
        non31 = (np.abs(tab_feats[:, :, 17]) < 0.5).astype(np.float32)
        logits = logits + pi + 2.50 * cf_cl - (0.15 + 2.50 * non31) * se_cl
    logits = np.where(valid_mask, logits, -1e9)
    return logits


def predict_with_scores(
    bank: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    checkpoint: Path | str | None = None,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Return (selected_indices, answer_cluster_posteriors, candidate_logits)."""
    embeddings = arrays['embeddings']
    tab_feats, valid_mask, parsed_answers = extract_tabular_and_meta(bank, embeddings)
    se_cl, cf_cl, _ = compute_step_completion_signals(bank, tab_feats, parsed_answers)

    ckpt_path: Path | None = None
    if checkpoint is not None:
        p = Path(checkpoint)
        if p.is_dir():
            candidate_file = p / 'model_bundle.npz'
            if candidate_file.exists():
                ckpt_path = candidate_file
        elif p.exists():
            ckpt_path = p
        elif p.with_suffix('.npz').exists():
            ckpt_path = p.with_suffix('.npz')

    if ckpt_path is None:
        default_p = Path(__file__).resolve().parent / 'runs' / 'model_bundle.npz'
        if default_p.exists():
            ckpt_path = default_p

    if ckpt_path is None or not ckpt_path.exists():
        logits = _heuristic_fallback_logits(tab_feats, valid_mask, se_cl=se_cl, cf_cl=cf_cl)
        choices, probs = select_with_cluster_aggregation(
            logits, valid_mask, parsed_answers, cluster_alpha=0.55, temperature=0.85
        )
        return choices, probs, logits

    with np.load(ckpt_path, allow_pickle=False) as bundle:
        pca_diff_mean = bundle['pca_diff_mean']
        pca_diff_basis = bundle['pca_diff_basis']
        pca_raw_mean = bundle['pca_raw_mean']
        pca_raw_basis = bundle['pca_raw_basis']
        cluster_dim = int(bundle['cluster_dim'][0])

        feat_mean = bundle['feat_mean']
        feat_std = bundle['feat_std']

        w_linear = bundle['w_linear']          # [K_folds, D]
        b_linear = bundle['b_linear']          # [K_folds]
        linear_weight = float(bundle['linear_weight'][0])
        mlp_weight = float(bundle['mlp_weight'][0])
        heur_weight = float(bundle['heur_weight'][0])

        w1_ens = bundle['w1_ens']              # [K_folds, D, H]
        b1_ens = bundle['b1_ens']              # [K_folds, H]
        w_peer_ens = bundle['w_peer_ens']      # [K_folds, H, H]
        w2_ens = bundle['w2_ens']              # [K_folds, H, 1]
        b2_ens = bundle['b2_ens']              # [K_folds, 1]
        w_skip_ens = bundle['w_skip_ens']      # [K_folds, D, 1]

        cluster_alpha = float(bundle['cluster_alpha'][0])
        temperature = float(bundle['temperature'][0])

        has_v3_specs = 'w_sym_spec' in bundle.files and 'w_emb_spec' in bundle.files
        if has_v3_specs:
            w_sym_spec = bundle['w_sym_spec']
            b_sym_spec = bundle['b_sym_spec']
            w_emb_spec = bundle['w_emb_spec']
            b_emb_spec = bundle['b_emb_spec']
            sym_spec_weight = float(bundle['sym_spec_weight'][0])
            emb_spec_weight = float(bundle['emb_spec_weight'][0])
        else:
            sym_spec_weight = 0.0
            emb_spec_weight = 0.0

    x_raw = build_model_features(
        tab_feats,
        embeddings,
        parsed_answers,
        pca_diff_mean,
        pca_diff_basis,
        pca_raw_mean,
        pca_raw_basis,
        cluster_dim=cluster_dim,
    )
    x_norm = np.clip((x_raw - feat_mean) / np.maximum(feat_std, 1e-6), -8.0, 8.0)

    # 1. Joint Listwise Linear Ensemble Logits
    lin_logits = _safe_fold_linear(x_norm, w_linear, b_linear)

    # 2. Joint Listwise Group-Interaction Peer-MLP Ensemble Logits
    n_folds = w1_ens.shape[0]
    mlp_logits = np.zeros((len(bank), 4), dtype=np.float32)
    for k in range(n_folds):
        mlp_logits += forward_mlp_member(
            x_norm,
            w1_ens[k],
            b1_ens[k],
            w_peer_ens[k],
            w2_ens[k],
            b2_ens[k],
            w_skip_ens[k],
        ) / float(n_folds)

    # 3. Step-Completion Asymmetry & Positive-Integer Structural Invariant Prior Logits
    heur_logits = compute_step_structural_prior(tab_feats, se_cl, cf_cl, valid_mask)

    total_logits = (
        linear_weight * lin_logits
        + mlp_weight * mlp_logits
        + heur_weight * heur_logits
    )

    # 4. Late-Fused Pure 54D Symbolic Specialist & Pure 168D Embedding Specialist
    if has_v3_specs and (sym_spec_weight > 0.0 or emb_spec_weight > 0.0):
        idx_sym_only = list(range(NUM_TAB_FEATURES))
        idx_emb_only = IDX_EMB_STAT + list(range(NUM_TAB_FEATURES, x_norm.shape[-1]))
        sym_logits = _safe_fold_linear(x_norm[:, :, idx_sym_only], w_sym_spec, b_sym_spec)
        emb_logits = _safe_fold_linear(x_norm[:, :, idx_emb_only], w_emb_spec, b_emb_spec)
        total_logits = total_logits + sym_spec_weight * sym_logits + emb_spec_weight * emb_logits

    choices, cluster_posteriors = select_with_cluster_aggregation(
        total_logits,
        valid_mask,
        parsed_answers,
        cluster_alpha=cluster_alpha,
        temperature=temperature,
    )
    return choices, cluster_posteriors, total_logits


def predict(bank, arrays, checkpoint=None):
    """Official entrypoint called by `predict.py`."""
    choices, _, _ = predict_with_scores(bank, arrays, checkpoint)
    return [int(c) for c in choices]


'''
def predict(bank, arrays, checkpoint):
    """Return one Python integer in 0..3 per question, in bank order.

    bank: list of question dictionaries with four candidate texts each.
    arrays['embeddings']: float32 NumPy array of shape [N, 4, 1024].
    checkpoint: pathlib.Path or None, supplied through --checkpoint.
    """
    raise NotImplementedError('Implement predict() in submission.py or supply your own --method file.')
'''