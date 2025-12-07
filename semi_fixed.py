import json
import logging
import warnings
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

# ----------------------------- Globals & IO -----------------------------
OUT_DIR = Path("out")
OUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
RANDOM_STATE = 42
DEFAULT_GLOBAL_THRESHOLD = 0.7
CALIBRATION_HOLDOUT = 0.10

SUB_TO_CAT_DEFAULT: Dict[str, str] = {
    "praise": "discussion",
    "solution approach": "functional",
    "logical": "functional",
    "support issues": "functional",
    "visual representation": "refactoring",
    "false positive": "false positive",
    "documentation": "documentation",
    "validation": "refactoring",
    "variable naming": "refactoring",
    "code organization": "refactoring",
    "design discussion": "discussion",
    "question": "discussion",
    "resource": "functional",
    "alternate output": "refactoring",
    "functional": "functional",
    "interface": "functional",
    "timing": "functional",
}

def _normalize_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().str.strip()

def _normalize_map(m: Dict[str, str]) -> Dict[str, str]:
    return {str(k).strip().lower(): str(v).strip().lower() for k, v in m.items()}

def _safe_min_class_count(y: np.ndarray) -> int:
    if y.size == 0:
        return 0
    binc = np.bincount(y)
    binc = binc[binc > 0]
    return int(binc.min()) if binc.size else 0

def _build_feature_pipeline() -> ColumnTransformer:
    return ColumnTransformer([
        ("comment_tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, dtype=np.float32), "comment"),
        ("code_tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, dtype=np.float32), "code"),
    ])

def _predict_proba_safe(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)
        except Exception:
            pass
    def _decision_function(m, X_):
        if hasattr(m, "decision_function"):
            return m.decision_function(X_)
        if hasattr(m, "estimator") and hasattr(m.estimator, "decision_function"):
            return m.estimator.decision_function(X_)
        if hasattr(m, "base_estimator") and hasattr(m.base_estimator, "decision_function"):
            return m.base_estimator.decision_function(X_)
        if hasattr(m, "named_steps") and "clf" in m.named_steps and hasattr(m.named_steps["clf"], "decision_function"):
            return m.named_steps["clf"].decision_function(X_)
        raise RuntimeError("Model provides neither predict_proba nor decision_function; cannot produce confidences.")
    scores = np.asarray(_decision_function(model, X))
    if scores.ndim == 1:
        z = np.clip(scores, -50, 50)
        p1 = 1.0 / (1.0 + np.exp(-z))
        p0 = 1.0 - p1
        return np.vstack([p0, p1]).T
    else:
        z = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(np.clip(z, -50, 50))
        return exp / exp.sum(axis=1, keepdims=True)

def _quick_svm_c_search(X_df: pd.DataFrame, y: np.ndarray, Cs=(0.25, 0.5, 1.0, 2.0), random_state: int = RANDOM_STATE) -> float:
    minc = _safe_min_class_count(y)
    if minc < 2:
        logging.info("[SVM] min_class_count < 2; skipping C search, using C=1.0")
        return 1.0
    n_splits = min(3, minc)
    fe = _build_feature_pipeline()
    best_c, best_f1 = 1.0, -1.0
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for C in Cs:
        base = Pipeline([("fe", fe), ("clf", LinearSVC(class_weight="balanced", C=C, random_state=random_state))])
        scores = []
        for tr, va in skf.split(X_df, y):
            base.fit(X_df.iloc[tr], y[tr])
            probs = _predict_proba_safe(base, X_df.iloc[va])
            pred = probs.argmax(axis=1)
            scores.append(f1_score(y[va], pred, average="macro"))
        f1m = float(np.mean(scores))
        if f1m > best_f1:
            best_f1, best_c = f1m, C
    logging.info(f"[SVM] Selected C={best_c} (cv macro-F1={best_f1:.4f})")
    return best_c

def _train_with_calibration(X_df: pd.DataFrame, y: np.ndarray, log_prefix: str = "[SUB] ", holdout_size: float = CALIBRATION_HOLDOUT, random_state: int = RANDOM_STATE):
    classes = np.unique(y)
    if len(classes) < 2:
        logging.warning(f"{log_prefix}Only one class present. Using DummyClassifier.")
        fe = _build_feature_pipeline()
        dummy = DummyClassifier(strategy="most_frequent")
        pipe = Pipeline([("fe", fe), ("clf", dummy)])
        pipe.fit(X_df, y)
        return pipe
    best_c = _quick_svm_c_search(X_df, y, Cs=(0.25, 0.5, 1.0, 2.0), random_state=random_state)
    base_svm = LinearSVC(class_weight="balanced", C=best_c, random_state=random_state)
    fe = _build_feature_pipeline()
    base_pipe = Pipeline([("fe", fe), ("clf", base_svm)])
    min_count = _safe_min_class_count(y)
    can_holdout = min_count >= 3
    if can_holdout:
        X_tr, X_cal, y_tr, y_cal = train_test_split(X_df, y, test_size=holdout_size, stratify=y, random_state=random_state)
        logging.info(f"{log_prefix}Training base SVM (hold-out calib, size={holdout_size}).")
        classes_tr, counts_tr = np.unique(y_tr, return_counts=True)
        freq = dict(zip(classes_tr, counts_tr))
        N = len(y_tr)
        K = len(classes_tr)
        alpha = 0.5
        w_tr = np.array([(N / (K * freq[yi])) ** alpha for yi in y_tr], dtype=np.float32)
        base_pipe.fit(X_tr, y_tr, clf__sample_weight=w_tr)
        method = "sigmoid" if min_count < 10 else "isotonic"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                calib = CalibratedClassifierCV(base_pipe, method=method, cv="prefit")
                calib.fit(X_cal, y_cal)
        except TypeError:
            cv_splits = min(3, min_count)
            calib = CalibratedClassifierCV(base_pipe, method=method, cv=cv_splits)
            calib.fit(X_df, y)
        return calib
    if min_count >= 2:
        cv_splits = min(3, min_count)
        logging.info(f"{log_prefix}Training SVM with adaptive CV calibration (cv={cv_splits}).")
        calib = CalibratedClassifierCV(base_pipe, method="sigmoid", cv=cv_splits)
        calib.fit(X_df, y)
        return calib
    logging.warning(f"{log_prefix}Too few samples per class for calibration; returning uncalibrated SVM.")
    base_pipe.fit(X_df, y)
    return base_pipe

def _compute_per_class_thresholds(model, X_cal_df: pd.DataFrame, y_cal: np.ndarray, class_names: List[str], default_thr: float = DEFAULT_GLOBAL_THRESHOLD) -> Dict[str, float]:
    thresholds = {c: default_thr for c in class_names}
    if X_cal_df is None or y_cal is None or len(y_cal) == 0:
        return thresholds
    probas = _predict_proba_safe(model, X_cal_df)
    y_true_idx = y_cal
    for cls_idx, cls_name in enumerate(class_names):
        is_cls = (y_true_idx == cls_idx).astype(int)
        score = probas[:, cls_idx]
        pos = is_cls.sum()
        neg = len(is_cls) - pos
        if pos == 0 or neg == 0:
            continue
        uniq = np.unique(score)
        cand = uniq[np.linspace(0, uniq.size - 1, min(400, uniq.size), dtype=int)]
        best_f1 = -1.0
        best_thr = thresholds[cls_name]
        for t in cand:
            pred = (score >= t).astype(int)
            f1 = f1_score(is_cls, pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(t)
        thresholds[cls_name] = best_thr
    return thresholds

def _save_metrics_report(model, X_eval_df: pd.DataFrame, y_eval: np.ndarray, sub_le: LabelEncoder, path: Path):
    try:
        y_proba = _predict_proba_safe(model, X_eval_df)
        y_pred = np.argmax(y_proba, axis=1)
    except Exception:
        y_pred = model.predict(X_eval_df)
    report = classification_report(y_eval, y_pred, target_names=[str(x) for x in sub_le.classes_], output_dict=True, zero_division=0)
    cm = confusion_matrix(y_eval, y_pred).tolist()
    payload = {"macro_f1": report.get("macro avg", {}).get("f1-score", None), "micro_f1": report.get("accuracy", None), "per_class": report, "confusion_matrix": cm}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def train_hierarchical(train_json: str, pseudo_json: Optional[str] = None, sub_to_cat_map: Optional[Dict[str, str]] = None):
    logging.info("Loading labeled data...")
    with open(train_json, "r", encoding="utf-8") as f:
        labeled = json.load(f)
    df = pd.DataFrame(labeled)
    df = df[df["comment"].notna() & df["code"].notna()].copy()
    if df.empty:
        raise ValueError("No valid samples after filtering non-null comment/code.")
    if "subcategory" not in df.columns:
        raise ValueError("Training data must include 'subcategory'.")
    df["subcategory"] = _normalize_series(df["subcategory"])
    if "category" in df.columns:
        df["category"] = _normalize_series(df["category"])
    if pseudo_json:
        logging.info("Loading pseudo-labeled data...")
        with open(pseudo_json, "r", encoding="utf-8") as f:
            pseudo = json.load(f)
        df_pseudo = pd.DataFrame(pseudo)
        if not df_pseudo.empty:
            for col in ("comment", "code"):
                if col not in df_pseudo.columns:
                    df_pseudo[col] = ""
            if "subcategory" in df_pseudo.columns:
                df_pseudo["subcategory"] = _normalize_series(df_pseudo["subcategory"])
            if "category" in df_pseudo.columns:
                df_pseudo["category"] = _normalize_series(df_pseudo["category"])
            df = pd.concat([df, df_pseudo], ignore_index=True)
    sub_le = LabelEncoder().fit(df["subcategory"])
    y_sub = sub_le.transform(df["subcategory"])
    sub_to_cat_map = _normalize_map(sub_to_cat_map or SUB_TO_CAT_DEFAULT)
    if "category" in df.columns and not df["category"].isna().all():
        g = df.groupby("subcategory")["category"].agg(lambda x: x.dropna().value_counts().idxmax() if len(x.dropna()) else None)
        for sub in sub_le.classes_:
            if sub not in sub_to_cat_map:
                majority = g.get(sub)
                if isinstance(majority, str):
                    sub_to_cat_map[sub] = majority.strip().lower()
    for sub in sub_le.classes_:
        sub_to_cat_map.setdefault(sub, "functional")
    X_df = df[["comment", "code"]].copy()
    min_count = _safe_min_class_count(y_sub)
    use_holdout = min_count >= 3
    X_train_df, X_cal_df, y_train, y_cal = X_df, None, y_sub, None
    if use_holdout:
        X_train_df, X_cal_df, y_train, y_cal = train_test_split(X_df, y_sub, test_size=CALIBRATION_HOLDOUT, stratify=y_sub, random_state=RANDOM_STATE)
    model = _train_with_calibration(X_train_df, y_train, log_prefix="[SUB] ", holdout_size=CALIBRATION_HOLDOUT, random_state=RANDOM_STATE)
    class_names = list(sub_le.classes_)
    if X_cal_df is not None and y_cal is not None:
        thresholds = _compute_per_class_thresholds(model, X_cal_df, y_cal, class_names, default_thr=DEFAULT_GLOBAL_THRESHOLD)
        _save_metrics_report(model, X_cal_df, y_cal, sub_le, OUT_DIR / "metrics.json")
    else:
        thresholds = {c: 0.55 for c in class_names}
    support_sr = pd.Series(y_train).value_counts()
    support_dict = support_sr.to_dict()
    MIN_THR, CAP_THR, MIN_SUPPORT = 0.60, 0.70, 10
    for idx, sub_name in enumerate(class_names):
        t = thresholds.get(sub_name, DEFAULT_GLOBAL_THRESHOLD)
        s = support_dict.get(idx, 0)
        if s < MIN_SUPPORT:
            t = max(MIN_THR, min(t, CAP_THR))
        thresholds[sub_name] = float(t)
    joblib.dump(model, OUT_DIR / "sub_model_pipeline.joblib")
    joblib.dump(sub_le, OUT_DIR / "sub_le_global.joblib")
    with open(OUT_DIR / "sub_to_cat_map.json", "w", encoding="utf-8") as f:
        json.dump(sub_to_cat_map, f, indent=2)
    with open(OUT_DIR / "sub_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)
    joblib.dump({}, OUT_DIR / "sub_heads.joblib")
    joblib.dump({}, OUT_DIR / "sub_lenc.joblib")
    with open(OUT_DIR / "cat_le.joblib.json", "w", encoding="utf-8") as f:
        json.dump({}, f)
    with open(OUT_DIR / "cat_model.joblib.json", "w", encoding="utf-8") as f:
        json.dump({}, f)
    logging.info("Training complete. Saved: sub_model_pipeline.joblib, sub_le_global.joblib, sub_to_cat_map.json, sub_thresholds.json, metrics.json")

def generate_pseudo_labels(unlabeled_json: str, output_json: str, threshold: Optional[float] = None):
    logging.info("Generating pseudo-labels with per-class thresholds...")
    model = joblib.load(OUT_DIR / "sub_model_pipeline.joblib")
    sub_le = joblib.load(OUT_DIR / "sub_le_global.joblib")
    with open(OUT_DIR / "sub_to_cat_map.json", "r", encoding="utf-8") as f:
        sub_to_cat = json.load(f)
    with open(OUT_DIR / "sub_thresholds.json", "r", encoding="utf-8") as f:
        learned_thr = json.load(f)
    with open(unlabeled_json, "r", encoding="utf-8") as f:
        raw = json.load(f)
    df = pd.DataFrame(raw)
    if "comment" not in df.columns:
        df["comment"] = ""
    if "code" not in df.columns:
        df["code"] = ""
    df = df[df["comment"].notna() & df["code"].notna()]
    if df.empty:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)
        logging.info(f"Saved 0 pseudo-labeled samples to {output_json} (no valid inputs).")
        return
    X = df[["comment", "code"]]
    probas = _predict_proba_safe(model, X)
    top_idx = np.argmax(probas, axis=1)
    top_conf = probas[np.arange(len(top_idx)), top_idx]
    subs = sub_le.inverse_transform(top_idx)
    pseudo = []
    kept, skipped = 0, 0
    for i, (sub, conf) in enumerate(zip(subs, top_conf)):
        thr = float(learned_thr.get(sub, threshold if threshold is not None else DEFAULT_GLOBAL_THRESHOLD))
        if conf < thr:
            skipped += 1
            continue
        sample = df.iloc[i].to_dict()
        sample["subcategory"] = sub
        sample["category"] = sub_to_cat.get(sub, "functional")
        sample["_confidence_sub"] = float(conf)
        pseudo.append(sample)
        kept += 1
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(pseudo, f, indent=2)
    logging.info(f"Saved {kept} pseudo-labeled samples to {output_json} (skipped {skipped}).")

def guess_category(comment: str, code: str = "") -> Tuple[str, str, float, float]:
    model = joblib.load(OUT_DIR / "sub_model_pipeline.joblib")
    sub_le = joblib.load(OUT_DIR / "sub_le_global.joblib")
    with open(OUT_DIR / "sub_to_cat_map.json", "r", encoding="utf-8") as f:
        sub_to_cat = json.load(f)
    X = pd.DataFrame([{"comment": comment, "code": code}])
    proba = _predict_proba_safe(model, X)[0]
    sub_idx = int(np.argmax(proba))
    sub_conf = float(proba[sub_idx])
    sub = sub_le.inverse_transform([sub_idx])[0]
    cat = sub_to_cat.get(sub, "functional")
    return cat, sub, sub_conf, sub_conf

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, help="Path to labeled training JSON", default="manual_labeled_data_clean_last.json")
    parser.add_argument("--pseudo", type=str, help="Path to pseudo-labeled JSON to merge during training")
    parser.add_argument("--unlabeled", type=str, help="Path to unlabeled JSON for pseudo-labeling")
    parser.add_argument("--generate", action="store_true", help="Generate pseudo-labels")
    parser.add_argument("--output", type=str, default="pseudo_labels.json", help="Output path for pseudo-labels")
    parser.add_argument("--threshold", type=float, default=None, help="Global threshold override for pseudo-labeling (otherwise use learned per-class)")
    args = parser.parse_args()
    if args.generate and args.unlabeled:
        generate_pseudo_labels(args.unlabeled, args.output, threshold=args.threshold)
    elif args.train:
        train_hierarchical(args.train, args.pseudo, sub_to_cat_map=SUB_TO_CAT_DEFAULT)
    else:
        print("Usage:")
        print(" --train <labeled.json> [--pseudo <pseudo.json>]")
        print(" --generate --unlabeled <unlabeled.json> --output <pseudo.json> [--threshold 0.75]")