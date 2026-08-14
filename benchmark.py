#!/usr/bin/env python3
"""
Benchmark script for LightGBM on Credit Card Fraud Detection dataset.
Measures data loading time, training time, accuracy metrics, and inference latency/throughput.
Outputs results in terminal and saves to benchmark_result.json.
"""

import time
import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
import lightgbm as lgb

def main():
    print("=" * 60)
    print("🚀 BẮT ĐẦU BENCHMARK LIGHTGBM - CREDIT CARD FRAUD DETECTION")
    print("=" * 60)

    # 1. Tìm dataset creditcard.csv
    possible_paths = [
        "creditcard.csv",
        "~/ml-benchmark/creditcard.csv",
        os.path.expanduser("~/ml-benchmark/creditcard.csv"),
        "/home/ubuntu/ml-benchmark/creditcard.csv",
        "./ml-benchmark/creditcard.csv"
    ]
    
    data_path = None
    for p in possible_paths:
        expanded_p = os.path.expanduser(p)
        if os.path.exists(expanded_p):
            data_path = expanded_p
            break

    if not data_path:
        raise FileNotFoundError(
            "Không tìm thấy file 'creditcard.csv'. Hãy đảm bảo bạn đã tải dataset từ Kaggle về ~/ml-benchmark/ "
            "bằng lệnh: kaggle datasets download -d mlg-ulb/creditcardfraud --unzip -p ~/ml-benchmark/"
        )

    print(f"[*] Đang nạp dữ liệu từ: {data_path}")
    
    # 2. Đo thời gian Load Dataset
    start_load = time.perf_counter()
    df = pd.read_csv(data_path)
    load_time_sec = time.perf_counter() - start_load
    print(f"[*] Kích thước dữ liệu: {df.shape[0]:,} dòng, {df.shape[1]} cột")
    print(f"[✓] Thời gian load data: {load_time_sec:.4f} giây")

    # Tách Features & Target
    X = df.drop(columns=['Class'])
    y = df['Class']

    # Chia tập train/test (80/20) với Stratified để giữ tỷ lệ gian lận
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Huấn luyện LGBMClassifier & Đo thời gian Training
    print("\n[*] Đang huấn luyện LightGBM...")
    clf = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    start_train = time.perf_counter()
    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )
    training_time_sec = time.perf_counter() - start_train
    best_iteration = clf.best_iteration_ if hasattr(clf, 'best_iteration_') and clf.best_iteration_ else 200
    print(f"[✓] Thời gian training: {training_time_sec:.4f} giây")
    print(f"[✓] Best iteration: {best_iteration}")

    # 4. Đánh giá Model trên tập Test
    print("\n[*] Đang đánh giá model trên tập kiểm thử...")
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc_roc = float(roc_auc_score(y_test, y_pred_proba))
    accuracy = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred))
    recall = float(recall_score(y_test, y_pred))

    # 5. Đo Inference Latency (1 dòng) & Throughput (1000 dòng)
    print("\n[*] Đang đo tốc độ suy luận (Inference)...")
    sample_1_row = X_test.iloc[[0]]
    sample_1000_rows = X_test.iloc[:1000]

    # Warmup
    for _ in range(10):
        _ = clf.predict_proba(sample_1_row)

    # Đo latency 1 dòng (lấy trung bình qua 100 lần chạy)
    latency_trials = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = clf.predict_proba(sample_1_row)
        latency_trials.append(time.perf_counter() - t0)
    inference_latency_ms = float(np.mean(latency_trials) * 1000)

    # Đo throughput 1000 dòng (lấy trung bình qua 20 lần chạy)
    throughput_trials = []
    for _ in range(20):
        t0 = time.perf_counter()
        _ = clf.predict_proba(sample_1000_rows)
        duration = time.perf_counter() - t0
        throughput_trials.append(1000 / duration)  # rows/sec
    inference_throughput_fps = float(np.mean(throughput_trials))

    # 6. Tổng hợp kết quả
    results = {
        "dataset_name": "Credit Card Fraud Detection",
        "total_samples": int(df.shape[0]),
        "features_count": int(df.shape[1] - 1),
        "data_load_time_sec": round(load_time_sec, 4),
        "training_time_sec": round(training_time_sec, 4),
        "best_iteration": int(best_iteration),
        "metrics": {
            "auc_roc": round(auc_roc, 6),
            "accuracy": round(accuracy, 6),
            "f1_score": round(f1, 6),
            "precision": round(precision, 6),
            "recall": round(recall, 6)
        },
        "inference_benchmark": {
            "latency_1_row_ms": round(inference_latency_ms, 4),
            "throughput_1000_rows_per_sec": round(inference_throughput_fps, 2)
        }
    }

    # In bảng kết quả đẹp mắt
    print("\n" + "=" * 60)
    print("📊 KẾT QUẢ BENCHMARK LIGHTGBM")
    print("=" * 60)
    print(f"| {'Metric':<35} | {'Kết quả':<18} |")
    print(f"|{'-' * 37}|{'-' * 20}|")
    print(f"| {'Thời gian load data':<35} | {load_time_sec:.4f} s           |")
    print(f"| {'Thời gian training':<35} | {training_time_sec:.4f} s           |")
    print(f"| {'Best iteration':<35} | {best_iteration:<18} |")
    print(f"| {'AUC-ROC':<35} | {auc_roc:.6f}           |")
    print(f"| {'Accuracy':<35} | {accuracy:.6f}           |")
    print(f"| {'F1-Score':<35} | {f1:.6f}           |")
    print(f"| {'Precision':<35} | {precision:.6f}           |")
    print(f"| {'Recall':<35} | {recall:.6f}           |")
    print(f"| {'Inference latency (1 row)':<35} | {inference_latency_ms:.4f} ms         |")
    print(f"| {'Inference throughput (1000 rows)':<35} | {inference_throughput_fps:.2f} rows/s     |")
    print("=" * 60)

    # Ghi ra file JSON
    output_json = "benchmark_result.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"\n[✓] Đã lưu toàn bộ kết quả vào file: {output_json}")

if __name__ == "__main__":
    main()
