#!/usr/bin/env python3
import os
import shutil
import subprocess

print("Installing kagglehub...")
subprocess.run(["sudo", "pip3", "install", "kagglehub"], check=True)

import kagglehub

print("Downloading dataset 'mlg-ulb/creditcardfraud'...")
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
print(f"Downloaded to: {path}")

os.makedirs("/home/ubuntu/ml-benchmark", exist_ok=True)
for f in os.listdir(path):
    src = os.path.join(path, f)
    dst = os.path.join("/home/ubuntu/ml-benchmark", f)
    shutil.copy(src, dst)

print("Files in /home/ubuntu/ml-benchmark:", os.listdir("/home/ubuntu/ml-benchmark"))

print("\n--- RUNNING BENCHMARK ---")
import sys
sys.path.insert(0, "/home/ubuntu")
import benchmark
benchmark.main()
