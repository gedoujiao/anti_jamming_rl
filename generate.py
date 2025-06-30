import numpy as np
import os
from pathlib import Path

# 参数设置
SAMPLES_PER_CLASS = 1000
IQ_LENGTH = 1024
SAVE_DIR = "iq_dataset"
os.makedirs(SAVE_DIR, exist_ok=True)

def generate_clean_signal(length=IQ_LENGTH, snr_db=20):
    t = np.arange(length)
    signal = np.exp(1j * 2 * np.pi * 0.01 * t)  # 基本载波
    noise = (np.random.randn(length) + 1j * np.random.randn(length)) / np.sqrt(2)
    noise *= 10 ** (-snr_db / 20)
    return signal + noise

def add_sine_interference(signal, freq=0.2):
    t = np.arange(len(signal))
    sine_jam = 0.5 * np.exp(1j * 2 * np.pi * freq * t)
    return signal + sine_jam

def add_burst_interference(signal, strength=5, width=30):
    burst = np.zeros_like(signal)
    start = np.random.randint(0, len(signal) - width)
    burst[start:start+width] = strength * (np.random.randn(width) + 1j * np.random.randn(width))
    return signal + burst

def add_freq_hopping_interference(signal, hops=5):
    length = len(signal)
    hop_len = length // hops
    interfered = signal.copy()
    for i in range(hops):
        t = np.arange(hop_len)
        freq = np.random.uniform(0.05, 0.3)
        start = i * hop_len
        interfered[start:start+hop_len] += 0.4 * np.exp(1j * 2 * np.pi * freq * t)
    return interfered

def generate_sample(label):
    base = generate_clean_signal()
    if label == 0:
        return base
    elif label == 1:
        return add_sine_interference(base)
    elif label == 2:
        return add_burst_interference(base)
    elif label == 3:
        return add_freq_hopping_interference(base)

# 生成数据
X, y = [], []
for label in range(4):
    for _ in range(SAMPLES_PER_CLASS):
        sig = generate_sample(label)
        iq = np.stack([sig.real, sig.imag], axis=1)
        X.append(iq)
        y.append(label)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

# 保存数据
np.save(os.path.join(SAVE_DIR, "X.npy"), X)
np.save(os.path.join(SAVE_DIR, "y.npy"), y)

print(f"数据生成完成：{X.shape[0]} 个样本，输入维度 {X.shape[1:]}")
