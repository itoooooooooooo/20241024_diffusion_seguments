import os
os.environ["OMP_NUM_THREADS"] = "2"
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class AudioDataset(Dataset):
    def __init__(self, data_path, n_fft, hop_length, n_mels, power, segment_length=128, segment_shift=64):
        self.data_path = data_path
        self.files = [f for f in os.listdir(data_path) if f.endswith(".wav")]
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.power = power
        self.segment_length = segment_length
        self.segment_shift = segment_shift
        self.segment_info = []  # セグメントごとにファイルと開始インデックスを記録
        
        # 各ファイルのセグメント情報を計算して segment_info に保存
        for file_name in self.files:
            file_path = os.path.join(self.data_path, file_name)
            y, sr = librosa.load(file_path, sr=None)
            mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=self.n_fft, 
                                                             hop_length=self.hop_length, 
                                                             n_mels=self.n_mels, 
                                                             power=self.power)
            log_mel_spectrogram = 20.0 / self.power * np.log10(mel_spectrogram + np.finfo(float).eps)
            num_frames = log_mel_spectrogram.shape[1]
            # 各セグメントの開始インデックスを計算して保存
            for start_idx in range(0, num_frames - self.segment_length + 1, self.segment_shift):
                self.segment_info.append((file_name, start_idx))

    def __len__(self):
        # segment_info の長さが全セグメント数になる
        return len(self.segment_info)

    def __getitem__(self, idx):
        file_name, start_idx = self.segment_info[idx]
        file_path = os.path.join(self.data_path, file_name)
        y, sr = librosa.load(file_path, sr=None)
        
        # メルスペクトログラムを計算
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=self.n_fft, 
                                                         hop_length=self.hop_length, 
                                                         n_mels=self.n_mels, 
                                                         power=self.power)
        log_mel_spectrogram = 20.0 / self.power * np.log10(mel_spectrogram + np.finfo(float).eps)
        
        # [0, 1] への正規化
        log_mel_spectrogram_min = log_mel_spectrogram.min()
        log_mel_spectrogram_max = log_mel_spectrogram.max()
        log_mel_spectrogram = (log_mel_spectrogram - log_mel_spectrogram_min) / (log_mel_spectrogram_max - log_mel_spectrogram_min + 1e-8)

        # セグメントを取得
        segment = log_mel_spectrogram[:, start_idx:start_idx + self.segment_length]
        segment = torch.tensor(segment)

        segment = segment.clone().detach().unsqueeze(0)  # Conv2D用にチャンネル追加

        label = 0 if "normal" in file_name else 1
        return segment, label


def get_dataloader(data_path, batch_size, n_fft, hop_length, n_mels, power, segment_length=128, segment_shift=64):
    dataset = AudioDataset(data_path, n_fft, hop_length, n_mels, power, segment_length, segment_shift)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


class AudioDataset_test(Dataset):
    def __init__(self, data_path, n_fft, hop_length, n_mels, power, segment_length=128, segment_shift=64):
        self.data_path = data_path
        self.files = [f for f in os.listdir(data_path) if f.endswith(".wav")]
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.power = power
        self.segment_length = segment_length
        self.segment_shift = segment_shift

    def __len__(self):
        # ファイル数を返す
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        file_path = os.path.join(self.data_path, file_name)
        y, sr = librosa.load(file_path, sr=None)
        
        # メルスペクトログラムを計算
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=self.n_fft, 
                                                         hop_length=self.hop_length, 
                                                         n_mels=self.n_mels, 
                                                         power=self.power)
        log_mel_spectrogram = 20.0 / self.power * np.log10(mel_spectrogram + np.finfo(float).eps)
        
        # [0, 1] への正規化
        log_mel_spectrogram_min = log_mel_spectrogram.min()
        log_mel_spectrogram_max = log_mel_spectrogram.max()
        log_mel_spectrogram = (log_mel_spectrogram - log_mel_spectrogram_min) / (log_mel_spectrogram_max - log_mel_spectrogram_min + 1e-8)

        # スライディングウィンドウでセグメントを取得
        num_frames = log_mel_spectrogram.shape[1]
        segments = []
        for start_idx in range(0, num_frames - self.segment_length + 1, self.segment_shift):
            segment = log_mel_spectrogram[:, start_idx:start_idx + self.segment_length]
            segment = torch.tensor(segment).clone().detach().unsqueeze(0)  # チャンネル追加
            segments.append(segment)

        # すべてのセグメントをまとめて1つのテンソルにする
        segments = torch.stack(segments)  # (N, 1, 128, segment_length)
        
        label = 0 if "normal" in file_name else 1
        return segments, label


def get_test_dataloader(data_path, batch_size, n_fft, hop_length, n_mels, power, segment_length=128, segment_shift=64):
    dataset = AudioDataset_test(data_path, n_fft, hop_length, n_mels, power, segment_length, segment_shift)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    return dataloader