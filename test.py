import os
os.environ["OMP_NUM_THREADS"] = "2"
import torch
import torch.nn as nn
from model import UNet, Diffuser
from data_loader import get_dataloader, get_test_dataloader
from sklearn.metrics import roc_auc_score, roc_curve, auc
import yaml
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

def calculate_anomaly_score(original, reconstructed, k_percent=0.3):
    # ピクセルごとの絶対誤差を計算
    pixel_errors = torch.abs(original - reconstructed)
    
    # 各サンプルのピクセル誤差を1次元にフラット化
    flat_errors = pixel_errors.view(pixel_errors.size(0), -1)
    
    # トップkのピクセル誤差の合計を取得
    k = int(flat_errors.size(1) * k_percent)  # トップk％のピクセル数を計算
    topk_errors, _ = torch.topk(flat_errors, k, dim=1, largest=True)
    
    # 異常スコアを計算（スコア = 1/(F*T) * トップkの誤差合計）
    anomaly_scores = topk_errors.sum(dim=1) / (128 * 128)
    
    return anomaly_scores

# YAMLの読み込み
with open("ae.yaml", "r") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_loader = get_test_dataloader(config['test_data_path'], config['batch_size'], config['n_fft'], config['hop_length'], config['n_mels'], config['power'])
model = UNet(in_ch=1).to(device)
model.load_state_dict(torch.load(config['model_directory'] + "/autoencoder_with_diffusion.pth"))
model.eval()
diffuser = Diffuser(num_timesteps=1000, device=device)

criterion = nn.MSELoss(reduction='none')

timestep = 40

# 音データごとのスコアを保存するリスト
results = []

with torch.no_grad():
    for data, labels in test_loader:
        data = data.to(device)

        # print(data.shape)  # (1, N, 1, 128, 313) -> Nはセグメント数
        data = data.squeeze(0)  # (N, 1, 128, 313) に戻す
        
        batch_size = data.size(0)  # バッチ内の音データ数
        
        segment_scores = []  # 音データごとにセグメントのスコアを保存するリスト

        # サンプルごとの損失を計算
        t = torch.full((batch_size,), timestep, device=device, dtype=torch.long)
        x_t, noise = diffuser.add_noise(data, t)
        reconstructed = diffuser.denoise(model, x_t, t)
        #loss = criterion(data, reconstructed)
        scores = calculate_anomaly_score(data, reconstructed, k_percent=0.1)

        # 各セグメントの損失を保存
        for i in range(batch_size):
            segment_scores.append(scores[i].mean().item())

        # 各音データのセグメントスコアの平均を計算し、ラベルとともに結果を保存
        avg_score = np.mean(segment_scores)
        results.append([avg_score, labels[0].item()])  # 同じ音データ内は同じラベルであると仮定

# 結果を NumPy 配列に変換して保存
results = np.array(results)
np.savetxt(config['result_directory'] + "/results.csv", results, delimiter=",", header="score,label")


# AUC, pAUCの計算
y_true = results[:, 1]
y_scores = results[:, 0]

# AUCの計算
auc_value = roc_auc_score(y_true, y_scores)

# ROC曲線を計算
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# pAUCの計算 (0 <= FPR <= 0.1 の範囲でのAUC)
fpr_limit = 0.1  # pAUCを計算するFPRの範囲
fpr_pauc = fpr[fpr <= fpr_limit]  # FPRが0.1以下の範囲
tpr_pauc = tpr[:len(fpr_pauc)]    # 対応するTPR
pauc_value = auc(fpr_pauc, tpr_pauc) / fpr_limit  # 正規化してpAUCを0-1スケールに

# AUCとpAUCの出力
print(f"AUC: {auc_value}")
print(f"pAUC (FPR <= {fpr_limit}): {pauc_value}")


# #ここからのコードは生成されたサンプルの確認用コード

# 画像を保存するディレクトリを確認し、存在しない場合は作成
comparison_image_directory = os.path.join(config['result_directory'], "reconstruction_comparison")
os.makedirs(comparison_image_directory, exist_ok=True)

# 再構成結果の比較画像を保存する関数
def save_single_comparison_image(original_segments, reconstructed_segments, segment_scores, file_index, label, result_image_directory):
    num_segments = original_segments.size(0)
    fig, axes = plt.subplots(num_segments, 2, figsize=(10, 5 * num_segments))

    for i in range(num_segments):
        # 元のログメルスペクトログラムを表示
        axes[i, 0].imshow(original_segments[i][0].cpu().numpy(), aspect='auto', origin='lower')
        axes[i, 0].set_title(f"Segment {i + 1} - Original")

        # 再構成されたログメルスペクトログラムを表示
        axes[i, 1].imshow(reconstructed_segments[i][0].cpu().numpy(), aspect='auto', origin='lower')
        axes[i, 1].set_title(f"Segment {i + 1} - Reconstructed | Score: {segment_scores[i]:.4f}")

    label_name = "normal" if label == 0 else "anomaly"
    plt.suptitle(f"File {file_index + 1} ({label_name.capitalize()}) Reconstruction Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(result_image_directory, f"comparison_file_{file_index + 1}_{label_name}.png"))
    plt.close(fig)

# 正常データと異常データの再構成結果を保存
num_files_to_save = 10
saved_file_count = 0

with torch.no_grad():
    for file_index, (data, labels) in enumerate(test_loader):
        if saved_file_count >= num_files_to_save:
            break

        data = data.to(device).squeeze(0)  # (3, 1, 128, 128) に戻す
        batch_size = data.size(0)
        
        # サンプルごとの再構成とスコアの計算
        t = torch.full((batch_size,), timestep, device=device, dtype=torch.long)
        x_t, noise = diffuser.add_noise(data, t)
        reconstructed = diffuser.denoise(model, x_t, t)
        loss = criterion(data, reconstructed)

        # 各セグメントのスコアを計算
        segment_scores = [loss[i].mean().item() for i in range(batch_size)]

        # 比較画像を保存
        save_single_comparison_image(data, reconstructed, segment_scores, file_index, labels[0].item(), comparison_image_directory)
        saved_file_count += 1

