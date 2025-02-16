import argparse
import math
import tempfile
import warnings
from pathlib import Path
import os

import cv2
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ここでは参照コードにあるDemo, utilsなどを直接同じファイル内に置いた体裁。
# 実際には from diff_talking_head import Demo, utils など適宜 import する必要があります。
from models import DiffTalkingHead
from utils import NullableArgs
from utils.media import combine_video_and_audio, convert_video, reencode_audio

warnings.filterwarnings('ignore', message='PySoundFile failed. Trying audioread instead.')


class Demo:
    def __init__(self, args, load_flame=True, load_renderer=True):
        # ここに示された __init__ の実装は参照コードと同様
        ...
    
    @torch.no_grad()
    def infer_from_file(self, audio_path, coef_path, out_path, style_path=None, tex_path=None, n_repetitions=1,
                        ignore_global_rot=False, cfg_mode=None, cfg_cond=None, cfg_scale=1.15):
        # 参照コードと同様
        ...
    
    @torch.no_grad()
    def infer_coeffs(self, audio, shape_coef, style_feat=None, n_repetitions=1,
                     cfg_mode=None, cfg_cond=None, cfg_scale=1.15, include_shape=False):
        # 参照コードと同様
        ...
    
    @torch.no_grad()
    def infer_vertices(self, audio_path, coef_path, style_path=None, n_repetitions=1, ignore_global_rot=False,
                       cfg_mode=None, cfg_cond=None, cfg_scale=1.15):
        # 参照コードと同様
        ...
    
    def save_coef_file(self, coef, out_path):
        # 参照コードと同様
        ...
    
    def render_to_video(self, verts_list, out_path, audio_path=None, texture=None):
        # 参照コードと同様
        ...
    
    @staticmethod
    def _pad_coef(coef, n_frames, elem_ndim=1):
        # 参照コードと同様
        ...
    
    @staticmethod
    def _get_model_path(exp_name, iteration):
        # 参照コードと同様
        ...


def main_inference(dataset_dir: Path, pred_dir: Path, demo_app: Demo, n_take: int = 10):
    """
    dataset_dir: 以下の構造を持つディレクトリ
        dataset_dir/
        └─ movieX/
            ├─ shape.npy
            ├─ audio16000.flac または audio16000.wav
            ├─ 001/
            │    ├─ exp.npy
            │    ├─ pose.npy
            │    └─ tex.npy
            └─ 002/
                 ├─ exp.npy
                 ├─ pose.npy
                 └─ tex.npy
            ... (以下フレーム番号ごと)
    
    pred_dir: 推論結果を保存するルートディレクトリ
        pred/
        └─ movieX/
            ├─ TAKE1/
            │    ├─ coeff.npy
            │    └─ vertices.npy
            ├─ TAKE2/
            │    ├─ coeff.npy
            │    └─ vertices.npy
            ...
    
    demo_app: 参照コードの Demo クラスのインスタンス (DiffTalkingHeadモデル)
    n_take: 同一動画・音声に対して繰り返し推論を行う回数
    """
    # dataset_dir 下のディレクトリ一覧を取得 (movie1, movie2, ... など)
    video_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]

    for video_dir in video_dirs:
        video_name = video_dir.name
        
        # shape.npy を読み込む
        shape_path = video_dir / "shape.npy"
        if not shape_path.exists():
            print(f"Skipping {video_name}: shape.npy not found.")
            continue
        shape_coef = np.load(shape_path)
        
        # 音声ファイル (audio16000.flac / audio16000.wav) を探す
        audio_path_flac = video_dir / "audio16000.flac"
        audio_path_wav = video_dir / "audio16000.wav"
        if audio_path_flac.exists():
            audio_path = audio_path_flac
        elif audio_path_wav.exists():
            audio_path = audio_path_wav
        else:
            print(f"Skipping {video_name}: audio file not found.")
            continue
        
        print(f"Processing video: {video_name}")
        
        # ---- 推論 ----
        # n_take回(例:10回) 推論を行う
        #   Demo.infer_coeffs で shape と音声から係数を生成
        #   → (キー 'shape','exp','pose','jaw','eye'等) x (n_take, フレーム数, 各次元)
        #   → それを頂点座標に変換 (頂点: [n_take, フレーム数, 5023, 3])
        n_repetitions = n_take  # 同じ音声で複数テイク生成するため
        
        # まず 3D係数を推論 (n_take 回分生成される)
        coef_dict = demo_app.infer_coeffs(
            audio=audio_path,
            shape_coef=shape_coef,
            style_feat=None,       # スタイルは使用しない例
            n_repetitions=n_repetitions,
            include_shape=True     # shape も含める
        )
        # 頂点を推論
        # (n_take, num_frame, 5023, 3)
        vertices_all = demo_app.model.flame(
            shape_params=torch.from_numpy(shape_coef).unsqueeze(0).float().to(demo_app.device)
        )  # もしFLAMEで生の頂点を取りたければこんな形、だが下記のようにcoef_dict_to_verticesが通常は便利
        vertices_all = demo_app.infer_vertices(
            audio_path=audio_path,
            coef_path=shape_coef,  # shapeだけ入れておけばOK (内部で同じ推論する形)
            style_path=None,
            n_repetitions=n_repetitions
        )
        
        # ただしすでに上で infer_coeffs しているので
        # utils.coef_dict_to_vertices(coef_dict, flame, ...) を直接呼ぶ方法でもOK
        # もし Demo 内部実装を使わずに自前で取りたい場合:
        #   from utils import coef_dict_to_vertices
        #   vertices_all = coef_dict_to_vertices(coef_dict, demo_app.flame, rot_repr=demo_app.rot_repr)
        # (戻り値 shape: [n_take, num_frame, 5023, 3])

        # ---- 保存 ----
        # coeff.npy は [num_frame] 次元の配列。各要素が {'shape': ..., 'exp': ..., 'pose': ...} という辞書
        # vertices.npy は [num_frame, 5023, 3] の浮動小数配列
        
        # coef_dict は {'shape': (n_take, L, 100), 'exp': (n_take, L, 50), 'pose': (n_take, L, 6), ...} のような構造
        # テイク別にまとめて保存する
        num_frame = coef_dict['exp'].shape[1]  # フレーム数 L (音声長による)
        
        for take_idx in range(n_take):
            take_name = f"TAKE{take_idx+1}"
            out_dir = pred_dir / video_name / take_name
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # coeff.npy を作る: shape (num_frame,) のオブジェクト配列
            coeff_array = np.empty((num_frame,), dtype=object)
            for f in range(num_frame):
                coeff_array[f] = {
                    'shape': coef_dict['shape'][take_idx, f].cpu().numpy(),
                    'exp':   coef_dict['exp'][take_idx, f].cpu().numpy(),
                    'pose':  coef_dict['pose'][take_idx, f].cpu().numpy()
                    # 必要に応じて jaw, eye などを追加
                }
            
            # vertices.npy: shape (num_frame, 5023, 3)
            verts_array = vertices_all[take_idx].cpu().numpy()  # [num_frame, 5023, 3]
            
            np.save(out_dir / "coeff.npy", coeff_array, allow_pickle=True)
            np.save(out_dir / "vertices.npy", verts_array)
            
        print(f" -> Saved {n_take} takes to {pred_dir / video_name}")


if __name__ == "__main__":
    """
    使い方の例:
      python this_script.py --dataset_dir /path/to/dataset_dir \
                            --pred_dir /path/to/pred \
                            --exp_name HDTF_TFHP \
                            --iter 1000000 \
                            --coef_stats path/to/stats.npz
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="dataset_dir を指定（動画フォルダが並ぶディレクトリ）")
    parser.add_argument("--pred_dir", type=str, default="pred",
                        help="推論結果を保存するディレクトリ")
    parser.add_argument("--exp_name", type=str, default="HDTF_TFHP")
    parser.add_argument("--iter", type=int, default=1000000)
    parser.add_argument("--coef_stats", type=str, default="datasets/HDTF_TFHP/lmdb/stats_train.npz")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_context_audio_feat", action="store_true",
                        help="音声特徴を過去と独立に扱うフラグ")
    parser.add_argument("--dynamic_threshold_ratio", type=float, default=0.0)
    parser.add_argument("--dynamic_threshold_min", type=float, default=1.0)
    parser.add_argument("--dynamic_threshold_max", type=float, default=4.0)
    parser.add_argument("--save_coef", action="store_true")
    parser.add_argument("--takes", type=int, default=10, help="同一音声を何回生成するか")
    
    args = parser.parse_args()
    
    # Demo インスタンス化 (モデルを読み込む)
    demo_args = argparse.Namespace(
        exp_name=args.exp_name,
        iter=args.iter,
        coef_stats=args.coef_stats,
        device=args.device,
        black_bg=False,
        no_context_audio_feat=args.no_context_audio_feat,
        dynamic_threshold_ratio=args.dynamic_threshold_ratio,
        dynamic_threshold_min=args.dynamic_threshold_min,
        dynamic_threshold_max=args.dynamic_threshold_max,
        save_coef=args.save_coef
    )
    
    demo_app = Demo(demo_args, load_flame=True, load_renderer=False)  # 頂点さえ取得できればレンダラは不要ならFalse
    
    dataset_dir = Path(args.dataset_dir)
    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    main_inference(dataset_dir, pred_dir, demo_app, n_take=args.takes)
    
    print("Done.")
