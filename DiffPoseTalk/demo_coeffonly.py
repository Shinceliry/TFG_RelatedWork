import os
import argparse
import numpy as np
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from demo import Demo

class TestDataset(Dataset):
    def __init__(self, gt_dir: Path, test_list: Path):
        self.gt_dir = gt_dir
        
        with open(test_list, "r") as f:
            all_candidates = [line.strip() for line in f if line.strip()]
        
        self.movie_list = []
        for name in all_candidates:
            if (gt_dir / name).is_dir():
                self.movie_list.append(name)
        
        if not self.movie_list:
            print("Error: Not Found Movies on test_list.txt ")
        
    def __len__(self):
        return len(self.movie_list)
    
    def __getitem__(self, idx):
        return self.movie_list[idx]

class InferenceRunner:
    def __init__(self, args):
        self.demo = Demo(args, load_flame=False, load_renderer=False)
        self.args = args

    def run_inference_and_save_batch(self, batch_videos):
        """
        batch_videos: list of strings (video/sequence names). Each one corresponds to
                      a subdirectory in gt_dir that has audio16000.flac and shape.npy
        """
        # 1. Read and pad audio
        audio_list = []
        shape_list = []
        max_length = 0

        for vid_name in batch_videos:
            audio_path = self.args.gt_dir / vid_name / "audio16000.flac"
            shape_path = self.args.gt_dir / vid_name / "shape.npy"

            # Load audio (1D array)
            wave, sr = torchaudio.load(str(audio_path))
            wave = wave.mean(dim=0)  # in case stereo, convert to mono
            if sr != 16000:
                raise ValueError(f"Expected sr=16000, got {sr}")

            audio_list.append(wave)
            if len(wave) > max_length:
                max_length = len(wave)

            # Load shape
            shape_data = np.load(str(shape_path))
            if shape_data.ndim == 2:
                shape_data = shape_data[0]
            shape_list.append(torch.from_numpy(shape_data).float())

        # 2. Pad all audio to same length
        batched_audio = []
        clip_lens = []
        for wave in audio_list:
            clip_lens.append(int(wave.shape[0] / 16000.0 * self.args.fps))
            pad_len = max_length - wave.shape[0]
            if pad_len > 0:
                wave_padded = torch.cat([wave, torch.zeros(pad_len, dtype=wave.dtype)])
            else:
                wave_padded = wave
            batched_audio.append(wave_padded.unsqueeze(0))

        batched_audio = torch.cat(batched_audio, dim=0)  # (N, L_max)

        # 3. Stack shape
        batched_shape = torch.stack(shape_list, dim=0)  # (N, shape_dim=100), e.g.

        # 4. Actually run batched inference
        coef_dict_batched = self.demo.batch_infer_coeffs(
            n_repetitions=self.args.num_takes, 
            audio=batched_audio,
            shape_coef=batched_shape,
            style_feat=None,
            cfg_mode=None,
            cfg_cond=None,
            cfg_scale=self.args.scale_audio,
            include_shape=True
        )

        # 5. Save each item
        for i, vid_name in enumerate(batch_videos):
            output_dir = self.args.output_dir / vid_name
            output_dir.mkdir(parents=True, exist_ok=True)

            # for demonstration we do num_takes
            for take_index in range(self.args.num_takes):
                take_dir = output_dir / f"TAKE{take_index + 1}"
                take_dir.mkdir(parents=True, exist_ok=True)
                exp_data  = coef_dict_batched["exp"]  [take_index, i, :clip_lens[i]]
                pose_data = coef_dict_batched["pose"] [take_index, i, :clip_lens[i]]
                shape_data= coef_dict_batched["shape"][take_index, i, :clip_lens[i]]

                frames_array = []
                for frame_idx in range(exp_data.shape[0]):
                    frames_array.append({
                        "shape_params": shape_data[frame_idx].cpu().numpy(),
                        "exp_params":   exp_data[frame_idx].cpu().numpy(),
                        "pose_params":  pose_data[frame_idx].cpu().numpy(),
                    })
                frames_array = np.array(frames_array, dtype=object)
                np.save(take_dir / "coeff.npy", frames_array)


    def run(self):
        """
        DataLoaderを使って test_list 内の動画ディレクトリ名をバッチで受け取り、
        run_inference_and_save_batch(...) を呼び出す
        """
        
        dataset = TestDataset(self.args.gt_dir, self.args.test_list)
        loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=16,
            drop_last=False
        )

        for batch_videos in tqdm(loader):
            self.run_inference_and_save_batch(batch_videos)


def main():
    parser = argparse.ArgumentParser(description="DiffPoseTalk 推論スクリプト - Batched Version (DataLoader)")
    parser.add_argument("--exp_name", type=str, default="head-SA-hubert-WM")
    parser.add_argument("--iter", type=int, default=110000)
    parser.add_argument("--coef_stats", type=str, default="datasets/stats_train.npz")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num_takes", type=int, default=3)
    parser.add_argument("--output_dir", type=Path, default="pred")
    parser.add_argument("--scale_audio", type=float, default=1.15)
    parser.add_argument("--gt_dir", type=Path, required=True)
    parser.add_argument("--test_list", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=24, help="Number of items to run in a single forward pass")
    parser.add_argument('--mode', type=str)
    parser.add_argument('--black_bg', action='store_true')
    parser.add_argument('--no_context_audio_feat', action='store_true')
    parser.add_argument('--dynamic_threshold_ratio', '-dtr', type=float, default=0.)
    parser.add_argument('--dynamic_threshold_min', '-dtmin', type=float, default=1.)
    parser.add_argument('--dynamic_threshold_max', '-dtmax', type=float, default=4.)
    parser.add_argument('--save_coef', action='store_true')
    parser.add_argument('--no_head', action='store_true')
    parser.add_argument("--fps", type=int, default=30, help="Movie FPS")

    args = parser.parse_args()
    runner = InferenceRunner(args)
    runner.run()


if __name__ == "__main__":
    main()