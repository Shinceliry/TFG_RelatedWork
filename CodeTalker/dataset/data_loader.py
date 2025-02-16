import os
import torch
import numpy as np
import random
import librosa
from tqdm import tqdm
from psbody.mesh import Mesh
from transformers import Wav2Vec2Processor
from collections import defaultdict
from torch.utils import data

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data_list, data_type="train", read_audio=False):
        self.data_list = data_list
        self.len = len(self.data_list)
        self.read_audio = read_audio
        self.data_type = data_type

        # 単一の "S0" だけを想定するためのダミー
        self.one_hot_labels = np.eye(1)

    def __getitem__(self, index):
        item = self.data_list[index]
        # 頂点
        vertice = item["vertice"]   # shape: [frame, vertex_dim]
        template = item["template"] # shape: [vertex_dim]
        file_name = item["name"]    # 例: "sample1.wav"

        # オーディオが必要なら
        if self.read_audio:
            audio = item["audio"]  # shape: [サンプル数] 
            # one_hot はダミー1件
            one_hot = self.one_hot_labels[0]  # shape: (1,)
            return torch.FloatTensor(audio), \
                   torch.FloatTensor(vertice), \
                   torch.FloatTensor(template), \
                   torch.FloatTensor(one_hot), \
                   file_name
        else:
            # VQ-VAE学習(stage1)ではオーディオ不要
            one_hot = self.one_hot_labels[0]
            return torch.FloatTensor(vertice), \
                   torch.FloatTensor(template), \
                   torch.FloatTensor(one_hot), \
                   file_name

    def __len__(self):
        return self.len


def read_data(args):
    """ 指定フォルダ内の wav と npy をペアで読み込み、train/val/test にランダム分割する """
    print("Loading data from:", args.data_root)
    audio_path = os.path.join(args.data_root, args.wav_path)
    vertices_path = os.path.join(args.data_root, args.vertices_path)

    # 単一テンプレートとして FLAME_sample.ply を読む
    template_file = os.path.join(args.data_root, args.template_file)
    mesh = Mesh(filename=template_file)
    template_vertices = mesh.v.ravel()  # shape: [5023*3=15069]など

    file_pairs = []
    processor = None
    if args.read_audio:
        processor = Wav2Vec2Processor.from_pretrained(args.wav2vec2model_path)

    # ディレクトリを走査して .wav に対応する .npy を探す
    for f in sorted(os.listdir(audio_path)):
        if f.endswith(".wav"):
            wav_fn = os.path.join(audio_path, f)
            npy_fn = os.path.join(vertices_path, f.replace(".wav", ".npy"))
            if not os.path.exists(npy_fn):
                continue

            # 音声読み込み
            audio_data = None
            if args.read_audio and processor is not None:
                speech_array, _ = librosa.load(wav_fn, sr=16000)
                input_values = processor(speech_array, sampling_rate=16000).input_values
                audio_data = np.squeeze(input_values)

            # 頂点読み込み [フレーム数, 頂点数*3]
            vert_data = np.load(npy_fn)  # shape: [frame, 15069]など

            item = {
                "audio": audio_data,
                "vertice": vert_data,
                "template": template_vertices,
                "name": f
            }
            file_pairs.append(item)

    # ランダム分割 (8:1:1) ※用途に合わせて自由に変更
    random.shuffle(file_pairs)
    n = len(file_pairs)
    n_train = int(0.8 * n)
    n_val   = int(0.9 * n)
    train_data = file_pairs[:n_train]
    val_data   = file_pairs[n_train:n_val]
    test_data  = file_pairs[n_val:]

    print('Loaded data: Train-{}, Val-{}, Test-{}'.format(len(train_data), len(val_data), len(test_data)))
    return train_data, val_data, test_data


def get_dataloaders(args):
    train_data, valid_data, test_data = read_data(args)

    ds_train = Dataset(train_data, data_type="train", read_audio=args.read_audio)
    ds_valid = Dataset(valid_data, data_type="val",   read_audio=args.read_audio)
    ds_test  = Dataset(test_data,  data_type="test",  read_audio=args.read_audio)

    dataloader = {}
    dataloader["train"] = data.DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    dataloader["valid"] = data.DataLoader(ds_valid, batch_size=1, shuffle=False, num_workers=args.workers)
    dataloader["test"]  = data.DataLoader(ds_test,  batch_size=1, shuffle=False, num_workers=args.workers)
    return dataloader