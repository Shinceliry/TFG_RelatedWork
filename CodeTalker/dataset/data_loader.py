import os
import torch
import numpy as np
import pickle
from tqdm import tqdm
from transformers import Wav2Vec2Processor
import librosa
from collections import defaultdict
from torch.utils import data 

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, args, data, subjects_dict, data_type="train", read_audio=False):
        self.data = data
        self.len = len(self.data)
        self.subjects_dict = subjects_dict
        self.data_type = data_type
        self.one_hot_labels = np.eye(len(subjects_dict["train"]))
        self.read_audio = read_audio
        self.args = args

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        file_name = self.data[index]["name"]
        vertice = self.data[index]["vertice"]
        template = self.data[index]["template"]
        
        if self.read_audio:
            audio = self.data[index]["audio"]
        
        if self.data_type == "train":
            if self.args.dataset == "MEAD":
                subject = file_name.split("_")[0]
            elif self.args.dataset == "RAVDESS":
                subject = file_name.split("-")[-1][:-4]
            else: # VOCASET, BIWI
                subject = "_".join(file_name.split("_")[:-1])
            one_hot = self.one_hot_labels[self.subjects_dict["train"].index(subject)]
        else:
            one_hot = self.one_hot_labels
        
        if self.read_audio:
            return torch.FloatTensor(audio), torch.FloatTensor(vertice), torch.FloatTensor(template), torch.FloatTensor(one_hot), file_name
        else:
            return torch.FloatTensor(vertice), torch.FloatTensor(template), torch.FloatTensor(one_hot), file_name

    def __len__(self):
        return self.len


def read_data(args):
    print("Loading data...")
    data = defaultdict(dict)
    train_data = []
    valid_data = []
    test_data = []

    audio_path = os.path.join(args.data_root, args.wav_path)
    vertices_path = os.path.join(args.data_root, args.vertices_path)
    
    if args.read_audio:
        processor = Wav2Vec2Processor.from_pretrained(args.wav2vec2model_path)

    template_file = os.path.join(args.data_root, args.template_file)
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin, encoding='latin1')
    
    for r, ds, fs in os.walk(audio_path):
        for f in tqdm(fs):
            if f.endswith("wav"):
                key = f.replace("wav", "npy")
                if args.read_audio:
                    wav_path = os.path.join(r, f)
                    speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
                    input_values = np.squeeze(processor(speech_array, sampling_rate=16000).input_values)
                    data[key]["audio"] = input_values
                else:
                    data[key]["audio"] = None
                
                if args.dataset in ["vocaset", "BIWI"]:
                    subject_id = "_".join(key.split("_")[:-1])
                    temp = templates[subject_id]
                else:
                    temp = templates['v_template']
                
                data[key]["name"] = f
                data[key]["template"] = temp.reshape((-1))
                vertice_path = os.path.join(vertices_path, key)
                
                if not os.path.exists(vertice_path):
                    print(f"{vertice_path} is not found.")
                    del data[key]
                else:
                    data[key]["vertice"] = np.load(vertice_path, allow_pickle=True)
                    
    subjects_dict = {
        "train": args.train_subjects.split(" "),
        "val": args.val_subjects.split(" "),
        "test": args.test_subjects.split(" ")
    }
    
    splits = {
        'vocaset': {'train': range(1, 41), 'val': range(21, 41), 'test': range(21, 41)},
        'BIWI': {'train': range(1, 33), 'val': range(33, 37), 'test': range(37, 41)},
        'MEAD': {'train': range(1, 61), 'val': range(1, 61), 'test': range(1, 61)},
        'RAVDESS': {'train': range(1, 3), 'val': range(1, 3), 'test': range(1, 3)}
    }
    
    for k, v in data.items():
        if args.dataset in ["vocaset", "BIWI"]:
            subject_id = "_".join(k.split("_")[:-1])
            sentence_id = int(k.split(".")[0][-2:])
        elif args.dataset == "MEAD":
            subject_id = k.split("_")[0] 
            sentence_id = int(k.split("_")[-1][:-4])
        elif args.dataset == "RAVDESS":
            subject_id = k.split("-")[-1][:-4]
            sentence_id = int(k.split("-")[-3]) 
        
        if subject_id in subjects_dict["train"] and sentence_id in splits[args.dataset]['train']:
            train_data.append(v)
        if subject_id in subjects_dict["val"] and sentence_id in splits[args.dataset]['val']:
            valid_data.append(v)
        if subject_id in subjects_dict["test"] and sentence_id in splits[args.dataset]['test']:
            test_data.append(v)


    print('Loaded data: Train-{}, Val-{}, Test-{}'.format(len(train_data), len(valid_data), len(test_data)))
    return train_data, valid_data, test_data, subjects_dict


def get_dataloaders(args):
    dataset = {}
    train_data, valid_data, test_data, subjects_dict = read_data(args)
    train_dataset = Dataset(args, train_data, subjects_dict, "train", args.read_audio)
    dataset["train"] = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    valid_dataset = Dataset(args, valid_data, subjects_dict, "val", args.read_audio)
    dataset["valid"] = data.DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_dataset = Dataset(args, test_data, subjects_dict, "test", args.read_audio)
    dataset["test"] = data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    
    return dataset


if __name__ == "__main__":
    get_dataloaders()