import os
import torch
from collections import defaultdict
from torch.utils import data
import copy
import numpy as np
import pickle
from tqdm import tqdm
import random, math
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor
import librosa    

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, args, data,subjects_dict,data_type="train"):
        self.data = data
        self.len = len(self.data)
        self.subjects_dict = subjects_dict
        self.data_type = data_type
        self.one_hot_labels = np.eye(len(subjects_dict["train"]))
        self.args = args

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        file_name = self.data[index]["name"]
        audio = self.data[index]["audio"]
        vertice = self.data[index]["vertice"]
        template = self.data[index]["template"]
        
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
        
        return torch.FloatTensor(audio),torch.FloatTensor(vertice), torch.FloatTensor(template), torch.FloatTensor(one_hot), file_name

    def __len__(self):
        return self.len

def read_data(args):
    print("Loading data...")
    data = defaultdict(dict)
    train_data = []
    valid_data = []
    test_data = []

    audio_path = os.path.join("data", args.dataset, args.wav_path)
    vertices_path = os.path.join("data", args.dataset, args.vertices_path)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    template_file = os.path.join("data", args.dataset, args.template_file)
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin,encoding='latin1')
    
    for r, ds, fs in os.walk(audio_path):
        for f in tqdm(fs):
            if f.endswith("wav"):
                wav_path = os.path.join(r,f)
                speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
                input_values = np.squeeze(processor(speech_array,sampling_rate=16000).input_values)
                key = f.replace("wav", "npy")
                data[key]["audio"] = input_values
                
                if args.dataset in ["vocaset", "BIWI"]:
                    subject_id = "_".join(key.split("_")[:-1])
                    temp = templates[subject_id]
                else:
                    temp = templates['v_template']
                    
                data[key]["name"] = f
                data[key]["template"] = temp.reshape((-1)) 
                vertice_path = os.path.join(vertices_path, key)
                
                if not os.path.exists(vertice_path):
                    del data[key]
                    print("Vertices Data Not Found! ", vertice_path)
                else:
                    data[key]["vertice"] = np.load(vertice_path, allow_pickle=True)

    subjects_dict = {}
    # subjects_dict["train"] = [i for i in args.train_subjects.split(" ")]
    # subjects_dict["val"] = [i for i in args.val_subjects.split(" ")]
    # subjects_dict["test"] = [i for i in args.test_subjects.split(" ")]
    subjects_dict["train"] = [i for i in args.train_subjects]
    subjects_dict["val"] = [i for i in args.val_subjects]
    subjects_dict["test"] = [i for i in args.test_subjects]

    splits = {
        'vocaset':{'train':range(1,41),'val':range(21,41),'test':range(21,41)},
        'BIWI':{'train':range(1,33),'val':range(33,37),'test':range(37,41)},
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
    if args.batch_size == 1:
        train_data = Dataset(args, train_data,subjects_dict,"train")
        dataset["train"] = data.DataLoader(dataset=train_data, batch_size=1, shuffle=True)
        valid_data = Dataset(args, valid_data,subjects_dict,"val")
        dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=1, shuffle=False)
        test_data = Dataset(args, test_data,subjects_dict,"test")
        dataset["test"] = data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
        return dataset
    else:
        train_data = Dataset(args, train_data, subjects_dict, "train")
        dataset["train"] = data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        valid_data = Dataset(args, valid_data, subjects_dict, "val")
        dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        test_data = Dataset(args, test_data, subjects_dict, "test")
        dataset["test"] = data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        return dataset

def collate_fn(batch):
    audios, vertices, templates, one_hots, file_names = zip(*batch)
    min_audio_len = min(a.shape[0] for a in audios)
    min_vert_len  = min(v.shape[0] for v in vertices)
    audio_batch = torch.stack([a[:min_audio_len] for a in audios], dim=0)  # (B, T_aud)
    vert_batch  = torch.stack([v[:min_vert_len] for v in vertices], dim=0)  # (B, T_vert, 3*N)
    template_batch = torch.stack(templates, dim=0)
    one_hot_batch = torch.stack(one_hots, dim=0)
    return audio_batch, vert_batch, template_batch, one_hot_batch, file_names

if __name__ == "__main__":
    get_dataloaders()