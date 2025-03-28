import os
import torch
from collections import defaultdict
from torch.utils import data
import numpy as np
import pickle
import random
from tqdm import tqdm
from transformers import Wav2Vec2Processor
from sklearn.model_selection import train_test_split
import librosa
from torch.nn.utils.rnn import pad_sequence


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, args, data, subjects_dict, data_type="train"):
        self.data = data
        self.len = len(self.data)
        self.subjects_dict = subjects_dict
        self.data_type = data_type
        self.one_hot_labels = np.eye(len(subjects_dict["train"]))
        self.args = args

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
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

        return torch.FloatTensor(audio), vertice, torch.FloatTensor(template), torch.FloatTensor(one_hot), file_name

    def __len__(self):
        return self.len


def read_data(args):
    print("Loading data...")
    data = defaultdict(dict)
    train_data = []
    valid_data = []
    test_data = []

    audio_path = os.path.join(args.data_path, args.dataset, args.wav_path)
    vertices_path = os.path.join(args.data_path, args.dataset, args.vertices_path)

    processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-xlarge-ls960-ft")

    template_file = os.path.join(args.data_path, args.dataset, args.template_file)
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin, encoding='latin1')

    indices_to_split = []
    # all_subjects = args.train_subjects.split() + args.val_subjects.split() + args.test_subjects.split()
    all_subjects = args.train_subjects + args.val_subjects + args.test_subjects
    for r, ds, fs in os.walk(audio_path):
        for f in tqdm(fs):
            if f.endswith("wav"):
                wav_path = os.path.join(r, f)
                key = f.replace("wav", "npy")
                
                if args.dataset in ["vocaset", "BIWI"]:
                    subject_id = "_".join(key.split("_")[:-1])
                    temp = templates.get(subject_id, np.zeros(args.vertice_dim))
                else:
                    temp = templates['v_template']
                
                if args.dataset == 'beat':
                    emotion_id = int(key.split(".")[0].split("_")[-2])
                    indices_to_split.append([sentence_id, emotion_id, subject_id])

                speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
                input_values = np.squeeze(processor(speech_array, return_tensors="pt", padding="longest",
                                         sampling_rate=sampling_rate).input_values)

                data[key]["audio"] = input_values
                data[key]["name"] = f
                data[key]["template"] = temp.reshape((-1))
                vertice_path = os.path.join(vertices_path, key)
                
                if not os.path.exists(vertice_path):
                    del data[key]
                    print("Vertices Data Not Found! ", vertice_path)
                else:
                    data[key]["vertice"] = vertice_path

    train_split = defaultdict(list)
    val_split = defaultdict(list)
    test_split = defaultdict(list)

    # for beat do a stratified split
    # it ensures a balanced representation of emotions across the sets
    if args.dataset == 'beat':
        indices_to_split = np.array(indices_to_split)
        train_indices, test_indices = train_test_split(
            indices_to_split, test_size=0.1, stratify=indices_to_split[:, 1:3], random_state=42
        )
        train_indices, val_indices = train_test_split(
            train_indices, test_size=1 / 9, stratify=train_indices[:, 1:3], random_state=42
        )

        print(train_indices.shape, val_indices.shape, test_indices.shape)

        for idx in train_indices:
            train_split[idx[-1]].append(int(idx[0]))
        for idx in val_indices:
            val_split[idx[-1]].append(int(idx[0]))
        for idx in test_indices:
            test_split[idx[-1]].append(int(idx[0]))

    indices = list(range(1, 2538))
    random.Random(1).shuffle(indices)
    nr_samples = 100
    splits = {
        'BIWI': {
            'train': range(1, 33),
            'val': range(33, 37),
            'test': range(37, 41)
        },
        'multiface': {
            'train': list(range(1, 41)),
            'val': list(range(41, 46)),
            'test': list(range(46, 51))
        },
        'damm_rig_equal': {
            'train': indices[:int(0.8 * nr_samples)],
            'val': indices[int(0.8 * nr_samples):int(0.9 * nr_samples)],
            'test': indices[int(0.9 * nr_samples):nr_samples]
        },
        'beat': {
            'train': train_split,
            'val': val_split,
            'test': test_split
        },
        'vocaset': {
            'train': range(1, 41), 
            'val': range(21, 41), 
            'test': range(21, 41)
        },
        'MEAD': {
            'train': range(1, 61), 
            'val': range(1, 61), 
            'test': range(1, 61)
        },
        'RAVDESS': {
            'train': range(1, 3), 
            'val': range(1, 3), 
            'test': range(1, 3)
        }
    }


    subjects_dict = {}
    # subjects_dict["train"] = [i for i in args.train_subjects.split(" ")]
    # subjects_dict["val"] = [i for i in args.val_subjects.split(" ")]
    # subjects_dict["test"] = [i for i in args.test_subjects.split(" ")]
    subjects_dict["train"] = [i for i in args.train_subjects]
    subjects_dict["val"] = [i for i in args.val_subjects]
    subjects_dict["test"] = [i for i in args.test_subjects]


    print(subjects_dict)

    for k, v in data.items():
        if args.dataset == 'beat':
            subject_id = k.split("_")[0]
            sentence_id = int(k.split(".")[0].split("_")[-1])
            if subject_id in subjects_dict["train"] and sentence_id in splits[args.dataset]['train'][subject_id]:
                train_data.append(v)
            elif subject_id in subjects_dict["val"] and sentence_id in splits[args.dataset]['val'][subject_id]:
                valid_data.append(v)
            elif subject_id in subjects_dict["test"] and sentence_id in splits[args.dataset]['test'][subject_id]:
                test_data.append(v)
        else:
            if args.dataset == 'BIWI' or args.dataset == 'vocaset':
                subject_id = "_".join(k.split("_")[:-1])
                sentence_id = int(k.split(".")[0][-2:])
            elif args.dataset == 'MEAD':
                subject_id = k.split("_")[0] 
                sentence_id = int(k.split("_")[-1][:-4])
            elif args.dataset == "RAVDESS":
                subject_id = k.split("-")[-1][:-4]
                sentence_id = int(k.split("-")[-3]) 
            
            if subject_id in subjects_dict["train"] and sentence_id in splits[args.dataset]['train']:
                train_data.append(v)
            elif subject_id in subjects_dict["val"] and sentence_id in splits[args.dataset]['val']:
                valid_data.append(v)
            elif subject_id in subjects_dict["test"] and sentence_id in splits[args.dataset]['test']:
                test_data.append(v)

    print('Loaded data: Train-{}, Val-{}, Test-{}'.format(len(train_data), len(valid_data), len(test_data)))
    return train_data, valid_data, test_data, subjects_dict


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloaders(args):
    g = torch.Generator()
    g.manual_seed(0)
    dataset = {}
    train_data, valid_data, test_data, subjects_dict = read_data(args)
    
    if args.batch_size == 1:
        train_data = Dataset(args, train_data,subjects_dict,"train")
        dataset["train"] = data.DataLoader(dataset=train_data, batch_size=1, shuffle=True, worker_init_fn=seed_worker, generator=g)
        valid_data = Dataset(args, valid_data,subjects_dict,"val")
        dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=1, shuffle=False)
        test_data = Dataset(args, test_data,subjects_dict,"test")
        dataset["test"] = data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
        return dataset
    else:
        train_data = Dataset(args, train_data, subjects_dict, "train")
        dataset["train"] = data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g, collate_fn=collate_fn)
        valid_data = Dataset(args, valid_data, subjects_dict, "val")
        dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        test_data = Dataset(args, test_data, subjects_dict, "test")
        dataset["test"] = data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        return dataset

def collate_fn(batch):
    audio_list, vertice_list, template_list, one_hot_list, file_name_list = zip(*batch)
    if isinstance(audio_list[0], torch.Tensor):
        audio_list = pad_sequence(audio_list, batch_first=True, padding_value=0)
        
    if isinstance(vertice_list[0], torch.Tensor):
        vertice_list = pad_sequence(vertice_list, batch_first=True, padding_value=0)
    
    template_list = torch.stack(template_list)
    one_hot_list = torch.stack(one_hot_list)

    return audio_list, vertice_list, template_list, one_hot_list, file_name_list

if __name__ == "__main__":
    get_dataloaders()

