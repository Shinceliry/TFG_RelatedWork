#!/usr/bin/env python
import os
import torch
import numpy as np
import librosa
import pickle

from transformers import Wav2Vec2Processor
from base.utilities import get_parser
from models import get_model
from base.baseTrainer import load_state_dict

import datetime

cfg = get_parser()

def main():
    global cfg
    model = get_model(cfg)
    if torch.cuda.is_available():
        model = model.cuda()

    if os.path.isfile(cfg.model_path):
        print("=> loading checkpoint '{}'".format(cfg.model_path))
        checkpoint = torch.load(cfg.model_path, map_location=lambda storage, loc: storage.cpu())
        load_state_dict(model, checkpoint['state_dict'], strict=False)
        print("=> loaded checkpoint '{}'".format(cfg.model_path))
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(cfg.model_path))

    model.eval()
    save_folder = cfg.demo_npy_save_folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    condition = cfg.condition
    subject = cfg.subject
    test(model, cfg.demo_wav_path, save_folder, condition, subject)


def test(model, wav_file, save_folder, condition, subject):
    print('Generating facial animation for {}...'.format(wav_file))
    
    template_file = os.path.join(cfg.data_root, cfg.template_file)
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin, encoding='latin1')

    train_subjects_list = [i for i in cfg.train_subjects.split(" ")]
    one_hot_labels = np.eye(len(train_subjects_list))
    iter = train_subjects_list.index(condition)
    one_hot = one_hot_labels[iter]
    one_hot = np.reshape(one_hot, (-1, one_hot.shape[0]))
    one_hot = torch.FloatTensor(one_hot).to(device='cuda')

    temp = templates[subject]
    template = temp.reshape((-1))
    template = np.reshape(template, (-1, template.shape[0]))
    template = torch.FloatTensor(template).to(device='cuda')

    test_name = os.path.basename(wav_file).split(".")[0]
    if not os.path.exists(os.path.join(save_folder, test_name)):
        os.makedirs(os.path.join(save_folder, test_name))
    predicted_vertices_path = os.path.join(save_folder, test_name, 'condition_' + condition + '_subject_' + subject + '.npy')
    # predicted_vertices_path = os.path.join(save_folder, test_name, datetime.datetime.now().strftime("%Y%m%d%H%M%S"), '.npy')
    speech_array, _ = librosa.load(wav_file, sr=16000)
    processor = Wav2Vec2Processor.from_pretrained(cfg.wav2vec2model_path)
    audio_feature = np.squeeze(processor(speech_array, sampling_rate=16000).input_values)
    audio_feature = np.reshape(audio_feature, (-1, audio_feature.shape[0]))
    audio_feature = torch.FloatTensor(audio_feature).to(device='cuda')

    with torch.no_grad():
        prediction = model.predict(audio_feature, template, one_hot)
        prediction = prediction.squeeze()  # (seq_len, V*3)
        np.save(predicted_vertices_path, prediction.detach().cpu().numpy())
        print(f'Save facial animation in {predicted_vertices_path}')

if __name__ == '__main__':
    main()