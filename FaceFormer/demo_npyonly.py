import numpy as np
import scipy.io.wavfile as wav
import librosa
import os, sys, shutil, argparse, copy, pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from faceformer import Faceformer
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor

@torch.no_grad()
def test_model(args):
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    # Build model
    model = Faceformer(args)
    model.load_state_dict(torch.load(os.path.join(args.faceformer_dir_path, args.dataset, '{}.pth'.format(args.model_name))))
    model = model.to(torch.device(args.device))
    model.eval()

    template_file = os.path.join(args.faceformer_dir_path, "data", args.dataset, args.template_path)
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin, encoding='latin1')

    train_subjects_list = [i for i in args.train_subjects.split(" ")]

    one_hot_labels = np.eye(len(train_subjects_list))
    iter = train_subjects_list.index(args.condition)
    one_hot = one_hot_labels[iter]
    one_hot = np.reshape(one_hot, (-1, one_hot.shape[0]))
    one_hot = torch.FloatTensor(one_hot).to(device=args.device)

    temp = templates[args.subject]

    template = temp.reshape((-1))
    template = np.reshape(template, (-1, template.shape[0]))
    template = torch.FloatTensor(template).to(device=args.device)

    wav_path = args.wav_path
    test_name = os.path.basename(wav_path).split(".")[0]
    speech_array, sampling_rate = librosa.load(os.path.join(wav_path), sr=16000)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    audio_feature = np.squeeze(processor(speech_array, sampling_rate=16000).input_values)
    audio_feature = np.reshape(audio_feature, (-1, audio_feature.shape[0]))
    audio_feature = torch.FloatTensor(audio_feature).to(device=args.device)

    prediction = model.predict(audio_feature, template, one_hot)
    prediction = prediction.squeeze()  # (seq_len, V*3)
    np.save(os.path.join(args.result_path, test_name), prediction.detach().cpu().numpy())

def main():
    parser = argparse.ArgumentParser(description='FaceFormer: Speech-Driven 3D Facial Animation with Transformers')
    parser.add_argument("--model_name", type=str, default="biwi")
    parser.add_argument("--dataset", type=str, default="BIWI", help='vocaset or BIWI')
    parser.add_argument("--fps", type=float, default=25, help='frame rate - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--feature_dim", type=int, default=128, help='64 for vocaset; 128 for BIWI')
    parser.add_argument("--period", type=int, default=25, help='period in PPE - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--vertice_dim", type=int, default=23370*3, help='number of vertices - 5023*3 for vocaset; 23370*3 for BIWI')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train_subjects", type=str, nargs="+", default="F2 F3 F4 M3 M4 M5")
    parser.add_argument("--test_subjects", type=str, nargs="+", default="F1 F5 F6 F7 F8 M1 M2 M6")
    parser.add_argument("--output_path", type=str, default="demo/output", help='path of the rendered video sequence')
    parser.add_argument("--wav_path", type=str, default="demo/wav/test.wav", help='path of the input audio signal')
    parser.add_argument("--result_path", type=str, default="demo/result", help='path of the predictions')
    parser.add_argument("--condition", type=str, default="M3", help='select a conditioning subject from train_subjects')
    parser.add_argument("--subject", type=str, default="M1", help='select a subject from test_subjects or train_subjects')
    parser.add_argument("--background_black", type=bool, default=True, help='whether to use black background')
    parser.add_argument("--template_path", type=str, default="templates.pkl", help='path of the personalized templates')
    parser.add_argument("--faceformer_dir_path", type=str, default="/home/mic/Desktop/sakaya/3DTFG/FaceFormer")
    args = parser.parse_args()   

    test_model(args)

if __name__ == "__main__":
    main()