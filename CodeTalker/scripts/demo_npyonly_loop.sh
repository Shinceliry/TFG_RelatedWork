#!/bin/bash

dataset="vocaset"
config_dir="config/${dataset}"
template_config="${config_dir}/tmp_demo.yaml"
output_config="${config_dir}/generated_demo.yaml"
wav_base_path="/mnt/mlnas/sakaya/Dataset/VOCASET/audio"
demo_output_base="demo/output/"
demo_npy_base="/mnt/mlnas/sakaya/Inference/CodeTalker/vocaset"

conditions=("FaceTalk_170728_03272_TA" "FaceTalk_170904_00128_TA" "FaceTalk_170725_00137_TA" "FaceTalk_170915_00223_TA" "FaceTalk_170811_03274_TA" "FaceTalk_170913_03279_TA" "FaceTalk_170904_03276_TA" "FaceTalk_170912_03278_TA")
subjects=("FaceTalk_170809_00138_TA" "FaceTalk_170731_00024_TA")

export PYTHONPATH=./

for condition in "${conditions[@]}"; do
    for subject in "${subjects[@]}"; do
        subject_wav_dir="${wav_base_path}/${subject}"
        
        if [ ! -d "${subject_wav_dir}" ]; then
            echo "Directory not found: ${subject_wav_dir}"
            continue
        fi

        wav_files=($(find "${subject_wav_dir}" -type f -name "*.wav"))

        if [ ${#wav_files[@]} -eq 0 ]; then
            echo "No WAV files found for subject: ${subject} in ${subject_wav_dir}"
            continue
        fi

        for wav_file in "${wav_files[@]}"; do
            cp "${template_config}" "${output_config}"
            sed -i "s|condition: .*|condition: ${condition}|" "${output_config}"
            sed -i "s|subject: .*|subject: ${subject}|" "${output_config}"
            sed -i "s|demo_wav_path: .*|demo_wav_path: ${wav_file}|" "${output_config}"
            sed -i "s|demo_output_path: .*|demo_output_path: ${demo_output_base}/${condition}_${subject}|" "${output_config}"
            sed -i "s|demo_npy_save_folder: .*|demo_npy_save_folder: ${demo_npy_base}/${condition}_${subject}|" "${output_config}"
            echo "Running demo for Condition: ${condition}, Subject: ${subject}, Wav: ${wav_file}"
            python main/demo_npyonly.py --config "${output_config}"
        done
    done
done