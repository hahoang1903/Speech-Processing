import os
import re

from pydub import AudioSegment
import shutil
from functools import reduce


def segment_sound(link_folder):
    """
    separate each word in a sentence
    :param link_folder: path of folder contains record and label of speech
    :return: segment sound word to folder data/segmented
    """
    os.makedirs('data/segmented', exist_ok=True)
    id_student = re.findall(r'[0-9]+', link_folder)[0]
    # Files name in folder of member us Audacity record
    file_names = set(file[:-4] for file in os.listdir(link_folder))
    # Sort name
    file_names = sorted(file_names)

    # Process time of sound and label of sound in file .txt
    for file in file_names:
        labels = []
        with open(f"{link_folder}/{file}.txt", "r") as f:
            for line in f:
                if line.strip() == "":
                    continue
                start, stop, label = line.split("\t")
                labels.append([start, stop, label.strip()])
        # export segment label
        for order, label in enumerate(labels):
            audio = AudioSegment.from_wav(f"{link_folder}/{file}.wav")
            audio = audio[int(float(label[0]) * 1000)                          :int(float(label[1]) * 1000)]
            os.makedirs(
                f"data/segmented/{id_student}/{label[2]}", exist_ok=True)
            audio.export(
                f"data/segmented/{id_student}/{label[2]}/{file}_{order+1}.wav", format="wav"
            )
    print(f'Segment {link_folder} done')


def generate_dtw_template(id_list):
    template_path = f'dtw/template_{reduce(lambda id1, id2: id1 + id2[-2:], sorted(id_list))}'
    if os.path.exists(template_path):
        return template_path

    os.makedirs(template_path, exist_ok=True)

    for label in ['A', 'B', 'len', 'xuong', 'phai', 'trai', 'nhay', 'ban', 'sil']:
        os.makedirs(f'{template_path}/{label}', exist_ok=True)
        for id in id_list:
            first_file = os.listdir(f'data/segmented/{id}/{label}')[0]
            shutil.copy(
                f'data/segmented/{id}/{label}/{first_file}', f'{template_path}/{label}'
            )

    return template_path
