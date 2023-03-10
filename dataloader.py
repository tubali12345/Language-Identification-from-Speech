import shutil
import wave
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from config import Config
from data.audio_augm import audio_augment
from data.audio_preprocess import load_audio


# add normalization
class ShortAudioDataSet(Dataset):
    def __init__(self, audio_path: str,
                 audio_format: str = 'wav',
                 sample_rate: int = Config.sample_rate,
                 sample_len: int = 8,
                 no_samples_per_class: int = 5.4 * 10 ** 5,
                 random_sample: bool = False,
                 with_augmentation: bool = False):
        super(ShortAudioDataSet, self).__init__()
        # self.audios = list(map(str, Path(audio_path).rglob(f'*.{audio_format}')))
        self.audio_path = audio_path
        self.audio_format = audio_format
        self.audios = self.make_data_paths(no_samples_per_class=no_samples_per_class)
        # self.audios = [path for path in tqdm(self.audios, desc='Preparing data') if
        #                len(load_audio(path)) >= sample_len * sample_rate]
        # self.print_no_samples_per_class()
        self.sample_rate = sample_rate
        self.sample_len = sample_len * sample_rate
        self.random_sample = random_sample
        self.with_augmentation = with_augmentation

    def __len__(self):
        return len(self.audios)

    def __getitem__(self, i):
        audio = audio_augment(load_audio(self.audios[i]), self.sample_rate) if self.with_augmentation \
            else load_audio(self.audios[i])
        audio_class = Config.class_dict[self.audios[i].split('/')[-3]]
        audio_len = len(audio)
        if audio_len >= self.sample_len:
            start = audio_len // np.random.uniform(audio_len / (audio_len - self.sample_len), audio_len / (
                    audio_len - self.sample_len) + 10) if self.random_sample else 0
            return torch.FloatTensor(audio[start:start + self.sample_len]), torch.tensor(audio_class)
        # return torch.FloatTensor(audio), torch.tensor(audio_class)

    def make_data_paths(self,
                        no_samples_per_class: int):
        audio_paths = []
        for directory in Path(self.audio_path).glob('*/'):
            for i, aud in tqdm(enumerate(directory.rglob(f'*.{self.audio_format}')),
                               desc=f'Collecting data for class {directory.name}'):
                if i < no_samples_per_class:
                    audio_paths.append(str(aud))
                else:
                    break
        return audio_paths

    def print_no_samples_per_class(self):
        print('Counting number of samples per language')
        lang_count = {lang: 0 for lang in Config.class_dict.keys()}
        for audio in self.audios:
            try:
                lang_count[audio.split('/')[-3]] += 1
            except:
                print(audio)
        print(lang_count)

    def split_audio(self, audio_path):
        audio = load_audio(audio_path)
        return [torch.FloatTensor(audio[i:i + self.sample_len]) for i in range(0, len(audio), self.sample_len)]

    def write_audio(self, audio: np.array, out_wav: str):
        wave_w = wave.open(out_wav, 'w')
        wave_w.setnchannels(1)
        wave_w.setsampwidth(2)
        wave_w.setframerate(self.sample_rate)
        wave_w.writeframesraw(((audio * (2 ** 15 - 1)).astype("<h")).tobytes())
        wave_w.close()

    def write_all_audio(self, audios: list):
        for i, audio in enumerate(audios):
            self.write_audio(audio, f'{str(audio)}_{i}')


class LongAudioDataSet(Dataset):
    def __init__(self, audio_path: str,
                 audio_format: str = 'wav',
                 sample_rate: int = 16000):
        super(LongAudioDataSet, self).__init__()
        self.audios = [path for path in list(map(str, Path(audio_path).rglob(f'*.{audio_format}')))]
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.audios)

    def __getitem__(self, i):
        audio = load_audio(self.audios[i])
        audio_class = Config.class_dict[self.audios[i].split('/')[-2]]
        return torch.FloatTensor(audio), torch.tensor(audio_class)


def collate_fn(batch):
    x, y = list(zip(*batch))
    return torch.nn.utils.rnn.pad_sequence(x, batch_first=True), torch.tensor(y)


def make_val_data(path: str, val_split: float = 0.15):
    p = Path(path)
    outdir = path.replace(p.name, p.name + '_val')
    for directory in p.glob('*/'):
        if directory.name != 'en':
            for sd in directory.glob('*/'):
                subdir = Path(f'{outdir}/{directory.name}/{sd.name}')
                subdir.mkdir(parents=True, exist_ok=True)
                for i, audio in tqdm(enumerate(sd.rglob('*.wav'))):
                    if np.random.uniform(0, 1) <= val_split:
                        shutil.move(str(audio), f'{subdir}/{audio.name}')
