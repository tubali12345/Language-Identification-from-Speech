import wave
from multiprocessing import Process
from pathlib import Path

import numpy as np
import torch
from librosa import filters
from tqdm import tqdm

from config import Config


def load_audio(wav_path_str, start_sample=0, num_samples=None) -> np.array:
    wav_r = wave.open(wav_path_str)
    nchannels, samplewidth, samplerate, nframes, comptype, compname = wav_r.getparams()

    # Read num_samples samples beginning from start_sample, convert to np.int16
    wav_r.setpos(start_sample)
    num_samples = num_samples if num_samples is not None else nframes
    audio_bytes = wav_r.readframes(num_samples)
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)

    audio_np = audio_np * (1 / 2 ** 15)

    return audio_np


def write_audio(audio, wav_path_str, sample_rate):
    wave_w = wave.open(wav_path_str, 'w')
    wave_w.setnchannels(1)
    wave_w.setsampwidth(2)
    wave_w.setframerate(sample_rate)
    wave_w.writeframesraw(((audio * (2 ** 15 - 1)).astype("<h")).tobytes())
    wave_w.close()


def split_audio(audio, chunk_length: int = 8):
    if len(audio) == Config.sample_rate * chunk_length:
        return [audio]
    return [audio[i:i + Config.sample_rate * chunk_length] for i in
            range(0, len(audio) - Config.sample_rate * chunk_length, Config.sample_rate * chunk_length)]


def write_all_audio(audios: list, dest_audio_path):
    for i, audio in enumerate(audios):
        write_audio(audio, f'{dest_audio_path.replace(".wav", f"_{i}.wav")}', Config.sample_rate)


def split_all_wav_from_path(path: str, out_path: str):
    for d in Path(path).glob('*/'):
        if d.name != 'archive.txt':
            out_dir = Path(f'{out_path}/{d.name}')
            out_dir.mkdir(parents=True, exist_ok=True)
            for i, wav in tqdm(enumerate(d.rglob('*.wav')), desc=path):
                # if i % 50 == 0:
                #     if len(list(Path(out_path).rglob("*.wav"))) > 10 ** 6:
                #         print(f'Enough samples in {path}')
                #         break
                try:
                    audio = load_audio(str(wav))
                    if len(audio) > Config.sample_rate * 9:
                        audios = split_audio(audio)
                        write_all_audio(audios, f'{out_dir}/{wav.name}')
                        Path(f'{out_dir}/archive.txt').open('a').write(f'{wav.name}\n')
                        wav.unlink()
                except Exception as e:
                    print(wav, e)


def remove_short(path: str):
    for audio in tqdm(Path(path).rglob('*.wav')):
        try:
            if len(load_audio(str(audio))) < 8 * Config.sample_rate:
                audio.unlink()
        except Exception as e:
            audio.unlink()
            print(e)


class Aud2Mel(torch.nn.Module):
    def __init__(self, n_mels, sample_rate, n_fft, win_len, hop_len):
        super().__init__()
        self.n_mels = n_mels
        self.sr = sample_rate
        self.n_fft = n_fft
        self.win_len = win_len
        self.hop_len = hop_len
        self.fmin, self.fmax = 0., 8000.

        window = torch.hann_window(self.win_len)
        mel_basis = torch.tensor(filters.mel(self.sr, self.n_fft, n_mels=self.n_mels,
                                             fmin=self.fmin, fmax=self.fmax)).unsqueeze(0)
        self.register_buffer("window", window)
        self.register_buffer("mel_basis", mel_basis)

    def forward(self, auds_tensor):
        bs = len(auds_tensor)
        pows_tensor = torch.stft(auds_tensor, n_fft=self.n_fft, win_length=self.win_len,
                                 hop_length=self.hop_len, window=self.window).pow(2.0).sum(-1)
        mels_tensor = self.mel_basis.expand(bs, -1, -1).bmm(pows_tensor).transpose(1, 2)
        return mels_tensor.clamp(min=1e-10).log10()


if __name__ == '__main__':
    path = "/home/turib/train_data"
    processes = []
    for directory in Path(path).glob("*/"):
        if directory.name in ['es', 'en', 'hu', 'tr', 'fr', 'de']:
            p = Process(target=split_all_wav_from_path,
                        args=(f"/home/turib/train_data/{directory.name}_long", f"/home/turib/train_data/{directory.name}"))
            p.start()
            processes.append(p)
    for p in processes:
        p.join()
    remove_short(path)
