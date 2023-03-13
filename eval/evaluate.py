from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from data.audio_preprocess import Aud2Mel, load_audio, split_audio
from data.dataloader import ShortAudioDataSet, collate_fn
from models.resnet import ResNet50LangDetection, ResNet101LangDetection


class EvaluateModel:
    def __init__(self,
                 model,
                 weights_path: str,
                 device: str):
        self.model = model
        self.model.to(device)
        self.model.load_state_dict(torch.load(weights_path))
        self.model.eval()
        self.aud_to_mel = Aud2Mel(Config.feature_dim, Config.sample_rate, 2048, 400, 160)
        self.device = device

    def evaluate_short(self, test_data_path: str) -> None:
        predicted = np.array([])
        labels = np.array([])
        pred_proba = np.array([])

        with torch.no_grad():
            for batch in tqdm(load_data(test_data_path), desc='Validating'):
                x, y = batch

                mel = self.aud_to_mel(x)
                out = self.model(mel.transpose(1, 2).to(self.device))
                _, batch_predicted = out.max(1)

                softmax = torch.nn.Softmax(dim=1)
                pred_proba = np.append(pred_proba, np.array(softmax(out).cpu()))

                predicted = np.append(predicted, np.array(batch_predicted.cpu()))
                labels = np.append(labels, np.array(y.cpu()))
            m = Metrics(predicted, labels, pred_proba.reshape((-1, 6)))
            print(m.acc_by_lang())
            print(m.accuracy())
            print(m.topk_accuracy([1, 2, 3, 5]))

    def evaluate_long(self,
                      test_data_path,
                      batch_audio_chunk: int = 64,
                      audio_format: str = 'wav') -> None:
        predicted = np.array([])
        labels = np.array([])

        with torch.no_grad():
            for aud_path in tqdm(list(map(str, Path(test_data_path).rglob(f'*.{audio_format}')))):
                audio_class = aud_path.split('/')[-2]
                predicted_class = self.predict_from_file(aud_path, batch_audio_chunk)
                predicted = np.append(predicted, Config.class_dict[predicted_class])
                labels = np.append(labels, Config.class_dict[audio_class])
            m = Metrics(predicted, labels)
            print(m.acc_by_lang())
            print(m.accuracy())

    def predict_from_file(self, audio_path: str, batch_audio_chunk: int) -> str:
        audio = load_audio(audio_path)
        audio_chunks = torch.cat([torch.unsqueeze(torch.FloatTensor(aud), 0) for aud in split_audio(audio)])

        predicted = np.array([])

        with torch.no_grad():
            for i in range(0, len(audio_chunks), batch_audio_chunk):
                mel = self.aud_to_mel(audio_chunks[i:i + batch_audio_chunk])
                out = self.model(mel.transpose(1, 2).to(self.device))
                _, batch_predicted = out.max(1)
                predicted = np.append(predicted, np.array(batch_predicted.cpu()))

        return list(Config.class_dict.keys())[
            list(Config.class_dict.values()).index(np.argmax(np.bincount(list(map(int, predicted)))))]


class Metrics:
    def __init__(self,
                 predicted: np.array,
                 labels: np.array,
                 pred_proba: np.array = None):
        assert predicted.shape == labels.shape, f'pred and labels tensor must have the same size, but got {predicted.shape} and {labels.shape}'
        self.predicted = predicted
        self.labels = labels
        self.pred_proba = pred_proba

    def accuracy(self) -> float:
        return np.equal(self.predicted, self.labels).sum() / self.labels.shape[0]

    def acc_by_lang(self) -> dict:
        corr_by_lang = {lang: 0 for lang in Config.class_dict.keys()}
        no_by_lang = {lang: 0 for lang in Config.class_dict.keys()}
        for i in range(len(self.predicted)):
            no_by_lang[list(Config.class_dict.keys())[list(Config.class_dict.values()).index(self.labels[i])]] += 1
            if self.predicted[i] == self.labels[i]:
                corr_by_lang[
                    list(Config.class_dict.keys())[list(Config.class_dict.values()).index(self.labels[i])]] += 1
        return {lang: corr / no_by_lang[lang] for lang, corr in corr_by_lang.items() if no_by_lang[lang] != 0}

    def topk_accuracy(self, list_k: list):
        return {k: sum(self.labels[i] in np.argpartition(self.pred_proba[i], -k)[-k:]
                       for i in range(self.pred_proba.shape[0])) / self.pred_proba.shape[0] for k in list_k}


def load_data(data_path: str,
              batch_size: int = 64,
              prefetch_factor: int = 4,
              num_workers: int = 1
              ) -> DataLoader:
    return DataLoader(ShortAudioDataSet(data_path, random_sample=False), batch_size=batch_size, shuffle=False,
                      collate_fn=collate_fn, pin_memory=False, num_workers=num_workers, prefetch_factor=prefetch_factor)


if __name__ == '__main__':
    eval_resnet50 = EvaluateModel(ResNet50LangDetection(num_classes=6),
                                  weights_path='/home/turib/lang_detection/weights/ResNet50/weights_03_06/model_2.pth',
                                  device='cuda:2')
    # eval_resnet50.evaluate_long(test_data_path='/home/turib/test_data_long')
    eval_resnet50.evaluate_short('/home/turib/val_data')
