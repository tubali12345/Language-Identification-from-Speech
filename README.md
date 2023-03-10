# Language identifier from audio


## Model Selection

5 Different architectures have been tested on the Common Voice dataset

Validation accuracy per model:

AttRNN: 46.17

Transofrmer: 51.17

CLSTM: 72.43

ResNet50: 89.25

Resnet101: 90.15

**Thus I moved forward with the ResNet architectures**

## Data used

Approximately 1000 hours of audio per language, 85%/15% train/validation split

**English:** youtube, MITocw, CommonVoice

**Spanish:** youtube, RTVE, CommonVoice

**Hungarian:** youtube, nava, CommonVoice

**French:** youtube, eslo, journal_en_francais_facile_podcast, CommonVoice

**Turkish:** youtube, CommonVoice

**German:** ARD, CommonVoice

## Detailed description

### Data preprocess:

The audio files have been converted to wav format (sample_rate=16000) and split into 8 second chunks without any filtering thus there can be chunks where there is no speech (intro song/music between speeches). Assuming that all data source has the same amount of 'noise' (where there is no speech) this does not effect the performance of the model.


Audio augmentation has been used so that the model becomes more robust to noise and data anomaly. (pitch shift, volume change, roll and telephone simulation)

### Training:

...

### TensorBoard:

Tracking the training metrics using TensorBoard

**How to use:**

1. Connect to the server using:

    **ssh -L 16006:127.0.0.1:6006 192.168.8.13**

2. Launch TensorBoard:

    **tensorboard --logdir=/home/turib/lang_detection/train/runs/ --port=6006**

3.  Open your browser and enter:

    **127.0.0.1:16006**
