# !pip install TensorFlowTTS
# !pip install git+https://github.com/repodiac/german_transliterate.git#egg=german_transliterate
import numpy as np
import soundfile as sf
import yaml

import tensorflow as tf

from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.inference import AutoProcessor

# initialize fastspeech2 model.
fastspeech2 = TFAutoModel.from_pretrained("tensorspeech/tts-fastspeech2-ljspeech-en")


# initialize mb_melgan model
mb_melgan = TFAutoModel.from_pretrained("tensorspeech/tts-mb_melgan-ljspeech-en")


# inference
processor = AutoProcessor.from_pretrained("tensorspeech/tts-fastspeech2-ljspeech-en")

def get_audio_from_text(text,fname):
    input_ids = processor.text_to_sequence(text)
    # fastspeech inference

    _, mel_after, _, _, _ = fastspeech2.inference(
        input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
        speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
        speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
        f0_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32),
        energy_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32),
    )

    # melgan inference
    audio_after = mb_melgan.inference(mel_after)[0, :, 0]

    # save to file
    sf.write(f'./{fname}.wav', audio_after, 22050, "PCM_16")


# get_audio_from_text("This finally Works!!!","file")