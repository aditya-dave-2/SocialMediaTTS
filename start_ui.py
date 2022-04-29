import gradio as gr
import numpy as np
import os
# tensorflow version
import tensorflow

# keras version
import keras

from os import listdir
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

from keras import Input, layers
from keras import optimizers
from tensorflow.keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.preprocessing import image
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Embedding, Dense, Activation, Flatten, Reshape, Dropout
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.models import Model
from tensorflow.keras.utils import to_categorical
from keras.models import load_model
from PIL import Image
import pickle


def greedySearch(photo):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break

    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

def beam_search_predictions(image, beam_index = 3):
    model = load_model('saved_model/comp_model.hdf5') #Make this smarter this isdumb
    with open('wordtoix_dict.pkl','rb') as f:
        wordtoix = pickle.load(f)
    with open('ixtoword.pkl','rb') as f:
        ixtoword = pickle.load(f)
    max_length = 38
    start = [wordtoix["startseq"]]
    start_word = [[start, 0.0]]
    while len(start_word[0][0]) < max_length:
        temp = []
        for s in start_word:
            par_caps = sequence.pad_sequences([s[0]], maxlen=max_length, padding='post')
            preds = model.predict([image,par_caps], verbose=0)
            word_preds = np.argsort(preds[0])[-beam_index:]
            # Getting the top <beam_index>(n) predictions and creating a 
            # new list so as to put them via the model again
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])
                    
        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-beam_index:]
    
    start_word = start_word[-1][0]
    intermediate_caption = [ixtoword[i] for i in start_word]
    final_caption = []
    
    for i in intermediate_caption:
        if i != 'endseq':
            final_caption.append(i)
        else:
            break

    final_caption = ' '.join(final_caption[1:])
    return final_caption


def preprocess(image):
    img = Image.fromarray(image)
    img = img.resize(size=(299, 299))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def encode(image):
    model = InceptionV3(weights='imagenet')
    model_new = Model(model.input, model.layers[-2].output)
    image = preprocess(image) 
    fea_vec = model_new.predict(image) 
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
    return fea_vec


def image_prediction(image):
    image = encode(image).reshape((1,2048))

    # print("Greedy Search:",greedySearch(image))
    # print("Beam Search, K = 3:",beam_search_predictions(image, beam_index = 3))
    # print("Beam Search, K = 5:",beam_search_predictions(image, beam_index = 5))
    # print("Beam Search, K = 7:",beam_search_predictions(image, beam_index = 7))
    # print("Beam Search, K = 10:",beam_search_predictions(image, beam_index = 10))
    return beam_search_predictions(image, beam_index = 3)






def greet(image):

    t_op = image_prediction(image)
    get_audio_file(t_op,"file")
    return t_op,"a.wav"


demo = gr.Interface(
    fn=greet,
    inputs=["image"],
    outputs=["text", "audio"],
)
if __name__ == "__main__":
    demo.launch()