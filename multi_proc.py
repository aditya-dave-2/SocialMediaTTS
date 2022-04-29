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
import multiprocessing as mp
from start_ui import beam_search_predictions,greedySearch
import csv

def worker(working_queue, output_queue):
    f = open(f"eval_data/{os.getpid()}.csv",'w')
    csv_out=csv.writer(f)
    csv_out.writerow(['image','prediction','search_param'])
    while True:
        if working_queue.empty() is True:
            f.close()
            break #this is supposed to end the process.
        else:

            picked = working_queue.get()
            image = picked[1].reshape((1,2048))

            if(picked[2] == -1):
                result_pred = greedySearch(image)
            else:
                result_pred = beam_search_predictions(image, beam_index = picked[2])


            csv_out.writerow((picked[0],result_pred,picked[2]))
            output_q.put((picked[0],result_pred,picked[2]))
           
    return



if __name__ == '__main__':
    with open('encoding_test.pkl','rb') as f:
        encoding_test = pickle.load(f)

    working_q = mp.Queue()
    output_q = mp.Queue()
    counter = 0
    for key,value in encoding_test.items():
        working_q.put((key,value,-1))
        working_q.put((key,value,3))
        working_q.put((key,value,7))
        working_q.put((key,value,10))

        # if counter>10:
        #     break

        counter+=1
    CPU_COUNT = mp.cpu_count()-8
    # CPU_COUNT = 4
    print(f"CPU_COUNT: {CPU_COUNT}")
    processes = [mp.Process(target=worker,args=(working_q, output_q)) for i in range(CPU_COUNT)]
    # processes = [mp.Process(target=worker,args=(working_q, output_q)) for i in range(3)]
    for proc in processes:
        proc.start()
    for proc in processes:
        proc.join()
    results_bank = []
    while True:
        if output_q.empty() is True:
            break
        else:
            results_bank.append(output_q.get())
    with open('results.pkl','wb') as f:
        pickle.dump(results_bank, f)
    print(len(results_bank))