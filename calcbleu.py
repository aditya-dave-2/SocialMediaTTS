import pandas as pd
import pickle
from nltk.translate import bleu_score
from nltk.tokenize import word_tokenize

with open('descriptions.pkl','rb') as f:
    descriptions = pickle.load(f)

df = pd.read_csv("evaluation/eval_data_combined.csv")


sf = bleu_score.SmoothingFunction()
s1=[]
s2 = []
des = []
counter= 0
for index, row in df.iterrows():
    image_descriptions = descriptions[row['image'][:-4]]
    image_descriptions_tokenized = [word_tokenize(text) for text in image_descriptions]
    prediction_tokenized = word_tokenize(row['prediction'])
    score = bleu_score.sentence_bleu(image_descriptions_tokenized, prediction_tokenized,smoothing_function=sf.method4) 
    score2 = bleu_score.sentence_bleu(image_descriptions_tokenized, prediction_tokenized,smoothing_function=sf.method3) 
    s1.append(score)
    s2.append(score2)
    des.append(image_descriptions)
    counter+=1
    # if counter>10:
    #     break


df = df.assign(ground_truth=des,score_method4=s1, score_method5=s2)
df.to_csv(
    'final_eval_with_scores_bleu.csv'
)
