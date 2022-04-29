import pandas as pd
import glob
import argparse

parser = argparse.ArgumentParser()

# parser.add_argument('--path', type=str, required=True)
# args = parser.parse_args()
# path = args.path
path='eval_data/'
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    # exit()
    df = pd.read_csv(filename,lineterminator='\n')
    li.append(df)   

frame = pd.concat(li, axis=0, ignore_index=True)
frame.to_csv(f"evaluation/eval_data_combined.csv")