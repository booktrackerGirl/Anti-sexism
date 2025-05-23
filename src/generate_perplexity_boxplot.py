#!/usr/bin/env python

import pandas as pd
import numpy as np
import os
import re
import ast
from datetime import datetime
import matplotlib.pyplot as plt
#imports Seaborn library and assigns shorthand 'sns'
import seaborn as sns

def llama_postprocessing(text):

    if text == 'neither': ## for those instances where it went to the error
        text = ['neither', 'neither', 'neither']

    else:
        ## first changing the string literals
        text = text.replace('2) anti-sexist', 'anti-sexist')
        text = text.replace('3) neither', 'neither')
        text = text.replace('1) sexist', 'sexist')
        text = text.replace('1) anti-sexist', 'anti-sexist')

        text = ast.literal_eval(text)

    return text

def main():

    parser.add_argument('--dataset_folder', help = 'give the name of the folder where our output dataset is', required = True, type = str)
    parser.add_argument('--model', help = 'give the name of the model', type= str)
    parser.add_argument('--output_folder', help = 'give the name of the output folder', required = True, type = str)
    parser.add_argument("--num_samples", type=int, default=3)

    args = parser.parse_args()

    data = pd.read_csv(f'./{args.dataset_folder}/{args.model}.csv', encoding='utf-8')

    if args.model == 'qwen':
        data['roleplay_results'] = qwen_df['roleplay_results'].apply(lambda x: ast.literal_eval(x.lower().replace("assistant\\n", "").replace('3) neither', 'neither').replace('3', 'neither')))
        data['content_results'] = qwen_df['content_results'].apply(lambda x: ast.literal_eval(x.lower().replace("assistant\\n", "").replace('3) neither', 'neither').replace('3', 'neither')))
        data['zeroshot_results'] = qwen_df['zeroshot_results'].apply(lambda x: ast.literal_eval(x.lower().replace("assistant\\n", "").replace('3) neither', 'neither').replace('3', 'neither')))
        data['fewshot_results'] = qwen_df['fewshot_results'].apply(lambda x: ast.literal_eval(x.lower().replace("assistant\\n", "").replace('3) neither', 'neither').replace('3', 'neither')))

    else:
        pass

    
    data['roleplay_ppl'] = data['roleplay_ppl'].apply(lambda x: ast.literal_eval(x))
    data['content_ppl'] = data['content_ppl'].apply(lambda x: ast.literal_eval(x))
    data['zeroshot_ppl'] = data['zeroshot_ppl'].apply(lambda x: ast.literal_eval(x))
    data['fewshot_ppl'] = data['fewshot_ppl'].apply(lambda x: ast.literal_eval(x))

    '''data['Roleplay'] = data['roleplay_ppl'].apply(lambda x: np.average(x))
    data['Content'] = data['content_ppl'].apply(lambda x: np.average(x))
    data['Zero-Shot'] = data['zeroshot_ppl'].apply(lambda x: np.average(x))
    data['Few-Shot'] = data['fewshot_ppl'].apply(lambda x: np.average(x))
    '''



