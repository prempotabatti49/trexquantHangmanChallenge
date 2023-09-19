import pandas as pd
import numpy as np
import os
import random
import string


def permute_list(lst):
    idx = lst.index('cls')
    lst_len = len(lst[:idx])
    json = {}
    all_permuted_lists = []
    for i in range(lst_len):
        sample_nos = random.sample(range(0, lst_len), i+1)
        new_lst = lst.copy()
        for j in sample_nos:
            new_lst[j] = '_'
        all_permuted_lists.append(new_lst)
    return all_permuted_lists

def pre_process(x, input_data=True):
    p = list(str(x)) + ['cls']
    q = p + [0] * (35 - len(p))
    if input_data:
        all_permuted_lists = permute_list(q)
        return all_permuted_lists
    else:
        return q

def create_input_OHE_features(x, char_mapping):
    input_matrix = []
    for i in range(len(x)):
        base_input_vector = [0] * 28
        if x[i] != 0:
            base_input_vector[char_mapping[x[i]]] = 1
            input_matrix.append(base_input_vector)
        else:
            input_matrix.append(base_input_vector)            
    return input_matrix

def create_output_OHE_features(x, char_mapping):
    output_matrix = []
    base_output_vector = [0] * 26
    for i in range(len(x)):
        if (x[i] == 0) or (x[i] == 'cls'):
            pass
        else:
            base_output_vector[char_mapping[x[i]] - 2] = 1
    return base_output_vector


def read_data():
    with open("words_250000_train.txt", "r") as f:
        df = f.read()
    return df

def prepare_data(x):
    df_final_input = x[0].apply(lambda x: {x: pre_process(x, True)})
    df_final_output = x[0].apply(lambda x: {x: pre_process(x, False)})
    return df_final_input, df_final_output

def get_char_mapping():
    char_mapping = {'_': 0, 'cls': 1}
    ct = 2
    for i in list(string.ascii_lowercase):
        char_mapping[i] = ct
        ct = ct + 1
    return char_mapping

def create_output_features(df_final_output):
    data_output = []
    char_mapping = get_char_mapping()
    for element in df_final_output:
        element = element.copy()
        output, inputs = element.keys(), element.values()
        inputs = list(inputs)
        output = list(output)[0]
        input_matrix = create_output_OHE_features(inputs[0], char_mapping)
        data_output.append([output, input_matrix])
    return  data_output


def create_input_features(df_final_input):
    ## PREDICTOR FEATURE CREATION
    data_input = []
    char_mapping = get_char_mapping()
    for element in df_final_input:
        element = element.copy()
        output, inputs = element.keys(), element.values()
        inputs = list(inputs)
        output = list(output)[0]
        for i in range(len(inputs[0])):
            input_matrix = create_input_OHE_features(inputs[0][i], char_mapping)
            data_input.append([output, input_matrix])

def create_features_dataframe(data_input_df, data_output_df):
    features = data_input_df.merge(data_output_df, on=0, how='left')
    features.columns = ['word', 'input', 'output']
    return features
