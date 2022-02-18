'''
Author: Baoyun Peng
Date: 2022-02-17 21:44:19
LastEditTime: 2022-02-18 09:57:13
LastEditors: Please set LastEditors
Description: xxx
FilePath: /ai_quality_control/main.py
'''
import os
import pandas as pd
import pdb
import json
import numpy as np

img_prefix = '../dataset_20220120'
T1_df = pd.read_csv('20220120_study_primary_id.csv')
T1_study_primary_id2id_dict = dict(
    zip(T1_df['study_primary_id'], T1_df['id']))
T1_study_primary_id2hospitalname_dict = dict(
    zip(T1_df['study_primary_id'], T1_df['patient_hospital_name']))
T2_df = pd.read_csv('20220120_content.csv')
T3_df = pd.read_csv('20220120_series_instance_uid.csv')

hospitals_scores_dict = {}

results_dict = {}
for idx, row in T3_df.iterrows():
    study_primary_id = row['study_primary_id']
    img_name = row['series_instance_uid'] + '.png'
    file_path = os.path.join(img_prefix, img_name)
    # get the corresponding sub_task_primary_id from T1 since T1.id=T2.sub_task_primary_id
    sub_task_primary_id = T1_study_primary_id2id_dict[study_primary_id]
    hospital_name = T1_study_primary_id2hospitalname_dict[study_primary_id]

    # get all sub_task_primary_id columns from T2 by sub_task_primary_id
    selected_columns = T2_df[T2_df['sub_task_primary_id']
                             == sub_task_primary_id]
    scores_list = selected_columns['option_content_score']
    if len(scores_list) == 0 or not os.path.exists(file_path):
        continue

    # extract scores from string, convert it to interger list
    scores = [json.loads(v)[0]['score'] for i, v in scores_list.items()]
    scores = [-1 if v == '' else int(v) for v in scores]
    scores = [1 if s == 5 else s for s in scores]
    results_dict[file_path] = scores

    if hospital_name not in hospitals_scores_dict:
        hospitals_scores_dict[hospital_name] = []
    hospitals_scores_dict[hospital_name].append(scores)

# compute ratio between neg and pos examples for whole dataset 
scores_matrix = np.array(list(results_dict.values()))
neg_mat = 1*(scores_matrix == 0)
pos_mat = 1*(scores_matrix == 1)
print(np.sum(neg_mat + pos_mat, axis=0))
print(1.0*np.sum(neg_mat, axis=0) / neg_mat.shape[0])

# compute the ratio between neg and pos examples for each hospital
for hospital, smat in  hospitals_scores_dict.items():
    smat = np.array(smat)
    neg_list = np.sum(1*(smat == 0), axis=0).tolist()
    pos_list = np.sum(1*(smat == 1), axis=0).tolist()
    ratio = [ 1.0*n/(p+n+1) for n,p in zip(neg_list, pos_list) ]
    hospitals_scores_dict[hospital] = [ratio]
print(hospitals_scores_dict)