'''
Author: Baoyun Peng
Date: 2022-02-17 21:44:19
LastEditTime: 2022-02-19 10:13:20
LastEditors: Please set LastEditors
Description: xxx
FilePath: /ai_quality_control/main.py
'''
import os
import pandas as pd
import json
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    description='Get ratio between positive and negative examples for whole dataset')
parser.add_argument('--dataset', type=str,
                    default='XrayDataset', help='dataset')
parser.add_argument('--data-root', type=str,
                    default='/root/QA/20220120', help='root path for dataset')


def get_pos_neg_ratio(data_root):

    T1_df = pd.read_csv(os.path.join(
        data_root, '20220120_study_primary_id.csv'))
    T1_study_primary_id2id_dict = dict(
        zip(T1_df['study_primary_id'], T1_df['id']))
    T1_study_primary_id2hospitalname_dict = dict(
        zip(T1_df['study_primary_id'], T1_df['patient_hospital_name']))
    T2_df = pd.read_csv(os.path.join(data_root, '20220120_content.csv'))
    T3_df = pd.read_csv(os.path.join(
        data_root, '20220120_series_instance_uid.csv'))

    imgs_list = []
    metas_list = []

    hospitals_ratio_dict = {}
    results_dict = {}
    for _, row in T3_df.iterrows():
        study_primary_id = row['study_primary_id']
        img_name = row['series_instance_uid'] + '.png'
        file_path = os.path.join(data_root, 'imgs', img_name)
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

        if hospital_name not in hospitals_ratio_dict:
            hospitals_ratio_dict[hospital_name] = []
        hospitals_ratio_dict[hospital_name].append(scores)

        imgs_list.append(img_name)
        metas_list.append(scores)

    # compute ratio between neg and pos examples for whole dataset
    scores_matrix = np.array(list(results_dict.values()))
    neg_mat = 1*(scores_matrix == 0)
    # pos_mat = 1*(scores_matrix == 1)
    pos_neg_ratio = np.round(1.0*np.sum(neg_mat, axis=0) / neg_mat.shape[0], 3)

    # compute the ratio between neg and pos examples for each hospital
    for hospital, smat in hospitals_ratio_dict.items():
        smat = np.array(smat)
        neg_list = np.sum(1*(smat == 0), axis=0).tolist()
        pos_list = np.sum(1*(smat == 1), axis=0).tolist()
        ratio = [1.0*n/(p+n+1) for n, p in zip(neg_list, pos_list)]
        hospitals_ratio_dict[hospital] = np.round(ratio, 3)
    return pos_neg_ratio, hospitals_ratio_dict, imgs_list, metas_list

if __name__ == "__main__":
    args = parser.parse_args()
    pos_neg_ratio, hospitals_ratio_dict, imgs_list, metas_list = get_pos_neg_ratio(
        args.data_root)

    metas_array = np.array(metas_list, dtype=np.int32)
    # save the ratio and list
    with open(os.path.join(args.data_root, 'images.lst'), 'w') as fout:
        fout.writelines(imgs_list)
    np.savetxt(os.path.join(args.data_root, 'metas.lst'), metas_array, fmt='%d')

    for idx in range(pos_neg_ratio.shape[0]):
        # save the ratio and list
        with open(os.path.join(args.data_root, f'meta_task_{idx}.lst'), 'w') as fout:
            fout.writelines("\n".join( [str(metas_array.shape[0])] + [str(v) for v in metas_array[:,idx].tolist()] ))