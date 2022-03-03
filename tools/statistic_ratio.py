'''
Author: Baoyun Peng
Date: 2022-02-17 21:44:19
LastEditTime: 2022-03-03 20:45:56
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
parser.add_argument('--data-root', type=str,
                    default='/root/QA/20220120', help='root path for dataset')
parser.add_argument('--json-file', type=str,
                    default='content_result.json', help='json_file for dataset')
parser.add_argument('--img-prefix', type=str,
                    default='/root/QA/image_batch12', help='prefix of image folder')


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
        hospitals_ratio_dict[hospital] = np.round(ratio, 3).tolist()
    return pos_neg_ratio, hospitals_ratio_dict, imgs_list, metas_list


def parse_json(json_path, img_prefix):
    with open(json_path) as fin:
        all_data = json.load(fin)
    imgs_list = []
    metas_list = []
    for key, scores in all_data.items():
        if not os.path.exists(os.path.join(img_prefix, key.strip()+'.png')) or "" in scores['content_score']:
            continue
        scores = [int(s) for s in scores['content_score']]
        scores = [1 if s > 0 else 0 for s in scores]
        imgs_list.append(key.strip()+'.png')
        metas_list.append(scores)

    # compute ratio between neg and pos examples for whole dataset
    scores_matrix = np.array(metas_list, dtype=np.int32)
    pos_mat = 1*(scores_matrix == 0)
    # pos_mat = 1*(scores_matrix == 1)
    pos_neg_ratio = np.round(1.0*np.sum(pos_mat, axis=0) / pos_mat.shape[0], 3)
    return pos_neg_ratio, imgs_list, metas_list

# divide the imgs_list and metas_list into train and test


def divide_train_test(imgs_list, metas_array, div_ratio=0.7):
    # divided dataset into train and test with 70:30 ratio
    all_idxs = np.array(range(len(imgs_list)), dtype=np.int32).tolist()
    train_idxs = np.random.choice(all_idxs, int(
        len(all_idxs)*div_ratio), replace=False)
    train_idxs.tolist().sort()
    test_idxs = list(set(all_idxs) - set(train_idxs))
    test_idxs.sort()
    train_list = [imgs_list[idx] for idx in train_idxs]
    train_meta = [metas_array[idx] for idx in train_idxs]
    test_list = [imgs_list[idx] for idx in test_idxs]
    test_meta = [metas_array[idx] for idx in test_idxs]
    return train_list, np.array(train_meta), test_list, np.array(test_meta)


if __name__ == "__main__":
    args = parser.parse_args()
    # pos_neg_ratio, hospitals_ratio_dict, imgs_list, metas_list = get_pos_neg_ratio(
    #     args.data_root)
    # # dump the hispitals results into json file
    # with open(os.path.join(args.data_root, 'stats_hospital.json'), 'w') as outfile:
    #     json.dump(hospitals_ratio_dict, outfile)
    # np.savetxt(os.path.join(args.data_root, 'stats_pos_ratio.txt'), pos_neg_ratio, fmt='%.3f')

    json_path = os.path.join(args.data_root, args.json_file)
    pos_neg_ratio, imgs_list, metas_list = parse_json(
        json_path, args.img_prefix)
    np.savetxt(os.path.join(args.data_root, 'stats_pos_ratio.txt'),
               pos_neg_ratio, fmt='%.3f')

    metas_array = np.array(metas_list, dtype=np.int32)
    # save the ratio and list
    with open(os.path.join(args.data_root, 'images.lst'), 'w') as fout:
        fout.writelines(imgs_list)
    np.savetxt(os.path.join(args.data_root, 'metas.lst'),
               metas_array, fmt='%d')

    train_list, train_meta, test_list, test_meta = divide_train_test(
        imgs_list, metas_array, div_ratio=0.7)
    # write the results into file
    with open(os.path.join(args.data_root, 'train_list.txt'), 'w') as fout:
        fout.writelines("\n".join(train_list))
    np.savetxt(os.path.join(args.data_root, 'train_meta.txt'),
               train_meta, fmt='%d')
    # write the results into file
    with open(os.path.join(args.data_root, 'test_list.txt'), 'w') as fout:
        fout.writelines("\n".join(test_list))
    np.savetxt(os.path.join(args.data_root, 'test_meta.txt'),
               test_meta, fmt='%d')

    selected_idxs = []
    for idx in range(pos_neg_ratio.shape[0]):
        if pos_neg_ratio[idx] > 0.15 and pos_neg_ratio[idx] < 0.85:
            selected_idxs.append(idx)
        task_train_meta = [str(v) for v in train_meta[:, idx].tolist()]
        task_test_meta = [str(v) for v in test_meta[:, idx].tolist()]
        with open(os.path.join(args.data_root, f'train_meta_task_{str(idx)}.txt'), 'w') as fout:
            fout.writelines(
                "\n".join([str(len(train_meta))] + task_train_meta))
        with open(os.path.join(args.data_root, f'test_meta_task_{str(idx)}.txt'), 'w') as fout:
            fout.writelines("\n".join([str(len(test_meta))] + task_test_meta))

    selected_train_meta = train_meta[:, selected_idxs]
    selected_test_meta = test_meta[:, selected_idxs]
    np.savetxt(os.path.join(args.data_root, 'selected_train_meta.txt'),
               selected_train_meta, fmt='%d')
    np.savetxt(os.path.join(args.data_root, 'selected_test_meta.txt'),
               selected_test_meta, fmt='%d')
