from jaad_data import JAAD
import pandas as pd
import os



def reshape_pid(database):
    for i in range(len(database['pid'])):
        database['pid'][i] = database['pid'][i][0][0]
    return database

def diff_predict(output1, output2):
    output = output2[(output1 == output2).all(axis=1) == False]
    return output

def search_data(dtype, output, database):
    pids = list(output.loc[:,'pid'])
    images = list(output.loc[:,'image'])
    gts = list(output.loc[:,'gt'])
    ys = list(output.loc[:,'y'])
    search_list = []
    for pid, image, gt, y in zip(pids, images,gts,ys):
        pid = pid.split('\'')[1]
        ped_index = database['pid'].index(pid)
        look_list = sum(database['look'][ped_index], [])
        video_id = image.split('/')[3]
        predict_point = database['image'][ped_index].index(image)
        obs_start = predict_point - 16
        obs_end = predict_point - 1
        obs_look = look_list[obs_start:obs_end]
        obs_is_look = 1 in obs_look
        is_look = 1 in look_list
        action = database['action'][ped_index][obs_start:obs_end]
        d = {
            'pid': pid,
            'video_id': video_id,
            'image': image,
            'gt': gt,
            'y': y,
            'look': look_list,
            'obs_look': obs_look,
            'obs is look': 1 if obs_is_look else 0,
            'is look': 1 if is_look else 0,
            'action': action[0] if len(set(action)) else action
        }
        search_list.append(d)
        
        # if not same_pid == pid[i]:
        #     bbox_on_image(database['bbox'][ped_index][predict_point], pid[i],image_path)
        #     same_pid = pid[i]

    saved_file_path = os.path.join('./analysis', dtype)
    if not os.path.isdir(saved_file_path):
        os.mkdir(saved_file_path)
    df = pd.DataFrame(search_list)
    df.to_csv(os.path.join(saved_file_path,'analysis_box.csv'), index=False)

    return

def get_database():
    opt = {
        'fstride': 1,
        'sample_type': 'beh',
        'seq_type': 'crossing',
        'data_split_type': 'default'
    }
    jaad = JAAD('../img')
    test_data = jaad.generate_data_trajectory_sequence('test', **opt)

    return test_data

if __name__== '__main__':
    database = get_database()

    analysis_path = 'data/models/jaad/PCPA/Hierarchical/result.csv'
    compared_path = 'data/models/jaad/PCPA/Hierarchical_no_box/result.csv'
    analysis_type = 'box'

    analysis = pd.read_csv(analysis_path)
    compared = pd.read_csv(compared_path)
    database = reshape_pid(database)

    diff_data = diff_predict(compared, analysis)

    search_data(analysis_type, diff_data ,database)