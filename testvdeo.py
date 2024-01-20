from tensorflow.keras.preprocessing import image
from pie_data import PIE
from jaad_data import JAAD
from action_predict import ActionPredict
import matplotlib.pyplot as plt
import numpy as np
import pickle
import csv

def set_video_num():
    video_list = [4, 3, 19, 16, 2, 9]
    set_list = ['set01', 'set02', 'set03', 'set04', 'set05', 'set06']
    return set_list, video_list

def plot_speed_pid(annotations):
    set_list, video_list = set_video_num()
    for vset, video in zip(set_list, video_list):
        for i in range(1,video+1):
            vid = 'video_'+str(i).zfill(4)
            vid_annots = annotations[vset][vid]['vehicle_annotations']
            speed = []
            for val in vid_annots.values():
                speed.append(val['OBD_speed'])
            x = np.arange(len(speed))
            plt.plot(x, speed)
            plt.savefig(f'speed/speed_{vset}_{vid}.jpg')
            plt.clf()

def get_speed(annotations, data_type):
    set_list, video_list = set_video_num()
    data = {}
    if data_type == 'pie':
        for vset, video in zip(set_list, video_list):
            speed_video = {}
            for i in range(1,video+1):
                vid = 'video_'+str(i).zfill(4)
                vid_annots = annotations[vset][vid]['vehicle_annotations']
                speed = []
                for val in vid_annots.values():
                    speed.append(val['OBD_speed'])
                speed_video[vid] = speed
            data[vset] = speed_video

    elif data_type =='jaad':
        for i in range(346):
            vid = f'video_{str(i+1).zfill(4)}'
            data[vid] = annotations[vid]['vehicle_annotations'].values()

    return data

def get_actions(speed_data):
    set_list, video_list = set_video_num()
    actions = {}
    for vset, num_video in zip(set_list, video_list):
        print(vset)
        for video in range(1, num_video+1):
            vid = 'video_'+str(video).zfill(4)
            print(vid)
            action_video = {}
            s_prev = 0
            start = 0
            speed = speed_data[vset][vid]
            action = np.zeros(len(speed))
            prev_state = 0
            state = 0
            i = 0
            while i <= len(speed):
                s = speed[i]
                if s == 0:
                    action[i] = 0
                else:
                    state = s - s_prev
                    if state == prev_state:
                        continue
                    else:
                        if prev_state > 0:
                            action[start:i+1] = 4
                        elif prev_state < 0:
                            action[start:i+1] = 3
                        else:
                            if i - start >= 30:
                                if s > 15:
                                    action[start:i+1] = 2
                                else:
                                    action[start:i+1] = 1
                            else:
                                continue
                s_prev = s
                prev_state = state
                i += 1
                start = i
            action_video[vid] = action
            actions[vset] = action_video
    
    saved_path = '../PIE/data_cache/actions.plk'
    with open(saved_path, 'wb') as f:
        pickle.dump(actions, f, pickle.HIGHEST_PROTOCOL)

    return actions

def state(diff):
    if diff > 0:
        return 'acc'
    elif diff < 0:
        return 'deacc'
    else:
        return 'const'


def get_action1(speed_data):
    vset = 'set01'
    vid = 'video_'+str(1).zfill(4)
    print(vid)
    action_video = {}
    s_prev = 0
    start = 0
    speed = speed_data[vset][vid]
    action = np.zeros(len(speed))
    prev_state = 0
    state = 0
    i = 0
    while i <= len(speed):
        s = speed[i]
        if s == 0:
            action[i] = 0
        else:
            state = state(s - s_prev)
            if state == prev_state:
                continue
            else:
                if prev_state > 0:
                    action[start:i+1] = 4
                elif prev_state < 0:
                    action[start:i+1] = 3
                else:
                    if i - start >= 30:
                        if s > 15:
                            action[start:i+1] = 2
                        else:
                            action[start:i+1] = 1
                    else:
                        continue
        s_prev = s
        prev_state = state
        i += 1
        start = i
    action_video[vid] = action
    actions[vset] = action_video
    
    saved_path = '../PIE/data_cache/actions.plk'
    with open(saved_path, 'wb') as f:
        pickle.dump(actions, f, pickle.HIGHEST_PROTOCOL)
# i = 4
# vset = 'set03'
# vid = 'video_'+str(i).zfill(4)
# vid_annots = annotations[vset][vid]['vehicle_annotations']
# speed = []
# for val in vid_annots.values():
#     speed.append(val['OBD_speed'])

# speed = np.array(speed)
# speed = speed[5000:5500]
# plt.plot(np.arange(len(speed)), speed)
# plt.savefig('speed_5000-5500.jpg')
# grad = np.gradient(speed)
# pass
def count_action(data, data_type):
    speed = []
    if data_type == 'jaad':
        for v in data.values():
            speed.extend(v)

    elif data_type == 'pie':
        set_list, video_list = set_video_num()
        for vset, video in zip(set_list, video_list):
            for i in range(1, video+1):
                vid = 'video_'+str(i).zfill(4)
                speed.extend(data[vset][vid])

    stop = speed.count(0)
    mv_slow = speed.count(1)
    mv_fast = speed.count(2)
    deacc = speed.count(3)
    acc = speed.count(4)

    print(f'stop: {stop}')
    print(f'move slow: {mv_slow}')
    print(f'move fast: {mv_fast}')
    print(f'deacc: {deacc}')
    print(f'acc: {acc}')


data_type = 'pie'

if data_type == 'pie':
    imdb = PIE(data_path='../PIE')
elif data_type == 'jaad':
    imdb = JAAD(data_path='../JAAD')

annotations = imdb.generate_database()
speed = get_speed(annotations, data_type)
speed = speed['set01']['video_0001']
grad = np.gradient(speed)
np.savetxt('./grad.csv', grad, fmt='%.3f', delimiter=',')
pass