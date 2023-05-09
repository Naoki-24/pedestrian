import os 
import csv
import pandas as pd

def get_start_point(set_id, video_id):
    root_path = '../PIE/action'
    action_path = os.path.join(root_path, f'{set_id}_{video_id}.csv')
    with open(action_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        start_data = [i for i in reader]

    for i in range(len(start_data)):
        action = start_data[i][1]
        if action == 'stopped':
            start_data[i][1] = 0
        elif action == 'moving slow':
            start_data[i][1] = 1
        elif action == 'moving fast':
            start_data[i][1] = 2
        elif action == 'decelerating':
            start_data[i][1] = 3
        elif action == 'accelerating':
            start_data[i][1] = 4

    return start_data

def get_full_action(set_id, video_id, frame_len):
    start_data = get_start_point(set_id, video_id)
    data = []
    for i in range(len(start_data)):
        start = int(float(start_data[i][0])*30)
        action = int(start_data[i][1])
        try:
            end = int(float(start_data[i+1][0])*30)
        except IndexError:
            end = frame_len

        actions = [[action] for _ in range(end-start)]
        data.extend(actions)
    return data

if __name__=='__main__':
    set_id = 'set01'
    video_id = 'video_0001'
    frames = 18000
    get_full_action(set_id, video_id, frames)