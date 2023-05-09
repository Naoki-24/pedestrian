from action_predict import action_prediction
from jaad_data import JAAD
import os
import sys
import yaml

def test_model(saved_files_path=None):

    with open(os.path.join(saved_files_path, 'configs.yaml'), 'r') as yamlfile:
        opts = yaml.safe_load(yamlfile)
    print(opts)
    model_opts = opts['model_opts']
    net_opts = opts['net_opts']
    data_opts = {'min_track_size':}

    imdb = JAAD(data_path=os.environ.copy()['JAAD_PATH'])
    method_class = action_prediction(model_opts['model'])(**net_opts)
    #beh_seq_train = imdb.generate_data_trajectory_sequence('train', **data_opts)
    #saved_files_path = method_class.train(beh_seq_train, **train_opts, model_opts=model_opts)

    beh_seq_test = imdb.generate_data_trajectory_sequence('test', **data_opts)
    acc, auc, f1, precision, recall = method_class.test(beh_seq_test, saved_files_path)


if __name__ == '__main__':
    saved_files_path = sys.argv[1]
    test_model(saved_files_path=saved_files_path)