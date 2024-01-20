from action_predict import ActionPredict
from action_predict import DataGenerator
from action_predict import attention_3d_block
from base_models import C3DNet
from base_models import I3DNet
from utils import *
from utils import *
from tensorflow.keras.layers import Input, Concatenate, Dense, Flatten
from tensorflow.keras.layers import GRUCell
from tensorflow.keras.layers import Dropout, LSTMCell
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Lambda
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import RandomNormal
from pcpa import MASK_PCPA

class LaterFusion(ActionPredict):

    """
    hierfusion MASK_PCPA
    Class init function

    Args:
        num_hidden_units: Number of recurrent hidden layers
        cell_type: Type of RNN cell
        **kwargs: Description
    """

    def __init__(self,
                 num_hidden_units=256,
                 cell_type='gru',
                 **kwargs):
        """
        Class init function

        Args:
            num_hidden_units: Number of recurrent hidden layers
            cell_type: Type of RNN cell
            **kwargs: Description
        """
        super().__init__(**kwargs)
        # Network parameters
        self._num_hidden_units = num_hidden_units
        self._rnn = self._gru if cell_type == 'gru' else self._lstm
        self._rnn_cell = GRUCell if cell_type == 'gru' else LSTMCell
        # assert self._backbone in ['c3d', 'i3d'], 'Incorrect backbone {}! Should be C3D or I3D'.format(self._backbone)
        self._3dconv = C3DNet if self._backbone == 'c3d' else I3DNet


        # dropout = 0.0,
        # dense_activation = 'sigmoid',
        # freeze_conv_layers = False,
        # weights = 'imagenet',
        # num_classes = 1,
        # backbone = 'vgg16',

        # self._dropout = dropout
        # self._dense_activation = dense_activation
        # self._freeze_conv_layers = False
        # self._weights = 'imagenet'
        # self._num_classes = 1
        # self._conv_models = {'vgg16': vgg16.VGG16, 'resnet50': resnet50.ResNet50, 'alexnet': AlexNet}
        # self._backbone ='vgg16'

    def _concat_tensor(self, x, input_tensor, data_type, return_sequence=True):
        current = [x, input_tensor]
        x = Concatenate(name='concat_'+data_type, axis=2)(current)
        x = self._rnn(name='enc_' + data_type, r_sequence=return_sequence)(x)
        return x

    def get_model(self, data_params):
        return_sequence = True
        data_sizes = data_params['data_sizes']
        data_types = data_params['data_types']
        network_inputs = []
        encoder_outputs = []
        att_enc_out = []
        core_size = len(data_sizes)
        num_vis_feature = 2
        attention_size = self._num_hidden_units

        if self._backbone == 'c3d' or self._backbone == 'i3d':
            for i in range(num_vis_feature):
                conv3d_model = self._3dconv()
                network_inputs.append(conv3d_model.input)

                if self._backbone == 'i3d':
                    x = Flatten(name='flatten_output')(conv3d_model.output)
                    x = Dense(name='emb_'+self._backbone,
                            units=attention_size,
                            activation='sigmoid')(x)
                else:
                    x = conv3d_model.output
                    x = Dense(name='emb_'+self._backbone,
                            units=attention_size,
                            activation='sigmoid')(x)

                att_enc_out.append(x)

        # else:
        #     for i in range(0, core_size):
        #         network_inputs.append(Input(shape=data_sizes[i], name='input_' + data_types[i]))

        #     for i in range(num_vis_feature):
        #         x = self._rnn(name='enc_' + str(i) + data_types[i], r_sequence=return_sequence)(network_inputs[i])
        #         encoder_outputs.append(x)

        #     x = self._rnn(name='enc_' + str(num_vis_feature) + data_types[num_vis_feature], r_sequence=return_sequence)(network_inputs[num_vis_feature])
        #     encoder_outputs.append(x)
        #     for i in range(num_vis_feature + 1, core_size):
        #         x = self._concat_tensor(x, network_inputs[i], data_types[i], return_sequence)
        #         encoder_outputs.append(x)

        else:
            # 0:local context, 1:global context, 2: box, 3: pose, 4:speed, 5:look

            for i in range(0, core_size):
                network_inputs.append(Input(shape=data_sizes[i], name='input_' + data_types[i]))

            x = self._rnn(name='enc_' + str(0) + data_types[0], r_sequence=return_sequence)(network_inputs[0])
            encoder_outputs.append(x)
            # x = self._rnn(name='enc_' + str(1) + data_types[1], r_sequence=return_sequence)(network_inputs[1])
            # encoder_outputs.append(x)
            
            pose = self._rnn(name=f'enc_{data_types[2]}', r_sequence=return_sequence)(network_inputs[2])
            # pose2 = self._rnn(name=f'enc_{data_types[3]}_int', r_sequence=return_sequence)(network_inputs[3])
            x = self._concat_tensor(pose, network_inputs[1], data_types[2] + '_' + data_types[1])
            encoder_outputs.append(x)
            x = self._concat_tensor(pose, network_inputs[3], data_types[2] + '_' + data_types[3])
            # x = self._concat_tensor(x, network_inputs[4], data_types[2]+'_'+data_types[3] + '_' + data_types[4])
            # x = self._concat_tensor(x, network_inputs[6], data_types[4]+'_'+data_types[2] + '_' + data_types[5] + '_' + data_types[6])
            encoder_outputs.append(x)

        if len(encoder_outputs) > 1:
            # for recurrent branches apply many-to-one attention block
            for i, enc_out in enumerate(encoder_outputs[0:]):
                x = attention_3d_block(enc_out, dense_size=attention_size, modality='_' + data_types[i])
                x = Dropout(0.5)(x)
                x = Lambda(lambda x: K.expand_dims(x, axis=1))(x)
                att_enc_out.append(x)
            # aplly many-to-one attention block to the attended modalities
            # vis_enc_out = att_enc_out[:num_vis_feature]
            # non_vis_enc_out =  att_enc_out[num_vis_feature:]
            # last_enc_out = [Concatenate(name='vis_concat', axis=1)(vis_enc_out), non_vis_enc_out[0], non_vis_enc_out[1]]
            # x = Concatenate(name='concat_modalities', axis=1)(last_enc_out)
            # encodings = attention_3d_block(x, dense_size=attention_size, modality='_modality')
            x = Concatenate(name='concat_modalities', axis=1)(att_enc_out)
            encodings = attention_3d_block(x, dense_size=attention_size, modality='_modality')

            # print(encodings.shape)
            # print(weights_softmax.shape)
        else:
            encodings = encoder_outputs[0]
            encodings = attention_3d_block(encodings, dense_size=attention_size, modality='_modality')

        encodings = Dropout(0.5)(encodings)

        # seed = RandomNormal(mean=0.0, stddev=0.05, seed=0)
        model_output = Dense(1, activation='sigmoid',
                             name='output_dense',
                             activity_regularizer=regularizers.l2(0.001))(encodings)
        # model_output = Dense(1, activation='sigmoid',
        #                      name='output_dense',
        #                      activity_regularizer=regularizers.l2(0.001),
        #                      kernel_initializer=seed)(encodings)

        net_model = Model(inputs=network_inputs,
                          outputs=model_output)
        return net_model


class HierarchicalFusion(ActionPredict):

    """
    hierfusion MASK_PCPA
    Class init function

    Args:
        num_hidden_units: Number of recurrent hidden layers
        cell_type: Type of RNN cell
        **kwargs: Description
    """

    def __init__(self,
                 num_hidden_units=256,
                 cell_type='gru',
                 **kwargs):
        """
        Class init function

        Args:
            num_hidden_units: Number of recurrent hidden layers
            cell_type: Type of RNN cell
            **kwargs: Description
        """
        super().__init__(**kwargs)
        # Network parameters
        self._num_hidden_units = num_hidden_units
        self._rnn = self._gru if cell_type == 'gru' else self._lstm
        self._rnn_cell = GRUCell if cell_type == 'gru' else LSTMCell
        # assert self._backbone in ['c3d', 'i3d'], 'Incorrect backbone {}! Should be C3D or I3D'.format(self._backbone)
        self._3dconv = C3DNet if self._backbone == 'c3d' else I3DNet

    def _concat_tensor(self, x, input_tensor, data_type, return_sequence=True, axis=2):
        current = [x, input_tensor]
        x = Concatenate(name='concat_'+ data_type, axis=axis)(current)
        x = self._rnn(name='enc_' + data_type, r_sequence=return_sequence)(x)
        return x

    def get_model(self, data_params):
        return_sequence = True
        data_sizes = data_params['data_sizes']
        data_types = data_params['data_types']
        network_inputs = []
        encoder_outputs = []
        att_enc_out = []
        core_size = len(data_sizes)
        num_vis_feature = 2
        attention_size = self._num_hidden_units

        if self._backbone == 'c3d' or self._backbone == 'i3d':
            for i in range(num_vis_feature):
                conv3d_model = self._3dconv()
                network_inputs.append(conv3d_model.input)

                if self._backbone == 'i3d':
                    x = Flatten(name='flatten_output')(conv3d_model.output)
                    x = Dense(name='emb_'+self._backbone,
                            units=attention_size,
                            activation='sigmoid')(x)
                else:
                    x = conv3d_model.output
                    x = Dense(name='emb_'+self._backbone,
                            units=attention_size,
                            activation='sigmoid')(x)

                att_enc_out.append(x)

        else:
            # 0:local context, 1:global context, 2: box, 3: pose, 4:look, 5:speed

            for i in range(0, core_size):
                network_inputs.append(Input(shape=data_sizes[i], name='input_' + data_types[i]))

            x = self._rnn(name='enc_' + str(0) + data_types[0], r_sequence=return_sequence)(network_inputs[0])
            encoder_outputs.append(x)
            x = self._rnn(name='enc_' + str(1) + data_types[1], r_sequence=return_sequence)(network_inputs[1])
            encoder_outputs.append(x)
            
            pose = self._rnn(name='enc_' + str(3) + data_types[4], r_sequence=return_sequence)(network_inputs[4])
            p_b = self._concat_tensor(pose, network_inputs[3], data_types[4]+'_'+data_types[3], return_sequence)
            encoder_outputs.append(p_b)
            x = self._concat_tensor(pose, network_inputs[2], data_types[4] + '_' + data_types[2])
            x = self._concat_tensor(x, network_inputs[5], data_types[4] + '_' + data_types[2] + '_' + data_types[5])
            # x = self._concat_tensor(x, network_inputs[6], data_types[4] + '_' + data_types[2] + '_' + data_types[5] + '_' + data_types[6])

            # x = self._concat_tensor(p_b, x, 'vis_feature')
            encoder_outputs.append(x)

        if len(encoder_outputs) > 1:
            # for recurrent branches apply many-to-one attention block
            for i, enc_out in enumerate(encoder_outputs[0:]):
                x = attention_3d_block(enc_out, dense_size=attention_size, modality='_' + data_types[i])
                x = Dropout(0.5)(x)
                x = Lambda(lambda x: K.expand_dims(x, axis=1))(x)
                att_enc_out.append(x)
            # aplly many-to-one attention block to the attended modalities

            vis_feature = Concatenate(name='concat_vis', axis=1)(att_enc_out[:2])
            non_vis_feature = Concatenate(name='concat_non_vis', axis=1)(att_enc_out[2:])

            encodings = Concatenate(name='concat_modalities', axis=1)([vis_feature, non_vis_feature])
            # encodings = Concatenate(name='concat_modalities', axis=1)(att_enc_out)
            encodings = attention_3d_block(encodings, dense_size=attention_size, modality='_modality')

        else:
            encodings = encoder_outputs[0]
            encodings = attention_3d_block(encodings, dense_size=attention_size, modality='_modality')

        encodings = Dropout(0.5)(encodings)
        model_output = Dense(1, activation='sigmoid',
                             name='output_dense',
                             activity_regularizer=regularizers.l2(0.001))(encodings)

        net_model = Model(inputs=network_inputs,
                          outputs=model_output)
        return net_model

class Pretrain_PIE(ActionPredict):
    def __init__(self, global_pooling='avg', regularizer_val=0.0001, backbone='vgg16', **kwargs):
        super().__init__(global_pooling, regularizer_val, backbone, **kwargs)
        self.model_folder = 'data/models/jaad/Pretrain_PIE/08Feb2023-22h10m22s'

    def get_data(self, data_type, data_raw, model_opts):
        assert model_opts['obs_length'] == 16
        model_opts['normalize_boxes'] = False
        self._generator = model_opts.get('generator', False)
        data_type_sizes_dict = {}
        process = model_opts.get('process', True)
        dataset = model_opts['dataset']
        data, neg_count, pos_count = self.get_data_sequence(data_type, data_raw, model_opts)

        data_type_sizes_dict['box'] = data['box'].shape[1:]
        if 'speed' in data.keys():
            data_type_sizes_dict['speed'] = data['speed'].shape[1:]
        # if 'context_cnn' in data.keys():
        #     data_type_sizes_dict['context_cnn'] = data['context_cnn'].shape[1:]

        # Store the type and size of each image
        _data = []
        data_sizes = []
        data_types = []

        model_opts_3d = model_opts.copy()

        for d_type in model_opts['obs_input_type']:
            if 'local' in d_type or 'context' in d_type or 'mask' in d_type:
                if self._backbone == 'c3d':
                    model_opts_3d['target_dim'] = (112, 112)
                model_opts_3d['process'] = False
                features, feat_shape = self.get_context_data(model_opts_3d, data, data_type, d_type)
            elif 'pose' in d_type:
                path_to_pose, _ = get_path(save_folder='poses',
                                           dataset=dataset,
                                           save_root_folder='data/features')
                features = get_pose(data['image'],
                                    data['ped_id'],
                                    data_type=data_type,
                                    file_path=path_to_pose,
                                    dataset=model_opts['dataset'])
                feat_shape = features.shape[1:]
            elif 'boxY' in d_type:
                features = data['box'][0:, 0:, 1::2]
                feat_shape = features.shape[1:]
            elif 'boxX' in d_type:
                features = data['box'][0:, 0:, 0::2]
                feat_shape = features.shape[1:]
            else:
                features = data[d_type]
                feat_shape = features.shape[1:]
            _data.append(features)
            data_sizes.append(feat_shape)
            data_types.append(d_type)
        # create the final data file to be returned
        if self._generator:
            _data = (DataGenerator(data=_data,
                                   labels=data['crossing'],
                                   data_sizes=data_sizes,
                                   process=process,
                                   global_pooling=None,
                                   input_type_list=model_opts['obs_input_type'],
                                   batch_size=model_opts['batch_size'],
                                   shuffle=data_type != 'test',
                                   to_fit=data_type != 'test'), data['crossing'])  # set y to None
        # global_pooling=self._global_pooling,
        else:
            _data = (_data, data['crossing'])

        return {'data': _data,
                'ped_id': data['ped_id'],
                'tte': data['tte'],
                'image': data['image'],
                'data_params': {'data_types': data_types, 'data_sizes': data_sizes},
                'count': {'neg_count': neg_count, 'pos_count': pos_count}}

    def get_model(self, param):
        model_path = os.path.join(self.model_folder, 'model.h5')
        model = load_model(model_path, compile=False)
        return model

def action_prediction(model_name):
    for cls in ActionPredict.__subclasses__():
        if cls.__name__ == model_name:
            return cls
    raise Exception('Model {} is not valid!'.format(model_name))