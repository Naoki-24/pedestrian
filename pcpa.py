from action_predict import ActionPredict, DataGenerator
from utils import *
from base_models import AlexNet, C3DNet, C3DNet2, convert_to_fcn
from base_models import I3DNet
from tensorflow.keras.layers import Input, Concatenate, Dense
from tensorflow.keras.layers import GRU, LSTM, GRUCell
from tensorflow.keras.layers import Dropout, LSTMCell, RNN
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Lambda, dot, concatenate, Activation
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from action_predict import attention_3d_block

class MASK_PCPA(ActionPredict):

    """
    MASK_PCPA: pedestrian crossing prediction combining local context with global context
    later fusion
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
        assert self._backbone in ['c3d', 'i3d'], 'Incorrect backbone {}! Should be C3D or I3D'.format(self._backbone)
        self._3dconv = C3DNet if self._backbone == 'c3d' else I3DNet
        self._3dconv2 = C3DNet2 if self._backbone == 'c3d' else I3DNet

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
                                   global_pooling=self._global_pooling,
                                   input_type_list=model_opts['obs_input_type'],
                                   batch_size=model_opts['batch_size'],
                                   shuffle=data_type != 'test',
                                   to_fit=data_type != 'test'), data['crossing']) # set y to None
        else:
            _data = (_data, data['crossing'])

        return {'data': _data,
                'ped_id': data['ped_id'],
                'tte': data['tte'],
                'image': data['image'],
                'data_params': {'data_types': data_types, 'data_sizes': data_sizes},
                'count': {'neg_count': neg_count, 'pos_count': pos_count}}

    def get_model(self, data_params):
        return_sequence = True
        data_sizes = data_params['data_sizes']
        data_types = data_params['data_types']
        network_inputs = []
        encoder_outputs = []
        core_size = len(data_sizes)

        conv3d_model = self._3dconv()
        network_inputs.append(Input(shape=data_sizes[0], name='input_' + data_types[0]))
        conv3d_model = self._3dconv(input_data=network_inputs[0])
        # network_inputs.append(conv3d_model.input)

        attention_size = self._num_hidden_units

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

        encoder_outputs.append(x)


        # image features from mask

        conv3d_model2 = self._3dconv2()
        network_inputs.append(Input(shape=data_sizes[1], name='input2_' + data_types[1]))
        conv3d_model2 = self._3dconv2(input_data=network_inputs[1])

        attention_size = self._num_hidden_units

        if self._backbone == 'i3d':
            x = Flatten(name='flatten_output_2')(conv3d_model2.output)
            x = Dense(name='emb_' + data_types[1] + self._backbone,
                      units=attention_size,
                      activation='sigmoid')(x)
        else:
            x = conv3d_model2.output
            x = Dense(name='emb_' + data_types[1] + self._backbone,
                      units=attention_size,
                      activation='sigmoid')(x)

        encoder_outputs.append(x)

        ##############################################

        for i in range(2, core_size):
            network_inputs.append(Input(shape=data_sizes[i], name='input_' + data_types[i]))
            encoder_outputs.append(self._rnn(name='enc_' + data_types[i], r_sequence=return_sequence)(network_inputs[i]))

        if len(encoder_outputs) > 1:
            att_enc_out = []
            x = Lambda(lambda x: K.expand_dims(x, axis=1))(encoder_outputs[0])
            att_enc_out.append(x) # first output is from 3d conv netwrok
            x = Lambda(lambda x: K.expand_dims(x, axis=1))(encoder_outputs[1])
            att_enc_out.append(x) # second output is from 3d conv netwrok
            # # for recurrent branches apply many-to-one attention block
            for i, enc_out in enumerate(encoder_outputs[2:]):
                x = attention_3d_block(enc_out, dense_size=attention_size, modality='_'+data_types[i])
                x = Dropout(0.5)(x)
                x = Lambda(lambda x: K.expand_dims(x, axis=1))(x)
                att_enc_out.append(x)
            # aplly many-to-one attention block to the attended modalities
            x = Concatenate(name='concat_modalities', axis=1)(att_enc_out)
            encodings = attention_3d_block(x, dense_size=attention_size, modality='_modality')

            #print(encodings.shape)
            #print(weights_softmax.shape)
        else:
            encodings = encoder_outputs[0]

        model_output = Dense(1, activation='sigmoid',
                             name='output_dense',
                             activity_regularizer=regularizers.l2(0.001))(encodings)

        net_model = Model(inputs=network_inputs,
                          outputs=model_output)
        return net_model