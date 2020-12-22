from metrics import manhattan_dist, cosine_sim

from keras.models import Model
from keras.layers import Input, Embedding, Bidirectional, Conv1D, LSTM, Lambda, Subtract, Dense, Dropout, GlobalAveragePooling1D
import keras.backend as K

"""
Extension of Siamese Bidirectional LSTM by adding CNN layer before LSTM.
Uses structures described in Neculoiu, Versteegh, and Rotaru (2016) and Pontes, Huet, Linhares, and Torres-Moreno (2018).
"""

class BiCNNLSTM(Model):
    def __init__(self, embedding_info, num_units, local_context, dist_func):
        super(BiCNNLSTM, self).__init__()
        self.embedding_dim = embedding_info[0]
        self.embedding_matrix = embedding_info[1]
        self.max_seq_length = embedding_info[2]
        self.num_units = num_units
        self.local_context = local_context
        self.dist_func = dist_func

        self.embedding_layer = Embedding(len(self.embedding_matrix), self.embedding_dim, 
                                         weights=[self.embedding_matrix], input_length=self.max_seq_length,
                                         trainable=False)
        
        self.convolution = Conv1D(filters=300, kernel_size=1, strides=self.local_context, activation='tanh')

        self.stacked_bilstm_start = Bidirectional(LSTM(self.num_units, return_sequences=True))
        self.stacked_bilstm = Bidirectional(LSTM(self.num_units, return_sequences=True))
        self.stacked_bilstm_final = Bidirectional(LSTM(self.num_units))

        self.dropout_2 = Dropout(0.2)
        self.dropout_4 = Dropout(0.4)

        self.pool = GlobalAveragePooling1D()

        self.dense = Dense(128)

        self.subtracted = Subtract()

        self.manhattan_dist = Lambda(function=lambda x: manhattan_dist(x))
        self.cosine_sim = Lambda(function=lambda x: cosine_sim(x[0], x[1]))

    def call(self, inputs):
        encoded_left = self.embedding_layer(inputs['left'])
        encoded_right = self.embedding_layer(inputs['right'])

        conv_left = self.convolution(encoded_left)
        conv_right = self.convolution(encoded_right)

        left_bilstm_1 = self.stacked_bilstm_start(conv_left)
        right_bilstm_1 = self.stacked_bilstm_start(conv_right)
        left_bilstm_1d = self.dropout_2(left_bilstm_1)
        right_bilstm_1d = self.dropout_2(right_bilstm_1)

        left_bilstm_2 = self.stacked_bilstm(left_bilstm_1d)
        right_bilstm_2 = self.stacked_bilstm(right_bilstm_1d)
        left_bilstm_2d = self.dropout_2(left_bilstm_2)
        right_bilstm_2d = self.dropout_2(right_bilstm_2)
        
        left_bilstm_3 = self.stacked_bilstm(left_bilstm_2d)
        right_bilstm_3 = self.stacked_bilstm(right_bilstm_2d)
        left_bilstm_3d = self.dropout_2(left_bilstm_3)
        right_bilstm_3d = self.dropout_2(right_bilstm_3)
        
        left_bilstm_4 = self.stacked_bilstm(left_bilstm_3d)
        right_bilstm_4 = self.stacked_bilstm(right_bilstm_3d)
        left_bilstm_4d = self.dropout_4(left_bilstm_4)
        right_bilstm_4d = self.dropout_4(right_bilstm_4)
        
        left_pooled = self.pool(left_bilstm_4d)
        right_pooled = self.pool(right_bilstm_4d)

        left_pooled_d = self.dropout_4(left_pooled)
        right_pooled_d = self.dropout_4(right_pooled)

        left_output = self.dense(left_pooled_d)
        right_output = self.dense(right_pooled_d)

        if self.dist_func == 'manhattan':
            subtracted = self.subtracted([left_output, right_output])
            manhattan_dist = self.manhattan_dist(subtracted)
            output = manhattan_dist

        elif self.dist_func == 'cosine':
            cosine_sim = self.cosine_sim([left_output, right_output])
            output = cosine_sim

        return output