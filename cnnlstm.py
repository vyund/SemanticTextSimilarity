from metrics import manhattan_dist, cosine_sim

from keras.models import Model
from keras.layers import Input, Embedding, Conv1D, LSTM, Lambda, Subtract
import keras.backend as K

"""
Siamese CNN-LSTM model as described in Pontes, Huet, Linhares, and Torres-Moreno (2018).
Modifications made to also handle Cosine Similarity, based on equation from Shirkhorshidi, Aghabozorgi, Wah (2015).
"""

class CNNLSTM(Model):
    def __init__(self, embedding_info, num_units, local_context, dist_func):
        super(CNNLSTM, self).__init__()
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
        self.shared_lstm = LSTM(self.num_units)

        self.subtracted = Subtract()

        self.manhattan_dist = Lambda(function=lambda x: manhattan_dist(x))
        self.cosine_sim = Lambda(function=lambda x: cosine_sim(x[0], x[1]))

    def call(self, inputs):
        encoded_left = self.embedding_layer(inputs['left'])
        encoded_right = self.embedding_layer(inputs['right'])

        conv_left = self.convolution(encoded_left)
        conv_right = self.convolution(encoded_right)

        left_output = self.shared_lstm(conv_left)
        right_output = self.shared_lstm(conv_right)

        if self.dist_func == 'manhattan':
            subtracted = self.subtracted([left_output, right_output])
            manhattan_dist = self.manhattan_dist(subtracted)
            output = manhattan_dist

        elif self.dist_func == 'cosine':
            cosine_sim = self.cosine_sim([left_output, right_output])
            output = cosine_sim

        return output