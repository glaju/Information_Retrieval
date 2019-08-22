from __future__ import print_function
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, Concatenate, concatenate, add, multiply, LeakyReLU, BatchNormalization, Activation
from keras.layers.wrappers import Bidirectional
from keras.optimizers import Adam
from keras import regularizers
from keras.utils import plot_model
from keras.layers.core import Dropout
import numpy as np
from keras.callbacks import TensorBoard
from keras.callbacks import ReduceLROnPlateau
from time import time
import re
import operator
import os
import json

def write_log(callback, name, logs, batch_no):
    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = logs
    summary_value.tag = name
    callback[0].writer.add_summary(summary, batch_no)
    callback[0].writer.flush()

if __name__ == '__main__':
    with open('./img_features/img_features.json', 'r') as f:
        feat = json.load(f)
        print(np.array(feat['image1']).shape)  # You may want to convert the list format into numpy
    f.close()

    feat_sum2 = dict()
    mi = 10000
    ma = 0
    for key, value in feat.items():
        feat_sum2[key] = np.array(value).flatten()
        mi_ = min(feat_sum2[key])
        ma_ = max(feat_sum2[key])
        if mi_ < mi:
            mi = mi_
        if ma_ > ma:
            ma = ma_

    feat_sum = dict()
    for key, value in feat_sum2.items():
        # mi = min(value)
        # ma = max(value)
        feat_sum[key] = np.array([(x - mi) / (ma - mi) for x in value])

    del feat
    print(feat_sum['image1'])
    print(np.array(feat_sum['image1']).shape)

    del feat_sum2

    batch_size = 256
    epochs = 100
    latent_dim = 128
    num_samples = 6795
    data_path = './qa.894.raw.train.txt'

    # Reading the training data
    input_texts = []
    target_texts = []
    qa_target_texts = []
    input_words = set()
    target_words = set()
    qa_target_words = set()
    input_images = []
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for i in range(0, len(lines), 2):
        input_text = lines[i].strip().split(' ')
        target_text = [None] * (len(lines[i].strip().split(' ')) + 2)
        target_text[0] = '\t'
        target_text[1:len(lines[i].strip().split(' ')) + 1] = lines[i].strip().split(' ')
        target_text[len(lines[i].strip().split(' ')) + 1] = '\n'
        input_images.append(input_text[len(input_text) - 2])
        input_text[len(input_text) - 2] = 'image'
        target_text[len(target_text) - 3] = 'image'
        qa_target_text = [None] * (len(lines[i + 1].strip().split(' ')) + 2)
        qa_target_text[0] = '\t'
        qa_target_text[1:len(lines[i + 1].strip().split(' ')) + 1] = re.split(', ', lines[i + 1].strip())
        qa_target_text[len(lines[i + 1].strip().split(' ')) + 1] = '\n'
        input_texts.append(input_text)
        target_texts.append(target_text)
        qa_target_texts.append(qa_target_text)

        for word in input_text:
            if word not in input_words:
                input_words.add(word)
        for word in target_text:
            if word not in target_words:
                target_words.add(word)
        for word in qa_target_text:
            if word not in qa_target_words:
                qa_target_words.add(word)

    input_words = sorted(list(input_words))
    target_words = sorted(list(target_words))
    qa_target_words = sorted(list(qa_target_words))
    num_encoder_tokens = len(input_words)
    num_decoder_tokens = len(target_words)
    num_qa_tokens = len(qa_target_words)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])
    max_qa_seq_length = max([len(txt) for txt in qa_target_texts])

    input_token_index = dict(
        [(word, i) for i, word in enumerate(input_words)])
    target_token_index = dict(
        [(word, i) for i, word in enumerate(target_words)])
    qa_token_index = dict(
        [(word, i) for i, word in enumerate(qa_target_words)])

    encoder_input_data = np.zeros(
        (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
        dtype='float32')
    image_input_data = np.zeros((len(input_texts), 100352), dtype='float32')
    decoder_input_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')
    decoder_target_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')
    qa_input_data = np.zeros(
        (len(input_texts), max_qa_seq_length, num_qa_tokens),
        dtype='float32')
    qa_target_data = np.zeros(
        (len(input_texts), max_qa_seq_length, num_qa_tokens),
        dtype='float32')

    # one-hot encoding of the questions, target of the autoencoder and answers
    for i, (input_text, target_text, qa_text) in enumerate(zip(input_texts, target_texts, qa_target_texts)):
        for t, word in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[word]] = 1.
            image_input_data[i, :] = feat_sum[input_images[i]]
        # I used teacher forcing, so the decoder target is one step ahead the decoder input
        for t, word in enumerate(target_text):
            decoder_input_data[i, t, target_token_index[word]] = 1.
            if t > 0:
                decoder_target_data[i, t - 1, target_token_index[word]] = 1.
        # I used teacher forcing, so the answering target is one step ahead the answering input
        for t, word in enumerate(qa_text):
            qa_input_data[i, t, qa_token_index[word]] = 1.
            if t > 0:
                qa_target_data[i, t - 1, qa_token_index[word]] = 1.

    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    # the encoder which is a bidirectional LSTM creates the hidden representation
    # encoder = Bidirectional(LSTM(latent_dim, return_state=True,
    #                              dropout=0.5, recurrent_dropout=0.5,
    #                              kernel_regularizer=regularizers.l2(0.0001),
    #                              recurrent_regularizer=regularizers.l2(0.0001)))
    encoder = Bidirectional(LSTM(latent_dim, return_state=True,
                                 dropout=0.5, recurrent_dropout=0.5))
    encoder_outputs, state_fh, state_fc, state_bh, state_bc = encoder(encoder_inputs)

    # concatenation of the forward and backward states because the decoder and answering parts are unidirectional LSTMs
    state_h = Concatenate()([state_fh, state_bh])
    state_c = Concatenate()([state_fc, state_bc])
    encoder_states = [state_h, state_c]

    # The decoder gets the previous target as input (teacher forcing)
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder_lstm = LSTM(latent_dim * 2, return_sequences=True, return_state=True,
                        dropout=0.5, recurrent_dropout=0.5)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    # Dense layer with softmax activation is used to predict the category of the output
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    image_input = Input(shape=(100352,))
    # image_features = Dense(units=latent_dim*4, activation='elu')(image_input)

    image_features = Dense(units=latent_dim * 2, kernel_regularizer=regularizers.l1_l2(0.00005, 0.00005))(image_input)
    image_features = BatchNormalization()(image_features)
    image_features = Activation('elu')(image_features)
    image_features = Dropout(0.5)(image_features)
    print(state_h)
    merged_c = concatenate([image_features, state_c])
    merged_h = concatenate([image_features, state_h])
    merged = [merged_h, merged_c]

    # The answering gets the previous target as input (teacher forcing)
    qa_inputs = Input(shape=(None, num_qa_tokens))
    # qa_lstm = LSTM(latent_dim * 4, return_sequences=True, return_state=True, dropout=0.5, recurrent_dropout=0.5,
    #                kernel_regularizer=regularizers.l2( 0.001),
    #                recurrent_regularizer=regularizers.l2( 0.001))
    qa_lstm = LSTM(latent_dim * 4, return_sequences=True, return_state=True, dropout=0.5, recurrent_dropout=0.5)
    qa_outputs, _, _ = qa_lstm(qa_inputs, initial_state=merged)
    # Dense layer with softmax activation is used to predict the category of the output
    qa_dense = Dense(num_qa_tokens, activation='softmax')
    qa_outputs = qa_dense(qa_outputs)

    # Definition of the autoencoder and of the question answering
    autoencoder = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    question_answer = Model([encoder_inputs, image_input, qa_inputs], qa_outputs)

    # I used categorical crossentropy because of the one-hot encoding
    autoencoder.compile(optimizer=Adam(lr=0.01), loss='categorical_crossentropy')
    question_answer.compile(optimizer=Adam(lr=0.01), loss='categorical_crossentropy')
    print(autoencoder.summary())
    print(question_answer.summary())

    callback = [TensorBoard(log_dir="logs/{}".format(time()))]
    callback[0].set_model(autoencoder)

    for e in range(epochs):
        print(e)
        # 27 is calculated by dividing the number of training questions by the batch size (6795/256)
        for b in range(27):
            batch_indices = np.random.randint(low=0, high=6795, size=batch_size)
            enc_inp = np.array([encoder_input_data[i, :, :] for i in batch_indices])
            #         dec_inp = np.array([decoder_input_data[i, :, :] for i in batch_indices])
            #         dec_tar = np.array([decoder_target_data[i, :, :] for i in batch_indices])
            qa_inp = np.array([qa_input_data[i, :, :] for i in batch_indices])
            qa_tar = np.array([qa_target_data[i, :, :] for i in batch_indices])
            img_inp = np.array([image_input_data[i, :] for i in batch_indices])

            # logs = autoencoder.train_on_batch([enc_inp, dec_inp], dec_tar)
            #         write_log(callback, 'train_loss', logs, e * 27 + (b + 1))
            log2 = question_answer.train_on_batch([enc_inp, img_inp, qa_inp], qa_tar)
        #         write_log(callback, 'qa_train_loss', log2, e * 27 + (b + 1))

        #     logs = autoencoder.test_on_batch([encoder_input_data[5377:5600, :, :],
        #                                           decoder_input_data[5377:5600, :, :]], decoder_target_data[5377:5600, :, :])

        #     logs2 = question_answer.test_on_batch([encoder_input_data[5377:5600, :, :],image_input_data[5377:5600, :],
        #                                     qa_input_data[5377:5600, :, :]], qa_target_data[5377:5600, :, :])
        # #     write_log(callback, 'val_loss', logs, e)
        #     write_log(callback, 'qa_val_loss', logs2, e)

        if e % 9 == 0:
            # autoencoder.save('autoencoder_2.h5')
            question_answer.save('question_answering_only.h5')