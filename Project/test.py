import numpy as np
from keras.models import load_model
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Concatenate, add, multiply
import re

def decode_sequence(encoder_model, decoder_model, input_seq, max_decoder_seq_length, reverse_target_word_index, num_decoder_tokens):
    # prediction of final states of the encoder
    states_value = encoder_model.predict(input_seq)

    # target sequence initialization
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # first char id '\t'
    target_seq[0, 0, target_token_index['\t']] = 1.

    # we generate the answers until we get '\n', the ned of sequence character
    stop_condition = False
    decoded_sentence = []
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # we sample one word
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_target_word_index[sampled_token_index]
        decoded_sentence.append(sampled_word)

        # we do it until we get the end of sequence char or until we reach the max length of the targets
        if (sampled_word == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # update of the target seq
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # update of the states of the model
        states_value = [h, c]

    return decoded_sentence


def decode_sequence_VQA(encoder_model, image_model, qa_model, input_seq, image_features, max_qa_seq_length,
                        reverse_target_word_index, num_qa_tokens):
    # prediction of final states of the encoder
    states_value = encoder_model.predict(input_seq)
    #     print(np.array(states_value).shape)

    states_value_im = image_model.predict(np.reshape(image_features, (1, 100352)))

    merged_1 = np.concatenate((states_value_im, states_value[0]), axis=1)
    merged_2 = np.concatenate((states_value_im, states_value[1]), axis=1)
    #     merged_1 =np.multiply(states_value_im, states_value[0])
    #     merged_2 = np.multiply(states_value_im, states_value[1])
    #     print(np.array(merged_1).shape)

    #     print(a)
    states_value = [merged_1, merged_2]
    #     print(np.array(states_value).shape)
    # target sequence initialization
    target_seq = np.zeros((1, 1, num_qa_tokens))
    # first char id '\t'
    target_seq[0, 0, target_token_index['\t']] = 1.

    # we generate the answers until we get '\n', the ned of sequence character
    stop_condition = False
    decoded_sentence = []
    while not stop_condition:

        output_tokens, h, c = qa_model.predict(
            [target_seq] + states_value)

        # we sample one word
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_target_word_index[sampled_token_index]
        decoded_sentence.append(sampled_word)

        # we do it until we get the end of sequence char or until we reach the max length of the targets
        if (sampled_word == '\n' or
                len(decoded_sentence) > max_qa_seq_length):
            stop_condition = True

        # update of the target seq
        target_seq = np.zeros((1, 1, num_qa_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # update of the states of the model
        states_value = [h, c]

    return decoded_sentence

if __name__ == '__main__':
    import json

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

    del feat
    feat_sum = dict()
    for key, value in feat_sum2.items():
        # mi = min(value)
        # ma = max(value)
        feat_sum[key] = np.array([(x - mi) / (ma - mi) for x in value])

    del feat_sum2
    print(feat_sum['image1'])
    print(np.array(feat_sum['image1']).shape)

    latent_dim = 128

    # reading the data
    data_path = './qa.894.raw.train.txt'
    test_data_path = './qa.894.raw.test.txt'

    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with open(test_data_path, 'r', encoding='utf-8') as f2:
        test_lines = f2.readlines()
    input_texts = []
    target_texts = []
    qa_target_texts = []
    input_words = set()
    target_words = set()
    qa_target_words = set()
    test_input_texts = []
    test_qa_truth = []
    input_images = []
    test_input_images = []
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

    for i in range(0, len(test_lines), 2):
        test_input_text = test_lines[i].strip().split(' ')
        test_input_texts.append(test_input_text)
        test_input_images.append(test_input_text[len(test_input_text) - 2])
        test_qa_line = re.split(', ', test_lines[i + 1].strip())
        test_qa_truth.append(test_qa_line)

    input_words = sorted(list(input_words))
    target_words = sorted(list(target_words))
    qa_target_words = sorted(list(qa_target_words))
    num_encoder_tokens = len(input_words)
    num_decoder_tokens = len(target_words)
    num_qa_tokens = len(qa_target_words)
    input_token_index = dict(
        [(word, i) for i, word in enumerate(input_words)])
    target_token_index = dict(
        [(word, i) for i, word in enumerate(target_words)])
    qa_token_index = dict(
        [(word, i) for i, word in enumerate(qa_target_words)])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_qa_seq_length = max([len(txt) for txt in qa_target_texts])

    encoder_input_data = np.zeros(
        (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
        dtype='float32')
    image_input_data = np.zeros((len(input_texts), 100352), dtype='float32')
    test_image_input_data = np.zeros((len(test_input_texts), 100352), dtype='float32')
    test_encoder_input_data = np.zeros(
        (len(test_input_texts), max_encoder_seq_length, num_encoder_tokens),
        dtype='float32')

    for i, input_text in enumerate(input_texts):
        for t, word in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[word]] = 1.
            image_input_data[i, :] = feat_sum[input_images[i]]
    for i, test_input_text in enumerate(test_input_texts):
        for t, word in enumerate(test_input_text):
            try:
                test_image_input_data[i, :] = feat_sum[test_input_images[i]]
                test_encoder_input_data[i, t, input_token_index[word]] = 1.

            except:
                # if the input word is not in our dictionary, we sample randomly a word from our dictionary
                rnd = np.random.randint(0, num_encoder_tokens)
                test_encoder_input_data[i, t, input_token_index[input_words[rnd]]] = 1.

    reverse_input_word_index = dict(
        (i, word) for word, i in input_token_index.items())
    reverse_target_word_index = dict(
        (i, word) for word, i in target_token_index.items())
    reverse_qa_word_index = dict(
        (i, word) for word, i in qa_token_index.items())

    autoencoder = load_model('./autoencoder_2.h5')
    question_answer = load_model('./question_answering_only.h5')

    encoder_inputs = question_answer.input[0]  # input_1
    encoder_outputs, state_fh_enc, state_fc_enc, state_bh_enc, state_bc_enc = question_answer.layers[
        5].output  # bilstm_1
    state_h_enc = Concatenate()([state_fh_enc, state_bh_enc])
    state_c_enc = Concatenate()([state_fc_enc, state_bc_enc])
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(latent_dim * 2,))
    decoder_state_input_c = Input(shape=(latent_dim * 2,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder_outputs, state_h, state_c = autoencoder.layers[5](decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = autoencoder.layers[6](decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    image_input = question_answer.input[1]
    image_outputs = question_answer.layers[6].output
    image_model = Model(image_input, image_outputs)

    # print(question_answer.input)
    qa_input_h = Input(shape=(latent_dim * 4,))
    qa_input_c = Input(shape=(latent_dim * 4,))
    qa_states_inputs = [qa_input_h, qa_input_c]

    qa_inputs = Input(shape=(None, num_qa_tokens))
    qa_outputs, qa_state_h, qa_state_c = question_answer.layers[12](qa_inputs, initial_state=qa_states_inputs)
    qa_states = [qa_state_h, qa_state_c]
    qa_outputs = question_answer.layers[13](qa_outputs)
    qa_model = Model([qa_inputs] + qa_states_inputs, [qa_outputs] + qa_states)

    print("models loaded")

    # percentages = []
    # for seq_index in range(len(input_texts)):
    #     input_seq = encoder_input_data[seq_index: seq_index + 1]
    #     decoded_sentence = decode_sequence(encoder_model, decoder_model, input_seq, max_decoder_seq_length,
    #                                        reverse_target_word_index, num_decoder_tokens)
    #     original = input_texts[seq_index]
    #     correct = 0
    #     for i in range(len(original)):
    #         try:
    #             if original[i] == decoded_sentence[i]:
    #                 correct += 1
    #         except:
    #             pass
    #     percentages.append((correct - 1) / len(original) * 100)
    #     print('-', seq_index, ' - ', (correct - 1) / len(original) * 100, '%')
    #     print('Input sentence:', input_texts[seq_index])
    #     print('Decoded sentence:', decoded_sentence)
    # acc_percentage = sum(percentages) / len(percentages)
    # print('Final train accuracy: ', acc_percentage, '%')

    # percentages = []
    # autoencoder_decoded = []
    # for seq_index in range(len(test_input_texts)):
    #
    #     input_seq = test_encoder_input_data[seq_index: seq_index + 1]
    #     decoded_sentence = decode_sequence(encoder_model, decoder_model, input_seq, max_decoder_seq_length,
    #                                        reverse_target_word_index, num_decoder_tokens)
    #     autoencoder_decoded.append(decoded_sentence[:-1])
    #     original = test_input_texts[seq_index]
    #     correct = 0
    #
    #     for i in range(len(original)):
    #         try:
    #             if original[i] == decoded_sentence[i]:
    #                 correct += 1
    #         except:
    #             pass
    #
    #     percentages.append(correct / len(original) * 100)
    #     print('-', seq_index, ' - ', correct / len(original) * 100, '%')
    #     print('Input sentence:', test_input_texts[seq_index])
    #     print('Decoded sentence:', decoded_sentence)
    # acc_percentage = sum(percentages) / len(percentages)
    # print('Final test accuracy: ', acc_percentage, '%')

    print("QUESTION ANSWERING")
    percentages = []
    qa_train_prediction = []
    for seq_index in range(len(input_texts)):
        input_seq = encoder_input_data[seq_index: seq_index + 1]
        image_features = image_input_data[seq_index]
        decoded_sentence = decode_sequence_VQA(encoder_model, image_model, qa_model, input_seq, image_features,
                                               max_qa_seq_length, reverse_qa_word_index,
                                               num_qa_tokens)
        question = input_texts[seq_index]
        qa_train_prediction.append(decoded_sentence[:-1])
        print(question)
        print(decoded_sentence)

    with open('./qa_train_ground_truth_only.txt', 'w') as f:
        for sentence in qa_target_texts:
            f.write(','.join(str(line) for line in sentence[1:-1]))
            f.write('\n')
    f.close()
    with open('qa_train_prediction_only.txt', 'w') as f:
        for sentence in qa_train_prediction:
            f.write(','.join(str(line) for line in sentence))
            f.write('\n')
    f.close()

    percentages = []
    qa_decoded = []
    for seq_index in range(len(test_input_texts)):
        input_seq = test_encoder_input_data[seq_index: seq_index + 1]
        image_features = test_image_input_data[seq_index]
        decoded_sentence = decode_sequence_VQA(encoder_model, image_model, qa_model, input_seq, image_features,
                                               max_qa_seq_length, reverse_qa_word_index,
                                               num_qa_tokens)
        qa_decoded.append(decoded_sentence[:-1])
        question = test_input_texts[seq_index]
        print(question)
        print(decoded_sentence)
    with open('./qa_ground_truth_only.txt', 'w') as f:
        for sentence in test_qa_truth:
            f.write(','.join(str(line) for line in sentence))
            f.write('\n')
    f.close()
    with open('qa_prediction_only.txt', 'w') as f:
        for sentence in qa_decoded:
            f.write(','.join(str(line) for line in sentence))
            f.write('\n')
    f.close()