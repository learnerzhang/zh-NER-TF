import sys, pickle, os, random
import numpy as np

## tags, BIO
# tag2label = {"O": 0,
#              "B-PER": 1, "I-PER": 2,
#              "B-LOC": 3, "I-LOC": 4,
#              "B-ORG": 5, "I-ORG": 6
#              }

tag2label = {'O': 0, 'B-WLXS': 1, 'I-WLXS': 2, 'B-QY': 3, 'I-QY': 4, 'B-NUM': 5, 'I-NUM': 6, 'B-DSZY': 7, 'I-DSZY': 8,
             'B-DD': 9, 'I-DD': 10, 'B-JG': 11, 'I-JG': 12, 'B-YY': 13, 'I-YY': 14, 'B-GQ': 15, 'I-GQ': 16, 'B-CS': 17,
             'I-CS': 18, 'B-RW': 19, 'I-RW': 20, 'B-SJ': 21, 'I-SJ': 22, 'B-CBS': 23, 'I-CBS': 24, 'B-XZQ': 25,
             'I-XZQ': 26, 'B-TEXT': 27, 'I-TEXT': 28, 'B-YSZP': 29, 'I-YSZP': 30, 'B-YYZJ': 31, 'I-YYZJ': 32,
             'B-XX': 33, 'I-XX': 34, 'B-DATE': 35, 'I-DATE': 36, 'B-TSZP': 37, 'I-TSZP': 38, 'B-SW': 39, 'I-SW': 40,
             'B-XKZY': 41, 'I-XKZY': 42, 'B-JD': 43, 'I-JD': 44, 'B-WZ': 45, 'I-WZ': 46, 'B-M': 47, 'I-M': 48,
             'B-QH': 49, 'I-QH': 50, 'B-ZP': 51, 'I-ZP': 52, 'B-LSRW': 53, 'I-LSRW': 54, 'B-GJ': 55, 'I-GJ': 56}


def read_corpus(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        if line != '\n':
            [char, label] = line.strip().split("\t")
            sent_.append(char)
            tag_.append(label)
        else:
            data.append((sent_, tag_))
            sent_, tag_ = [], []

    return data


def vocab_build(vocab_path, corpus_path, min_count):
    """

    :param vocab_path:
    :param corpus_path:
    :param min_count:
    :return:
    """
    data = read_corpus(corpus_path)
    word2id = {}
    for sent_, tag_ in data:
        for word in sent_:
            if word.isdigit():
                word = '<NUM>'
            elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
                word = '<ENG>'
            if word not in word2id:
                word2id[word] = [len(word2id) + 1, 1]
            else:
                word2id[word][1] += 1
    low_freq_words = []
    for word, [word_id, word_freq] in word2id.items():
        if word_freq < min_count and word != '<NUM>' and word != '<ENG>':
            low_freq_words.append(word)
    for word in low_freq_words:
        del word2id[word]

    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0

    print(len(word2id))
    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw)


def sentence2id(sent, word2id):
    """

    :param sent:
    :param word2id:
    :return:
    """
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            word = '<ENG>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id


def read_dictionary(vocab_path):
    """

    :param vocab_path:
    :return:
    """
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id


def random_embedding(vocab, embedding_dim):
    """

    :param vocab:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def pad_sequences(sequences, pad_mark=0):
    """

    :param sequences:
    :param pad_mark:
    :return:
    """
    max_len = max(map(lambda x: len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


def batch_yield(data, batch_size, vocab, tag2label, shuffle=False):
    """

    :param data:
    :param batch_size:
    :param vocab:
    :param tag2label:
    :param shuffle:
    :return:
    """
    if shuffle:
        random.shuffle(data)

    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab)
        label_ = [tag2label[tag] for tag in tag_]

        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, labels


if __name__ == '__main__':
    vocab_build(vocab_path="./lic2019/word2id.pkl", corpus_path="./lic2019/train_data", min_count=3)
