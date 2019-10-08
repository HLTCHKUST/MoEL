import torch
import torch.utils.data as data
import random
import math
import os
import logging 
from utils import config
import pickle
from tqdm import tqdm
import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=1)
import re
import time
import nltk
nltk.download('punkt')


class Lang:
    def __init__(self, init_index2word):
        self.word2index = {str(v): int(k) for k, v in init_index2word.items()}
        self.word2count = {str(v): 1 for k, v in init_index2word.items()}
        self.index2word = init_index2word 
        self.n_words = len(init_index2word)  # Count default tokens
      
    def index_words(self, sentence):
        for word in sentence:
            self.index_word(word.strip())

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1



def clean(sentence, word_pairs):
    sentence = sentence.lower()
    for k, v in word_pairs.items():
        sentence = sentence.replace(k,v)
    sentence = nltk.word_tokenize(sentence)
    return sentence
def read_langs(vocab):
    word_pairs = {"it's":"it is", "don't":"do not", "doesn't":"does not", "didn't":"did not", "you'd":"you would", "you're":"you are", "you'll":"you will", "i'm":"i am", "they're":"they are", "that's":"that is", "what's":"what is", "couldn't":"could not", "i've":"i have", "we've":"we have", "can't":"cannot", "i'd":"i would", "i'd":"i would", "aren't":"are not", "isn't":"is not", "wasn't":"was not", "weren't":"were not", "won't":"will not", "there's":"there is", "there're":"there are"}
    train_context = np.load('empathetic-dialogue/sys_dialog_texts.train.npy',allow_pickle=True)
    train_target = np.load('empathetic-dialogue/sys_target_texts.train.npy',allow_pickle=True)
    train_emotion = np.load('empathetic-dialogue/sys_emotion_texts.train.npy',allow_pickle=True)
    train_situation = np.load('empathetic-dialogue/sys_situation_texts.train.npy',allow_pickle=True)

    dev_context = np.load('empathetic-dialogue/sys_dialog_texts.dev.npy',allow_pickle=True)
    dev_target = np.load('empathetic-dialogue/sys_target_texts.dev.npy',allow_pickle=True)
    dev_emotion = np.load('empathetic-dialogue/sys_emotion_texts.dev.npy',allow_pickle=True)
    dev_situation = np.load('empathetic-dialogue/sys_situation_texts.dev.npy',allow_pickle=True)
    
    test_context = np.load('empathetic-dialogue/sys_dialog_texts.test.npy',allow_pickle=True)
    test_target = np.load('empathetic-dialogue/sys_target_texts.test.npy',allow_pickle=True)
    test_emotion = np.load('empathetic-dialogue/sys_emotion_texts.test.npy',allow_pickle=True)
    test_situation = np.load('empathetic-dialogue/sys_situation_texts.test.npy',allow_pickle=True)

    data_train = {'context':[],'target':[],'emotion':[], 'situation':[]}
    data_dev = {'context':[],'target':[],'emotion':[], 'situation':[]}
    data_test = {'context':[],'target':[],'emotion':[], 'situation':[]}

    for context in train_context:
        u_list = []
        for u in context:
            u = clean(u, word_pairs)
            u_list.append(u)
            vocab.index_words(u)
        data_train['context'].append(u_list)
    for target in train_target:
        target = clean(target, word_pairs)
        data_train['target'].append(target)
        vocab.index_words(target)
    for situation in train_situation:
        situation = clean(situation, word_pairs)
        data_train['situation'].append(situation)
        vocab.index_words(situation)
    for emotion in train_emotion:
        data_train['emotion'].append(emotion)
    assert len(data_train['context']) == len(data_train['target']) == len(data_train['emotion']) == len(data_train['situation'])

    for context in dev_context:
        u_list = []
        for u in context:
            u = clean(u, word_pairs)
            u_list.append(u)
            vocab.index_words(u)
        data_dev['context'].append(u_list)
    for target in dev_target:
        target = clean(target, word_pairs)
        data_dev['target'].append(target)
        vocab.index_words(target)
    for situation in dev_situation:
        situation = clean(situation, word_pairs)
        data_dev['situation'].append(situation)
        vocab.index_words(situation)
    for emotion in dev_emotion:
        data_dev['emotion'].append(emotion)
    assert len(data_dev['context']) == len(data_dev['target']) == len(data_dev['emotion']) == len(data_dev['situation'])

    for context in test_context:
        u_list = []
        for u in context:
            u = clean(u, word_pairs)
            u_list.append(u)
            vocab.index_words(u)
        data_test['context'].append(u_list)
    for target in test_target:
        target = clean(target, word_pairs)
        data_test['target'].append(target)
        vocab.index_words(target)
    for situation in test_situation:
        situation = clean(situation, word_pairs)
        data_test['situation'].append(situation)
        vocab.index_words(situation)
    for emotion in test_emotion:
        data_test['emotion'].append(emotion)
    assert len(data_test['context']) == len(data_test['target']) == len(data_test['emotion']) == len(data_test['situation'])
    return data_train, data_dev, data_test, vocab


def load_dataset():
    if(os.path.exists('empathetic-dialogue/dataset_preproc.p')):
        print("LOADING empathetic_dialogue")
        with open('empathetic-dialogue/dataset_preproc.p', "rb") as f:
            [data_tra, data_val, data_tst, vocab] = pickle.load(f)
    else:
        print("Building dataset...")
        data_tra, data_val, data_tst, vocab  = read_langs(vocab=Lang({config.UNK_idx: "UNK", config.PAD_idx: "PAD", config.EOS_idx: "EOS", config.SOS_idx: "SOS", config.USR_idx:"USR", config.SYS_idx:"SYS", config.CLS_idx:"CLS"})) 
        with open('empathetic-dialogue/dataset_preproc.p', "wb") as f:
            pickle.dump([data_tra, data_val, data_tst, vocab], f)
            print("Saved PICKLE")
    for i in range(3):
        print('[situation]:', ' '.join(data_tra['situation'][i]))
        print('[emotion]:', data_tra['emotion'][i])
        print('[context]:', [' '.join(u) for u in data_tra['context'][i]])
        print('[target]:', ' '.join(data_tra['target'][i]))
        print(" ")
    return data_tra, data_val, data_tst, vocab

