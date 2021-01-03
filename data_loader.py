import numpy as np
from copy import deepcopy
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import random

UNCASED = '/Users/jiangjunfeng/mainland/private/GMN_Chatbot/model/chinese_L-12_H-768_A-12'
VOCAB_SIZE = 21128
MAX_SEQ_LEN = 50
MAX_TURN_NUM = 5
NEG_NUM = 1

tokenizer = BertTokenizer.from_pretrained(UNCASED)

def tokenize_and_pad(sentence, tokenizer):
    ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('[CLS]' + sentence + '[SEP]'))
    if len(ids) < MAX_SEQ_LEN:
        ids += [0] * (MAX_SEQ_LEN - len(ids))
    else:
        ids = ids[:MAX_SEQ_LEN]
    return ids


def load_data_from_file(filename, tokenize=False, lang='cn', read_case_num=1000000):
    with open(filename) as f:
        header = f.readline()
        lines = f.readlines()

    contexts, responses, labels = [], [], []
    marker = 1 if lang == 'cn' else 0

    print('loading...')
    cache_utterances = []
    for line in lines:
        items = line.strip().split(',')
        id, speaker, context = items[0], items[1], ','.join(items[2:])
        if id == marker:
            cache_utterances.append(context)
        else:
            marker = id
            if len(cache_utterances) > 1:
                contexts.append(cache_utterances[:-1])
                responses.append(cache_utterances[-1])
                labels.append(1.)
            cache_utterances = [context]
            read_case_num -= 1
            if read_case_num < 0:
                break

    # make negative samples
    print('making negative samples...')
    for k in range(len(contexts)):
        context = deepcopy(contexts[k])
        sampled_indices = np.random.randint(0, len(responses) - 1, size=[NEG_NUM])
        for i in range(NEG_NUM):
            contexts.append(context)
            responses.append(responses[sampled_indices[i]])
            labels.append(0.)

    shuffle_ids = list(range(len(labels)))
    random.shuffle(shuffle_ids)
    print(shuffle_ids)

    if tokenize:
        print('tokenizing...')
        for i in range(len(contexts)):
            for j in range(len(contexts[i])):
                contexts[i][j] = tokenize_and_pad(contexts[i][j], tokenizer)
            if len(contexts[i]) < MAX_TURN_NUM:
                contexts[i] += [torch.zeros(size=[50])] * (MAX_TURN_NUM - len(contexts[i]))
            else:
                contexts[i] = contexts[i][:MAX_TURN_NUM]

        for i in range(len(responses)):
            responses[i] = tokenize_and_pad(responses[i], tokenizer)

        contexts = torch.LongTensor(contexts)[shuffle_ids, :]
        responses = torch.LongTensor(responses)[shuffle_ids, :]

    contexts_graph_adjs, response_graph_adjs = [], []

    for context in contexts:
        context_graph_adjs = []
        for input_ids in context:
            graph_adj = [[], []]
            for i in range(len(input_ids) - 1):
                graph_adj[0].append(i)
                graph_adj[1].append(i + 1)
            context_graph_adjs.append(graph_adj)
        contexts_graph_adjs.append(context_graph_adjs)

    for input_ids in responses:
        graph_adj = [[], []]
        for i in range(len(input_ids) - 1):
            graph_adj[0].append(i)
            graph_adj[1].append(i + 1)
        response_graph_adjs.append(graph_adj)

    contexts_graph_adjs = torch.LongTensor(contexts_graph_adjs)[shuffle_ids, :, :]
    response_graph_adjs = torch.LongTensor(response_graph_adjs)[shuffle_ids, :, :]
    labels = torch.tensor(labels)[shuffle_ids]

    return contexts, responses, contexts_graph_adjs, response_graph_adjs, labels


def prepare_data(context, response):
    context_input_ids = []
    response_input_ids = []
    context_graph_adjs = []
    response_graph_adjs = []

    tokenizer = BertTokenizer.from_pretrained(UNCASED)
    for sentence in context:
        ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('[CLS]' + sentence + '[SEP]'))
        if len(ids) < MAX_SEQ_LEN:
            ids += [0] * (MAX_SEQ_LEN - len(ids))
        else:
            ids = ids[:MAX_SEQ_LEN]
        context_input_ids.append(ids)
    for sentence in response:
        ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('[CLS]' + sentence + '[SEP]'))
        if len(ids) < MAX_SEQ_LEN:
            ids += [0] * (MAX_SEQ_LEN - len(ids))
        else:
            ids = ids[:MAX_SEQ_LEN]
        response_input_ids.append(ids)

    for input_ids in context_input_ids:
        graph_adj = [[], []]
        for i in range(len(input_ids) - 1):
            graph_adj[0].append(i)
            graph_adj[1].append(i + 1)
        context_graph_adjs.append(graph_adj)

    for input_ids in response_input_ids:
        graph_adj = [[], []]
        for i in range(len(input_ids) - 1):
            graph_adj[0].append(i)
            graph_adj[1].append(i + 1)
        response_graph_adjs.append(graph_adj)

    return context_input_ids, response_input_ids, context_graph_adjs, response_graph_adjs


if __name__ == '__main__':
    context = ['医生你好，我觉得我最近头有点疼。', '是吗？在哪个位置？', '在这个位置']
    response = ['哦，这种情况很有可能是偏头痛。']

    # print(prepare_data(context, response))

    contexts, responses = load_data_from_file('/Users/jiangjunfeng/mainland/private/GMN_Chatbot/data/2012_split_by_idname.csv')

    print(contexts[1])
    print(responses[1])
