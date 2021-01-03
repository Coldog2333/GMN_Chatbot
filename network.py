import torch
import torch.nn.functional as F
import torch_geometric as gtorch
from torch_geometric.data import Data
from transformers import BertTokenizer, BertModel

UNCASED = '/Users/jiangjunfeng/mainland/private/GMN_Chatbot/model/chinese_L-12_H-768_A-12'
VOCAB_SIZE = 21128
MAX_SEQ_LEN = 50
MAX_TURN_NUM = 5


class GMN(torch.nn.Module):
    def __init__(self, emdedding_dim=768, use_bert=False):
        super(GMN, self).__init__()
        self.word_embedding = torch.nn.Embedding(num_embeddings=VOCAB_SIZE, embedding_dim=emdedding_dim)

        self.bert_representer = BERT_Representer() if use_bert else None

        self.conv = gtorch.nn.GCNConv(in_channels=emdedding_dim, out_channels=10)
        self.cross_w = torch.nn.Parameter(torch.rand(size=[emdedding_dim, 10]))

        self.attention_coef = torch.nn.Parameter(torch.rand(size=[MAX_SEQ_LEN, MAX_SEQ_LEN]))
        self.assign_weight = torch.nn.Parameter(torch.rand(size=[MAX_SEQ_LEN, 10]))
        self.multi_perspective_FFN = torch.nn.Sequential(torch.nn.Linear(in_features=10 + 1, out_features=20),
                                                         torch.nn.ReLU(),
                                                         torch.nn.Linear(in_features=20, out_features=20))

        self.FFN = torch.nn.Sequential(torch.nn.Linear(in_features=4 * 20, out_features=40),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(in_features=40, out_features=1))

    def self_forward(self, utterance_input, response_input, utterance_graph_adj, response_graph_adj):
        utterance_input = self.bert_representer(utterance_input)
        response_input = self.bert_representer(response_input)

        utterance_feature = self.conv(utterance_input, utterance_graph_adj)
        response_feature = self.conv(response_input, response_graph_adj)

        return utterance_feature, response_feature

    def cross_forward(self, utterance_input, response_input):
        utterance_input = self.bert_representer(utterance_input)
        response_input = self.bert_representer(response_input)

        utterance_feature = torch.matmul(self.attention_coef, torch.matmul(response_input, self.cross_w))
        response_feature = torch.matmul(self.attention_coef, torch.matmul(utterance_input, self.cross_w))

        return utterance_feature, response_feature

    def multi_perspective_matching(self, self_feature, cross_feature):
        self_feature = self_feature.squeeze()
        cross_feature = cross_feature.squeeze()
        distances = torch.cosine_similarity(self.assign_weight * self_feature, self.assign_weight * cross_feature, dim=1)
        h = self.multi_perspective_FFN(torch.cat([distances.reshape(-1, 1), self_feature], dim=-1))
        return h

    def ut_forward(self, utterance_input, response_input, utterance_graph_adj, response_graph_adj):
        # assume the inputs are:
        # utterance input: [batch size, (max)_num_of_word, input_dim]
        # response input: [batch size, (max)_num_of_word, input_dim]
        # utterance graph adj: [batch size, 2, num_of_edge]

        utterance_input = utterance_input.reshape(1, -1)
        response_input = response_input.reshape(1, -1)

        utterance_self_feature, response_self_feature = self.self_forward(utterance_input, response_input, utterance_graph_adj, response_graph_adj)
        utterance_cross_feature, response_cross_feature = self.cross_forward(utterance_input, response_input)

        utterance_matching_feature = self.multi_perspective_matching(utterance_self_feature, utterance_cross_feature)
        response_matching_feature = self.multi_perspective_matching(response_self_feature, response_cross_feature)

        g_u = torch.max(utterance_matching_feature, dim=0)[0]
        g_r = torch.max(response_matching_feature, dim=0)[0]

        logits = self.FFN(torch.cat([g_u, g_r, g_u * g_r, torch.abs(g_u - g_r)], dim=-1))

        p = logits
        return p

    def forward(self, context_input, response_input, context_graph_adjs, response_graph_adj):
        batch_ps = []
        batch_size = context_input.shape[0]
        for i in range(batch_size):
            ps = torch.zeros(size=[1])
            for j in range(MAX_TURN_NUM):
                p = self.ut_forward(context_input[i][j],
                                    response_input[i],
                                    context_graph_adjs[i][j],
                                    response_graph_adj[i])
                ps += p
            batch_ps.append(torch.sigmoid(ps))

        return torch.stack(batch_ps, dim=0)


class BERT_Representer(torch.nn.Module):
    def __init__(self):
        super(BERT_Representer, self).__init__()
        self.bert = BertModel.from_pretrained(UNCASED)

    def forward(self, input_ids):
        input_features = self.bert(input_ids)[0]
        return input_features


if __name__ == '__main__':
    exit(0)
    context = ['医生你好，我觉得我最近头有点疼。', '是吗？在哪个位置？', '在这个位置']
    response = ['哦，这种情况很有可能是偏头痛。']

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

    gmn = GMN(emdedding_dim=128)
    # utterance_self_feature, response_self_feature = gmn.self_forward(utterance_input=torch.tensor(context_input_ids[0]),
    #                                                                  response_input=torch.tensor(response_input_ids[0]),
    #                                                                  utterance_graph_adj=torch.tensor(context_graph_adjs[0]),
    #                                                                  response_graph_adj=torch.tensor(response_graph_adjs[0]))
    #
    # print(utterance_self_feature.shape, response_self_feature.shape)
    #
    # utterance_cross_feature, response_cross_feature = gmn.cross_forward(utterance_input=torch.tensor(context_input_ids[0]),
    #                                                                     response_input=torch.tensor(response_input_ids[0]))
    #
    #
    # print(utterance_cross_feature.shape, response_cross_feature.shape)
    #
    # gmn.multi_perspective_matching(utterance_self_feature, utterance_cross_feature)

    logits = gmn(utterance_input=torch.tensor(context_input_ids[0]),
                response_input=torch.tensor(response_input_ids[0]),
                utterance_graph_adj=torch.tensor(context_graph_adjs[0]),
                response_graph_adj=torch.tensor(response_graph_adjs[0]))

    print(logits)
    print(logits.shape)