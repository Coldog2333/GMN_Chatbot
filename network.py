import torch
import torch_geometric as gtorch
from torch_geometric.data import Data
from transformers import BertTokenizer

UNCASED = '/Users/jiangjunfeng/mainland/private/GMN_Chatbot/model/chinese_L-12_H-768_A-12'
VOCAB_SIZE = 21128

class GMN(torch.nn.Module):
    def __init__(self, emdedding_dim=128):
        super(GMN, self).__init__()
        self.word_embedding = torch.nn.Embedding(num_embeddings=VOCAB_SIZE, embedding_dim=emdedding_dim)
        self.conv = gtorch.nn.GCNConv(in_channels=128, out_channels=10)
        self.cross_w = torch.nn.Parameter(torch.rand(size=[128, 10]))

    def self_forward(self, utterance_input, response_input, utterance_graph_adj, response_graph_adj):
        utterance_input = self.word_embedding(utterance_input)
        response_input = self.word_embedding(response_input)

        utterance_feature = self.conv(utterance_input, utterance_graph_adj)
        response_feature = self.conv(response_input, response_graph_adj)

        return utterance_feature, response_feature

    def cross_forward(self, utterance_input, response_input, utterance_graph_adj, response_graph_adj):
        utterance_input = self.word_embedding(utterance_input)
        response_input = self.word_embedding(response_input)

        utterance_feature = torch.matmul(self.cross_w, response_input)
        response_feature = torch.matmul(self.cross_w, utterance_input)

        return utterance_feature, response_feature

    def forward(self, utterance_input, response_input, utterance_graph_adj, response_graph_adj):
        # assume the inputs are:
        # utterance input: [batch size, (max)_num_of_word, input_dim]
        # response input: [batch size, (max)_num_of_word, input_dim]
        # utterance graph adj: [batch size, 2, num_of_edge]
        input = torch.cat([utterance_input, response_input], dim=1)
        graph_adj = torch.cat([utterance_graph_adj, response_graph_adj])


class BERT_Representer(torch.nn.Module):
    def __init__(self):
        super(BERT_Representer, self).__init__()

    def forward(self, input_ids):
        pass


if __name__ == '__main__':
    context = ['医生你好，我觉得我最近头有点疼。', '是吗？在哪个位置？', '在这个位置']
    response = ['哦，这种情况很有可能是偏头痛。']

    context_input_ids = []
    response_input_ids = []
    context_graph_adjs = []
    response_graph_adjs = []

    tokenizer = BertTokenizer.from_pretrained(UNCASED)
    for sentence in context:
        context_input_ids.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize('[CLS]' + sentence + '[SEP]')))
    for sentence in response:
        response_input_ids.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize('[CLS]' + sentence + '[SEP]')))

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

    print(context_graph_adjs)

    conv = gtorch.nn.GCNConv(in_channels=1,
                             out_channels=2)

    # 边，shape = [2,num_edge]
    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)
    # 点，shape = [num_nodes, num_node_features]
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)

    print(conv(data.x, data.edge_index))