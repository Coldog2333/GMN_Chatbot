import torch
import torch.nn

from network import BERT_Representer, GMN
from data_loader import load_data_from_file

EPOCH = 10
LEARNING_RATE = 1e-4


if __name__ == '__main__':
    contexts, repsonses, contexts_graph_adjs, response_graph_adjs = load_data_from_file('/Users/jiangjunfeng/mainland/private/GMN_Chatbot/data/2011_split_by_idname.csv', tokenize=True, read_case_num=2)
    labels = torch.tensor([[1.], [1.]])

    # print(contexts.shape, repsonses.shape, contexts_graph_adjs.shape, response_graph_adjs.shape)

    # model = BERT_Representer()
    # print(model(repsonses)[0].shape)    # get representation of responses

    net = GMN(emdedding_dim=768, use_bert=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    loss_function = torch.nn.BCELoss()

    for epoch in range(EPOCH):
        net.train()
        output_p = net(contexts, repsonses, contexts_graph_adjs, response_graph_adjs)
        loss = loss_function(output_p, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(loss.item())
