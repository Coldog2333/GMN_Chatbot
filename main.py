import torch
import torch.nn
import numpy as np
import argparse

from network import GMN
from data_loader import load_data_from_file

EPOCH = 10
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
DEVICE = 'cuda'


def train(args):
    eval_contexts, eval_repsonses, eval_contexts_graph_adjs, eval_response_graph_adjs, eval_labels = load_data_from_file(args.dir + 'data/2020_split_by_idname.csv', tokenize=True)

    contexts, repsonses, contexts_graph_adjs, response_graph_adjs, labels = load_data_from_file(args.dir + 'data/from_2011_to_2019_split_by_idname.csv', tokenize=True)

    # move tensors to the specified devices.
    if DEVICE == 'cuda':
        for tensor in [eval_contexts, eval_repsonses, eval_contexts_graph_adjs, eval_response_graph_adjs, eval_labels,
                       contexts, repsonses, contexts_graph_adjs, response_graph_adjs, labels]:
            tensor.cuda()
    else:
        for tensor in [eval_contexts, eval_repsonses, eval_contexts_graph_adjs, eval_response_graph_adjs, eval_labels,
                       contexts, repsonses, contexts_graph_adjs, response_graph_adjs, labels]:
            tensor.cpu()

    net = GMN(emdedding_dim=768, use_bert=True).to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    loss_function = torch.nn.BCELoss()

    print('start training...')
    print('dataset size: %s' % labels.shape)
    for epoch in range(EPOCH):
        net.train()
        losses = []
        for k in range(len(labels) // BATCH_SIZE):
            output_p = net(contexts[k * BATCH_SIZE:(k+1) * BATCH_SIZE],
                           repsonses[k * BATCH_SIZE:(k+1) * BATCH_SIZE],
                           contexts_graph_adjs[k * BATCH_SIZE:(k+1) * BATCH_SIZE],
                           response_graph_adjs[k * BATCH_SIZE:(k+1) * BATCH_SIZE])

            loss = loss_function(output_p, labels[k * BATCH_SIZE:(k+1) * BATCH_SIZE])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
        print('loss: %.2f' % np.mean(losses))
        evaluate(net, eval_contexts, eval_repsonses, eval_contexts_graph_adjs, eval_response_graph_adjs, eval_labels)


def evaluate(net, contexts, repsonses, contexts_graph_adjs, response_graph_adjs, labels):
    net.eval()
    acc = 0.
    for k in range(len(labels) // BATCH_SIZE):
        output_p = net(contexts[k * BATCH_SIZE:(k + 1) * BATCH_SIZE],
                       repsonses[k * BATCH_SIZE:(k + 1) * BATCH_SIZE],
                       contexts_graph_adjs[k * BATCH_SIZE:(k + 1) * BATCH_SIZE],
                       response_graph_adjs[k * BATCH_SIZE:(k + 1) * BATCH_SIZE])

        ground_truth = labels[k * BATCH_SIZE:(k + 1) * BATCH_SIZE]
        for i in range(BATCH_SIZE):
            if (output_p[i] >= 0.5 and ground_truth[i] == 1) or (output_p[i] < 0.5 and ground_truth[i] == 0):
                acc += 1

    print('acc: %.2f' % (acc / len(labels)))


def debug(args):
    contexts, repsonses, contexts_graph_adjs, response_graph_adjs, labels = load_data_from_file(args.dir + 'data/2011_split_by_idname.csv', tokenize=True, read_case_num=10)

    net = GMN(emdedding_dim=768, use_bert=True).to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    loss_function = torch.nn.BCELoss()

    print('start training...')
    print('dataset size: %s' % labels.shape)
    for epoch in range(EPOCH):
        net.train()
        losses = []
        for k in range(len(labels) // BATCH_SIZE):
            output_p = net(contexts[k * BATCH_SIZE:(k+1) * BATCH_SIZE],
                           repsonses[k * BATCH_SIZE:(k+1) * BATCH_SIZE],
                           contexts_graph_adjs[k * BATCH_SIZE:(k+1) * BATCH_SIZE],
                           response_graph_adjs[k * BATCH_SIZE:(k+1) * BATCH_SIZE])

            loss = loss_function(output_p, labels[k * BATCH_SIZE:(k+1) * BATCH_SIZE])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
        print('loss: %.2f' % np.mean(losses))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='/Users/jiangjunfeng/mainland/private/GMN_Chatbot/')
    args = parser.parse_args()

    train(args)
