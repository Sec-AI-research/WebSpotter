import torch
from core.preprocess import build_vocb, convert_sent_to_id
from classification.utils import TextDataset
from classification.train import _train_single as pytorch_train_single
from classification.test import predict as pytorch_predict
from model.cnn.textcnn import TextCNN
import numpy as np


def _get_text_traindataloader(string_list, label_list):
    train_data = [list(s) for s in string_list]

    word2id, id2word = build_vocb(train_data)
    train_data = convert_sent_to_id(train_data, word2id, max_len=100)
    train_data = torch.tensor(train_data)
    train_labels = torch.tensor(label_list)

    trainset = TextDataset(train_data, train_labels)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    return trainset, trainloader, word2id, id2word

def _get_text_testdataloader(string_list, word2id):
    test_data = [list(s) for s in string_list]
    test_data = convert_sent_to_id(test_data, word2id, max_len=100)
    test_data = torch.tensor(test_data)
    fake_test_labels = torch.tensor([0 for _ in range(len(test_data))])
    testset = TextDataset(test_data, fake_test_labels)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)
    return testloader

class Textcnn_wrapper:
    def __init__(self, vocab_size, word2id, id2word, num_classes=2, dropout=0.0, emb_dim=256, lr=0.001, epochs=5, batch_size=64, device='cuda:0'):
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.emb_dim = emb_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = TextCNN(vocab_size=self.vocab_size, emb_dim=self.emb_dim, dropout=self.dropout, n_class=self.num_classes)
        self.word2id = word2id
        self.id2word = id2word
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.device = device
        self.model.to(self.device)
        self.cache = {}
    
    def fit(self, trainloader):
        self.model.train()
        for epoch in range(self.epochs):
            pytorch_train_single(self.model, epoch, trainloader, self.optimizer, self.loss_fn, self.device)
    
    def predict(self, string_list):
        uncached_strings = []
        for string in string_list:
            if string not in self.cache:
                uncached_strings.append(string)

        # predict uncache data
        if uncached_strings:
            testloader = _get_text_testdataloader(uncached_strings, self.word2id)
            self.model.eval()
            predictions = pytorch_predict(self.model, testloader, self.device)

            # update cache
            for string, prediction in zip(uncached_strings, predictions):
                self.cache[string] = prediction
        
        # Ensure the results are in the same order as the input
        result = [self.cache[string] for string in string_list]
        result = np.array(result)
        return result

