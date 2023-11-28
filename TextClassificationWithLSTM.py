# Text Classification with LSTM/RNN
# Dataset: https://www.kaggle.com/datasets/fstcap/weibo-senti-100k
#
import torch
import torchtext.data.functional as textF

import torch.nn.functional as F
import numpy as np
import pandas as pd
import jieba
# pip install scikit-learn
from sklearn.model_selection import train_test_split

class TextProcessor():
    def __init__(self, min_occurrences=2, padding_len=50):
        self.min_occurrences = min_occurrences
        self.padding_len = padding_len
        self.words_cnt = 0
        self.vocab = None

    @classmethod
    def tokenize(self, str):
        txt = str.replace('！','').replace('，','').replace('。','').replace('@','').replace('/','')
        # split and return list
        return jieba.lcut(txt)

    def build_vocab(self, csv):
        word_cnt = pd.value_counts(np.concatenate(csv.review.values))
        # Delete the words which has few occurrences
        word_cnt = word_cnt[word_cnt > self.min_occurrences]
        # Encode word into occurrence
        word_list = list(word_cnt.index)
        self.word_index = dict((word, word_list.index(word) + 1) for word in word_list)
        vocab = csv.review.apply(lambda t : [self.word_index.get(word, 0) for word in t])

        return (len(self.word_index) + 1, vocab)

    def load_file(self, csv_path):
        csv = pd.read_csv(csv_path)
        # Tokenize the review column
        csv['review'] = csv.review.apply(TextProcessor.tokenize)
        # Map word into number
        self.words_cnt, self.vocab = self.build_vocab(csv)
        # Padding
        padding_text =  [v + (self.padding_len - len(v)) * [0] if len(v)<=self.padding_len else v[:self.padding_len] for v in self.vocab]
        padding_text = np.array(padding_text)

        labels = csv.label.values

        return (padding_text, labels)
 

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, vocabs, labels):
        self.vocabs = vocabs
        self.labels= labels
    
    def __getitem__(self,index):
        vocab = torch.LongTensor(self.vocabs[index])
        label= self.labels[index]
        return (vocab, label)
        
    def __len__(self):
        return len(self.vocabs)
    

class CommentClassification(torch.nn.Module):
    def __init__(self, num_classes, words_cnt, embedding_dim=128, hidden_size=256, rnn_layers=3, bidirectional=True) :
        super(CommentClassification, self).__init__()
        # Encode word into embedding_dim vector
        self.embedding = torch.nn.Embedding(words_cnt, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim,
            hidden_size,
            num_layers=rnn_layers,
            dropout=0.5, 
            bidirectional=bidirectional
        )

        rnn_out = hidden_size
        if bidirectional:
            # Hidden layer double for bidirectional
            rnn_out = hidden_size * 2

        self.fc1 = torch.nn.Linear(rnn_out, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.output = torch.nn.Linear(64, num_classes)

    def forward(self, x):
        y = self.embedding(x)
        r_o, _ = self.lstm(y)
        r_o = r_o[-1]
        y = F.dropout(F.relu(self.fc1(r_o)))
        y = F.dropout(F.relu(self.fc2(y)))
        y = self.output(y)
        return y


def build_dataloader(path, batch_size=32, padding_len=50):
    txt_processor = TextProcessor(2, padding_len)
    padding_text, labels = txt_processor.load_file(path)
    x_train, x_test, y_train, y_test = train_test_split(padding_text, labels)

    train_ds = TextDataset(x_train, y_train)
    test_ds = TextDataset(x_test, y_test)

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return (txt_processor, train_dl, test_dl)

def train(model, device, dataloader):
    model.train()

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epoch_loss = 0
    epoch_correct = 0
    
    for x, y in dataloader:
        x = x.permute(1, 0)
        x = x.to(device)
        y = y.to(device)

        predicted = model(x)
        loss = loss_func(predicted, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            epoch_correct += (predicted.argmax(1) == y).type(torch.float).sum().item()
            epoch_loss += loss.item()

    return (epoch_loss, epoch_correct)


def test(model, device, dataloader):
    model.eval()
    loss_func = torch.nn.CrossEntropyLoss()
    epoch_loss = 0
    epoch_correct = 0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.permute(1, 0)
            x = x.to(device)
            y = y.to(device)
            
            predicted = model(x)
            loss = loss_func(predicted, y)

            epoch_correct += (predicted.argmax(1) == y).type(torch.float).sum().item()
            epoch_loss += loss.item()

    return (epoch_loss, epoch_correct)

def fit(epoch=20):
    padding_len = 50
    txt_processor, train_dl, test_dl = build_dataloader('./Data/weibo_senti_100k.csv', padding_len=padding_len)
    model = CommentClassification(2, txt_processor.words_cnt)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    total_train_data_cnt = len(train_dl.dataset)
    num_train_batch = len(train_dl)
    total_test_data_cnt = len(test_dl.dataset)
    num_test_batch = len(test_dl)

    best_accuracy = 0.0

    for i in range(epoch):
        epoch_train_loss, epoch_train_correct = train(model, device, train_dl)
        avg_train_loss = epoch_train_loss/num_train_batch
        avg_train_accuracy = epoch_train_correct/total_train_data_cnt

        epoch_test_loss, epoch_test_correct = test(model, device, test_dl)
        avg_test_loss = epoch_test_loss/num_test_batch
        avg_test_accuracy = epoch_test_correct/total_test_data_cnt

        msg_template = ("Epoch {:2d} - Train accuracy: {:.2f}%, Train loss: {:.6f}; Test accuracy: {:.2f}%, Test loss: {:.6f}")
        print(msg_template.format(i+1, avg_train_accuracy*100, avg_train_loss, avg_test_accuracy*100, avg_test_loss))

        if avg_test_accuracy > best_accuracy:
            best_accuracy = avg_test_accuracy
            torch.save(model.state_dict(), 'lstm_comments.model')


if __name__ == '__main__':
    fit(5)


