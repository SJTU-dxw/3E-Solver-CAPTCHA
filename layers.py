import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import math

USE_CUDA = torch.cuda.is_available()


class CNN(nn.Module):
    """
    input: [batch_size, 3, 64, 128]
    output: [batch_size, 32, 256]
    """

    def __init__(self):
        super(CNN, self).__init__()
        
        self.resblk_1 = ResBlk(3, 32)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2)
        self.dropout_1 = nn.Dropout(0.1)
        
        self.resblk_2 = ResBlk(32, 64)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2)
        self.dropout_2 = nn.Dropout(0.1)
        
        self.resblk_3 = ResBlk(64, 128)
        self.maxpool_3 = nn.MaxPool2d(kernel_size=(2, 1))
        self.dropout_3 = nn.Dropout(0.1)
        
        self.resblk_4 = ResBlk(128, 256)
        self.maxpool_4 = nn.MaxPool2d(kernel_size=(2, 1))
        self.dropout_4 = nn.Dropout(0.1)
        
        self.resblk_5 = ResBlk(256, 256)
        self.maxpool_5 = nn.MaxPool2d(kernel_size=(4, 1))
        self.dropout_5 = nn.Dropout(0.1)

    def forward(self, x):
        out = x
        
        out = self.resblk_1(out)
        out = self.maxpool_1(out)
        out = self.dropout_1(out)
        
        out = self.resblk_2(out)
        out = self.maxpool_2(out)
        out = self.dropout_2(out)
        
        out = self.resblk_3(out)
        out = self.maxpool_3(out)
        out = self.dropout_3(out)
        
        out = self.resblk_4(out)
        out = self.maxpool_4(out)
        out = self.dropout_4(out)
        
        out = self.resblk_5(out)
        out = self.maxpool_5(out)
        out = self.dropout_5(out)
        
        out = out.squeeze(2)
        out = out.transpose(1, 2)

        return out


class Encoder(nn.Module):
    """
    input: [batch_size, 32, 256]
    output: [batch_size, 32, 128]
    """

    def __init__(self, num_rnn_layers=2, rnn_hidden_size=128, dropout=0.5):
        super(Encoder, self).__init__()
        self.num_rnn_layers = num_rnn_layers
        self.rnn_hidden_size = rnn_hidden_size

        self.gru = nn.GRU(256, rnn_hidden_size, num_rnn_layers,
                          batch_first=True,
                          dropout=dropout)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = Variable(torch.zeros(self.num_rnn_layers, batch_size, self.rnn_hidden_size))
        if USE_CUDA:
            h0 = h0.cuda()
        out, hidden = self.gru(x, h0)

        return out


class HybirdDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size=128, num_rnn_layers=2, dropout=0.5):
        super(HybirdDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_rnn_layers = num_rnn_layers

        self.attn = DotProductAttentionLayer()
        self.gru = nn.GRU(hidden_size, hidden_size,
                          num_rnn_layers, batch_first=True,
                          dropout=dropout)

        self.wc = nn.Linear(2 * hidden_size, hidden_size)

        self.tanh = nn.Tanh()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def forward_train(self, encoder_outputs, max_len, y):
        batch_size = encoder_outputs.size(0)
        last_hidden = Variable(torch.zeros(self.num_rnn_layers, batch_size, self.hidden_size))
        if USE_CUDA:
            last_hidden = last_hidden.cuda()

        input = y[:, :max_len - 1]  # [batch, max_len-1]
        embed_input = self.embedding(input)  # [batch, max_len-1, 128]
        query, _ = self.gru(embed_input, last_hidden)  # [batch, max_len-1, 128]
        key = encoder_outputs  # [batch, 32, 128]
        value = encoder_outputs  # [batch, 32, 128]

        weighted_context = self.attn(query, key, value)  # [batch, max_len-1, 128]
        output = self.tanh(self.wc(torch.cat((query, weighted_context), 2)))  # [batch, max_len-1, 128]

        return output

    def forward_step(self, input, last_hidden, encoder_outputs):
        embed_input = self.embedding(input)
        output, hidden = self.gru(embed_input.unsqueeze(1), last_hidden)
        output = output.squeeze(1)

        query = output.unsqueeze(1)  # [batch, 1, 128]
        key = encoder_outputs  # [batch, 32, 128]
        value = encoder_outputs  # [batch, 32, 128]

        weighted_context = self.attn(query, key, value).squeeze(1)
        output = self.tanh(self.wc(torch.cat((output, weighted_context), 1)))
        return output, hidden

class ResBlk(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ResBlk, self).__init__()
        
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(ch_out)
        
        self.ch_out = ch_out
        self.ch_in = ch_in

        if ch_out != ch_in:
            self.extra_conv = nn.Conv2d(ch_in, ch_out, kernel_size=(1, 1), stride=(1, 1))
            self.extra_bn = nn.BatchNorm2d(ch_out)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.ch_out != self.ch_in:
            x = self.extra_conv(x)
            x = self.extra_bn(x)
        out = x + out

        return out


class DotProductAttentionLayer(nn.Module):
    def __init__(self):
        super(DotProductAttentionLayer, self).__init__()

    def forward(self, query, key, value):
        logits = torch.matmul(query, key.permute(0, 2, 1))  # [len, 256]*[256, 32]=[len, 32]

        alpha = F.softmax(logits, dim=-1)
        weighted_context = torch.matmul(alpha, value)  # [len, 32] * [32, 256]=[len, 256]

        return weighted_context
