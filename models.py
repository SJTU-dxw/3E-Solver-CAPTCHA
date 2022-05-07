import torch.nn as nn
import torch
from torch.autograd import Variable
from layers import CNN, Encoder, HybirdDecoder

USE_CUDA = torch.cuda.is_available()


class CNNSeq2Seq(nn.Module):
    def __init__(self, vocab_size, max_len, hidden_size=128):
        super(CNNSeq2Seq, self).__init__()
        self.max_len = max_len

        self.backbone = CNN()
        self.encoder = Encoder()
        self.decoder = HybirdDecoder(vocab_size=vocab_size)
        self.prediction = nn.Linear(hidden_size, vocab_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0.0)

    def forward_train(self, x, y):        
        out = self.backbone(x)
        encoder_outputs = self.encoder(out)

        vocab_out = self.decoder.forward_train(encoder_outputs, self.max_len, y)
        vocab_out = self.prediction(vocab_out)
        return vocab_out

    def forward_test(self, x):         
        out = self.backbone(x)
        encoder_outputs = self.encoder(out)

        outputs = []
        batch_size = x.size(0)
        input = torch.zeros([batch_size]).long()
        if USE_CUDA:
            input = input.cuda()

        last_hidden = Variable(torch.zeros(self.decoder.num_rnn_layers, batch_size, self.decoder.hidden_size))
        if USE_CUDA:
            last_hidden = last_hidden.cuda()

        for i in range(self.max_len-1):
            output, last_hidden = self.decoder.forward_step(input, last_hidden, encoder_outputs)
            output = self.prediction(output)
            input = output.max(1)[1]
            outputs.append(output.unsqueeze(1))

        return torch.cat(outputs, dim=1)
        
    def forward_together(self, x_label, x_nolabel, y):
        x_all = torch.cat([x_label, x_nolabel], dim=0)
        out_all = self.backbone(x_all)
        
        out_label = out_all[:x_label.size(0)]
        out_nolabel = out_all[x_label.size(0):]
        
        encoder_outputs = self.encoder(out_label)

        vocab_out = self.decoder.forward_train(encoder_outputs, self.max_len, y)
        vocab_out = self.prediction(vocab_out)
        
        encoder_outputs = self.encoder(out_nolabel)

        outputs = []
        batch_size = x_nolabel.size(0)
        input = torch.zeros([batch_size]).long()
        if USE_CUDA:
            input = input.cuda()

        last_hidden = Variable(torch.zeros(self.decoder.num_rnn_layers, batch_size, self.decoder.hidden_size))
        if USE_CUDA:
            last_hidden = last_hidden.cuda()

        for i in range(self.max_len-1):
            output, last_hidden = self.decoder.forward_step(input, last_hidden, encoder_outputs)
            output = self.prediction(output)
            input = output.max(1)[1]
            outputs.append(output.unsqueeze(1))

        return vocab_out, torch.cat(outputs, dim=1)
