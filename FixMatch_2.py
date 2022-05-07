import argparse
import torch
from torch import optim
import matplotlib
from torch.autograd import Variable
import pprint

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_datasets_mean_teacher
from models import CNNSeq2Seq
from util import compute_seq_acc, Seq2SeqLoss, ConsistentLoss, ConsistentLoss_MT_Temperature, \
    get_current_consistency_weight
import random
import time

parser = argparse.ArgumentParser(description='PyTorch Captcha Training Using Mean-Teacher')

parser.add_argument('--dataset', default='google', type=str, help="the name of dataset")
parser.add_argument('--label', default="500.txt", type=str, help='the labels of captcha images used for training')
parser.add_argument('--batch-size', default=32, type=int, help='batch size for training and test')
parser.add_argument('--secondary-batch-size', default=64, type=int, help='batch size for unlabel')
parser.add_argument('--unlabeled-number', default=5000, type=int, help='the number of unlabeled images')
parser.add_argument('--threshold', default=0.95, type=float, help='pseudo label threshold')
parser.add_argument('--epoch', default=700, type=int, help='the number of training epochs')
parser.add_argument('--t', default=0.5, type=float, help='temperature of MT')
parser.add_argument('--weight', default=100.0, type=float, help='the weight of consistency loss')
parser.add_argument('--teachforce', action="store_false", help='whether to use teaching force(Default: True)')
parser.add_argument('--lr', default=0.02, type=float, help='learning rate')
parser.add_argument('--seed', default=42, type=int, help='running seed')

args = parser.parse_args()

SEED = args.seed
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.benchmark = True

pprint.pprint(args)
USE_CUDA = torch.cuda.is_available()

LR = args.lr
NUM_EPOCHS = args.epoch

dataloader_train_labeled, dataloader_train_nolabeled, dataloader_test, id2token, MAXLEN, _ = load_datasets_mean_teacher(
    args)

print("token:", "".join(list(id2token.values())))

model = CNNSeq2Seq(vocab_size=len(id2token), max_len=MAXLEN)
model_ema = CNNSeq2Seq(vocab_size=len(id2token), max_len=MAXLEN)

class_criterion = Seq2SeqLoss()
consistent_criterion = ConsistentLoss(args.threshold)
consistent_criterion_mt = ConsistentLoss_MT_Temperature(args.t)

if USE_CUDA:
    model = model.cuda()
    model_ema = model_ema.cuda()
    class_criterion = class_criterion.cuda()
    consistent_criterion = consistent_criterion.cuda()
    consistent_criterion_mt = consistent_criterion_mt.cuda()

for param_main, param_ema in zip(model.parameters(), model_ema.parameters()):
    param_ema.data.copy_(param_main.data)  # initialize
    param_ema.requires_grad = False  # not update by gradient

params = list(filter(lambda p: p.requires_grad, model.parameters()))
optimizer = optim.SGD(params, lr=LR, momentum=0.9, weight_decay=5e-4, nesterov=True)

train_loss_class = []
train_loss_consistency = []
train_loss_consistency_mt = []
train_accclevel = []
train_accuracy = []

test_class_loss = []
test_accclevel = []
test_accuracy = []

test_class_loss_ema = []
test_accclevel_ema = []
test_accuracy_ema = []

for epoch in range(NUM_EPOCHS):
    time_epoch = time.time()
    loss_1 = loss_2 = loss_mt = accuracy = accclevel = 0

    iter_label = dataloader_train_labeled.__iter__()
    for num_iter, (inputs_u_w, inputs_u_s) in enumerate(dataloader_train_nolabeled):
        inputs_x, targets_x = iter_label.next()

        if USE_CUDA:
            targets_x = targets_x.cuda()
            inputs_x = inputs_x.cuda()
            inputs_u_w = inputs_u_w.cuda()
            inputs_u_s = inputs_u_s.cuda()

        batch_size = inputs_x.size(0)
        if args.teachforce:
            logits_x = model.forward_train(inputs_x, targets_x)
            logits_u_w = model_ema.forward_test(inputs_u_w)
            logits_u_s = model.forward_test(inputs_u_s)
        else:
            logits_x = model.forward_test(inputs_x)
            logits_u_w = model_ema.forward_test(inputs_u_w)
            logits_u_s = model.forward_test(inputs_u_s)

        optimizer.zero_grad()

        Lx = class_criterion(logits_x, targets_x)

        Lu = 2 * consistent_criterion(logits_u_w.detach(), logits_u_s)
        Lu_mt = get_current_consistency_weight(args.weight, args.epoch - 100, epoch) * consistent_criterion_mt(
            logits_u_s, logits_u_w.detach())

        max_len = targets_x.size(1)
        acccl, acc = compute_seq_acc(logits_x, targets_x, max_len)
        acccl /= args.batch_size
        acc /= args.batch_size

        loss_all = Lx + Lu + Lu_mt
        loss_all.backward()
        optimizer.step()

        for ema_param, param in zip(model_ema.parameters(), model.parameters()):
            ema_param.data.mul_(0.999).add_(1 - 0.999, param.data)

        loss_1 += Lx.item()
        loss_2 += Lu.item()
        loss_mt += Lu_mt.item()
        accclevel += acccl
        accuracy += acc

    train_loss_class.append(loss_1 / len(dataloader_train_nolabeled))
    train_loss_consistency.append(loss_2 / len(dataloader_train_nolabeled))
    train_loss_consistency_mt.append(loss_mt / len(dataloader_train_nolabeled))
    train_accclevel.append(accclevel / len(dataloader_train_nolabeled))
    train_accuracy.append(accuracy / len(dataloader_train_nolabeled))
    print("{} epoch train\n"
          "class loss: {} consistent loss {} consistent loss mt {}\n"
          "accuracy {} accclevel {}".format(epoch, train_loss_class[-1], train_loss_consistency[-1],
                                            train_loss_consistency_mt[-1], train_accuracy[-1],
                                            train_accclevel[-1]))

    model = model.eval()
    loss = accuracy = accclevel = total = 0
    for num_iter, (x, y) in enumerate(dataloader_test):
        x = Variable(x)
        y = Variable(y)
        if USE_CUDA:
            x = x.cuda()
            y = y.cuda()

        outputs = model.forward_test(x)

        loss_batch = class_criterion(outputs, y)

        max_len = y.size(1)
        acccl, acc = compute_seq_acc(outputs, y, max_len)

        loss += loss_batch.item()
        accclevel += acccl
        accuracy += acc
        total += y.size(0)

    test_class_loss.append(loss / len(dataloader_test))
    test_accclevel.append(accclevel / total)
    test_accuracy.append(accuracy / total)
    print("test loss: {}\n"
          "accuracy {} accclevel {}".format(test_class_loss[-1], test_accuracy[-1], test_accclevel[-1]))
    print(f"{accuracy}/{total}")
    model = model.train()

    model_ema = model_ema.eval()
    loss = accuracy = accclevel = total = 0
    for num_iter, (x, y) in enumerate(dataloader_test):
        x = Variable(x)
        y = Variable(y)
        if USE_CUDA:
            x = x.cuda()
            y = y.cuda()

        outputs = model_ema.forward_test(x)

        loss_batch = class_criterion(outputs, y)

        max_len = y.size(1)
        acccl, acc = compute_seq_acc(outputs, y, max_len)

        loss += loss_batch.item()
        accclevel += acccl
        accuracy += acc
        total += y.size(0)

    test_class_loss_ema.append(loss / len(dataloader_test))
    test_accclevel_ema.append(accclevel / total)
    test_accuracy_ema.append(accuracy / total)
    print("test loss: {}\n"
          "accuracy {} accclevel {}".format(test_class_loss_ema[-1], test_accuracy_ema[-1], test_accclevel_ema[-1]))
    print(f"{accuracy}/{total}")
    model_ema = model_ema.train()

    print(f"epoch time {time.time()-time_epoch}\n")

    if (epoch + 1) % 100 == 0:
        fig = plt.figure(figsize=(20, 10))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.plot(train_loss_class, 'r', label='train_class_loss')
        ax1.plot(train_loss_consistency, 'y', label='train_consistency_loss')
        ax1.plot(train_loss_consistency_mt, 'm', label='train_loss_consistency_mt')
        ax1.plot(test_class_loss, 'b', label='test_class_loss')
        ax1.plot(test_class_loss_ema, 'g', label='test_class_loss_ema')

        ax1.legend()
        ax2.plot(test_accuracy, 'b', label='test_accuracy')
        ax2.plot(test_accuracy_ema, 'g', label='test_accuracy_ema')
        ax2.plot(train_accuracy, 'r', label='train_accuracy')
        ax2.legend()
        test_acc_array = np.array(test_accuracy_ema)
        max_indx = np.argmax(test_acc_array)
        show_max = '[' + str(max_indx) + " " + str(test_acc_array[max_indx].item()) + ']'
        ax2.annotate(show_max, xytext=(max_indx, test_acc_array[max_indx].item()),
                     xy=(max_indx, test_acc_array[max_indx].item()))

        path = "FixMatch_2_" + args.dataset + "_" + str(args.label) + "_" + str(args.unlabeled_number) + "_" + str(
            args.teachforce) + "_" + str(args.weight) + "_" + str(args.threshold) + "_" + str(args.t) + "_" + str(
            args.seed)
        fig.savefig("result/" + path + ".png")
        np.save("result/" + path + "_test_accuracy_ema.npy", np.array(test_accuracy_ema))
        np.save("result/" + path + "_train_accuracy.npy", np.array(train_accuracy))
        np.save("result/" + path + "_test_class_loss_ema.npy", np.array(test_class_loss_ema))
        np.save("result/" + path + "_train_loss_class.npy", np.array(train_loss_class))

fig = plt.figure(figsize=(20, 10))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.plot(train_loss_class, 'r', label='train_class_loss')
ax1.plot(train_loss_consistency, 'y', label='train_consistency_loss')
ax1.plot(train_loss_consistency_mt, 'm', label='train_loss_consistency_mt')
ax1.plot(test_class_loss, 'b', label='test_class_loss')
ax1.plot(test_class_loss_ema, 'g', label='test_class_loss_ema')

ax1.legend()
ax2.plot(test_accuracy, 'b', label='test_accuracy')
ax2.plot(test_accuracy_ema, 'g', label='test_accuracy_ema')
ax2.plot(train_accuracy, 'r', label='train_accuracy')
ax2.legend()
test_acc_array = np.array(test_accuracy_ema)
max_indx = np.argmax(test_acc_array)
show_max = '[' + str(max_indx) + " " + str(test_acc_array[max_indx].item()) + ']'
ax2.annotate(show_max, xytext=(max_indx, test_acc_array[max_indx].item()),
             xy=(max_indx, test_acc_array[max_indx].item()))

path = "FixMatch_2_" + args.dataset + "_" + str(args.label) + "_" + str(args.unlabeled_number) + "_" + str(
    args.teachforce) + "_" + str(args.weight) + "_" + str(args.threshold) + "_" + str(args.t) + "_" + str(args.seed)
fig.savefig("result/" + path + ".png")
np.save("result/" + path + "_test_accuracy_ema.npy", np.array(test_accuracy_ema))
np.save("result/" + path + "_train_accuracy.npy", np.array(train_accuracy))
np.save("result/" + path + "_test_class_loss_ema.npy", np.array(test_class_loss_ema))
np.save("result/" + path + "_train_loss_class.npy", np.array(train_loss_class))
