
import time
import pandas as pd
import cv2
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from model.Net import FistNet
from utils.dataset import DealDataSet
device = torch.device("cpu")


def get_parse():
    parse = argparse.ArgumentParser(description='For model use')

    parse.add_argument('--type', type=str, default='test')
    # For train
    parse.add_argument('--train_file', type=str, default='./data/train.npz')
    parse.add_argument('--val_file', type=str, default='./data/val.npz')
    parse.add_argument('--test_type', type=int, default=0)
    parse.add_argument('--test_img', type=str, default='./data/test_images/test0.jpg')
    parse.add_argument('--test_video', type=str, default='./data/test_images/output.avi')
    parse.add_argument('--begin_epoch', type=int, default=0)
    parse.add_argument('--max_epoch', type=int, default=90)
    parse.add_argument('--batch_size', type=int, default=32)
    parse.add_argument('--print_batches', type=int, default=50)
    parse.add_argument('--lr', type=float, default=0.001)
    parse.add_argument('--step_lr', type=int, default=30)
    parse.add_argument('--save_address', type=str, default='./checkpoints/')
    return parse.parse_args()


def train(arg):
    # 加载数据集
    train_data = DealDataSet(arg.train_file)
    train_loader = DataLoader(dataset=train_data, batch_size=arg.batch_size, shuffle=True)

    val_data = DealDataSet(arg.val_file)
    val_loader = DataLoader(dataset=val_data, batch_size=arg.batch_size, shuffle=True)

    # 网络模型
    net = FistNet()
    net.to(device)

    # 构造一个优化器和学习率调度程序
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=arg.lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=arg.step_lr, gamma=0.1)

    # 定义损失函数
    criterion = nn.BCELoss()

    if arg.begin_epoch:
        net.load_state_dict(torch.load(arg.save_address + '%d.pth' % (arg.begin_epoch - 1)))

    dict_loss = {"epoch": [], "loss": [], "acc": []}
    for epoch in range(arg.begin_epoch, arg.max_epoch):
        net.train()
        print("*" * 100)
        print('start epoch %d' % epoch)
        running_loss = 0
        start = time.time()
        for i, data in enumerate(train_loader):
            # load data
            inputs, labels = data
            # 将这些数据转换成Variable类型
            inputs, labels = Variable(inputs), Variable(labels)

            # 接下来就是跑模型的环节
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % arg.print_batches == arg.print_batches - 1:  
                print('[%d, %5d] loss: %.3f' % (epoch, i + 1, running_loss / (arg.print_batches - 1)))
                print('%d batches cost %.2f s' % (arg.print_batches, time.time() - start))
                running_loss = 0.0
                start = time.time()

        lr_scheduler.step()
        val_loss, accuracy = val(net, val_loader, criterion)
        print('After epoch %d, loss: %.3f, accuracy: %.2f' % (epoch, val_loss, accuracy))
        dict_loss['epoch'].append(epoch)
        dict_loss['loss'].append(val_loss)
        dict_loss['acc'].append(accuracy)
        df_loss = pd.DataFrame(dict_loss)
        df_loss.to_csv(arg.save_address + 'loss_%d.csv' % arg.begin_epoch)
        torch.save(net.state_dict(), arg.save_address + '%d.pth' % epoch)


def val(model, dataloader, criterion):
    """
    计算在验证集上的准确率
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs, labels = data
            # 将这些数据转换成Variable类型
            inputs, labels = Variable(inputs), Variable(labels)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            predicted = (outputs > 0.5).view(1, -1)
            labels = labels.byte()

            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        val_loss = val_loss / (i + 1)
        accuracy = 100 * correct / total

    return val_loss, accuracy


def test(arg):
    # load model
    net = FistNet()
    net.to(device)
    net.load_state_dict(torch.load(arg.save_address + '%d.pth' % 71))
    net.eval()
    if arg.test_type:
        img = cv2.imread(arg.test_img)
        frame = cv2.resize(img, (64, 64))
        frame = torch.Tensor(frame.transpose(2, 0, 1)).unsqueeze(0).float()
        y = net(frame) > 0.5
        if y.item():
            msg = 'Positive'  # 手心朝前
        else:
            msg = 'Negative'  # 手背朝前

        font = cv2.cv2.FONT_HERSHEY_SIMPLEX
        img = cv2.putText(img, msg, (50, 300), font, 1.2, (255, 0, 0), 2)
        cv2.imwrite('Test.jpg', img)
    else:
        source = arg.test_video
        camera = cv2.VideoCapture(source)
        while True:
            ret, img = camera.read()
            if ret:
                frame = cv2.resize(img, (64, 64))
                frame = torch.Tensor(frame.transpose(2, 0, 1)).unsqueeze(0).float()
                y = net(frame) > 0.5
                if y.item():
                    msg = 'Negative'  # 手背朝前
                else:
                    msg = 'Positive'  # 手心朝前

                font = cv2.cv2.FONT_HERSHEY_SIMPLEX
                img = cv2.putText(img, msg, (50, 300), font, 1.2, (255, 0, 0), 2)
                cv2.imshow('Test', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    args = get_parse()

    if args.type == 'train':
        train(args)
    else:
        test(args)
