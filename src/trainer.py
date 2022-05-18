import torch
from .config import *

batch_size = 256


def train(net, trainloader, epoch, Loss, optim, device):
    print('\n======Training in Epoch: %d======' % epoch)
    net.train()

    train_loss = 0
    correct = 0
    total = 0

    # 输出各分类准确率
    # 创建各个分类准确率的字典
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    class_correct = {classname: 0 for classname in classes}
    class_total = {classname: 0 for classname in classes}

    for batch_idx, (inputs, targets) in enumerate(trainloader):

        inputs, targets = inputs.to(device), targets.to(device)
        optim.zero_grad()
        outputs = net(inputs)

        loss = Loss(outputs, targets)
        loss.backward()
        optim.step()

        train_loss += loss.item()
        _, predictions = outputs.max(1)

        # 记录各个分类的准确率
        for target, prediction in zip(targets, predictions):
            if target == prediction:
                class_correct[classes[target]] += 1
            class_total[classes[target]] += 1

        total += targets.size(0)
        correct += predictions.eq(targets).sum().item()
        if batch_idx % 20 == 0:
            print("======Batch %d======" % batch_idx)
            print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

            # 将损失和准确率记录到对应的txt文件中
            with open('loss_records.txt', 'a+') as f:
                f.write('%d %.3f\n' % (epoch * 9 + batch_idx / 20, train_loss / (batch_idx + 1)))

            with open('accurate_records.txt', 'a+') as f:
                f.write('%d %.3f\n' % (epoch * 9 + batch_idx / 20, 100. * correct / total))

    # 打印各分类准确率
    for classname, correct_n in class_correct.items():
        accuracy = 100 * float(correct_n) / class_total[classname]
        print("Accuracy of class {:5s} is {:.1f} %".format(classname, accuracy))


def test(net, testloader, epoch, Loss, device):
    args = get_args()
    net.eval()
    print('\n======Testing in Epoch: %d======' % epoch)
    test_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)

        loss = Loss(outputs, targets)
        test_loss += loss.item()
        _, predictions = outputs.max(1)
        total += targets.size(0)
        correct += predictions.eq(targets).sum().item()

    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > args.best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        args.best_acc = acc
