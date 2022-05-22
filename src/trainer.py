import torch
from .config import *


def train(net, trainloader, epoch, Loss, optim, device,classes):
    args = get_args()
    print('\n======Training in Epoch: %d======' % epoch)
    net.train()

    train_loss = 0
    correct = 0
    total = 0

    # 输出各分类准确率
    # 创建各个分类准确率的字典
    class_correct = {classname: 0 for classname in classes}
    class_total = {classname: 0 for classname in classes}

    for batch_idx, (inputs, targets) in enumerate(trainloader):

        inputs, targets = inputs.to(device), targets.to(device)
        optim.zero_grad()
        outputs = net(inputs)
        
        if args.mode == 4: # GoogleNet需要将输出转换为tensor
            outputs = outputs.logits

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
        if batch_idx % 50 == 0:
            print("======Batch %d======" % batch_idx)
            print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        total_batch = batch_idx + 1


    loss_total = train_loss / total_batch
    acc_total = correct / total

    print('EPOCH %d TOTAL: Loss: %.3f | Acc: %.3f%% (%d/%d)' % (epoch, loss_total, 100. * acc_total, correct, total))
    # 打印各分类准确率
    for classname, correct_n in class_correct.items():
        accuracy = 100 * float(correct_n) / class_total[classname]
        print("Accuracy of class {:5s} is {:.1f} %".format(classname, accuracy))

    return loss_total, acc_total


def test(net, testloader, epoch, Loss, device, classes):
    args = get_args()
    net.eval()
    print('\n======Testing in Epoch: %d======' % epoch)
    test_loss = 0
    correct = 0
    total = 0

    class_correct = {classname: 0 for classname in classes}
    class_total = {classname: 0 for classname in classes}

    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)

        loss = Loss(outputs, targets)
        test_loss += loss.item()
        _, predictions = outputs.max(1)
        total += targets.size(0)
        correct += predictions.eq(targets).sum().item()

        for target, prediction in zip(targets, predictions):
            if target == prediction:
                class_correct[classes[target]] += 1
            class_total[classes[target]] += 1

    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    # 打印各分类准确率
    for classname, correct_n in class_correct.items():
        accuracy = 100 * float(correct_n) / class_total[classname]
        print("Accuracy of class {:5s} is {:.1f} %".format(classname, accuracy))

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
        torch.save(state, './checkpoint/ckpt_'+str(int(acc))+'.pth')
        args.best_acc = acc

    return test_loss / (batch_idx + 1), correct / total