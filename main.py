import torch
from torch import optim
import time
import torchvision.models as models

from src.dataset import create_dataset
from src.config import get_args
from src.cnn import *
from src.trainer import *
from utils.log import make_print_to_file
from utils.plot import plot


def run():

    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Your device is: ', device)

    print('======creating dataset======')

    # Data Loader
    trainloader, testloader = create_dataset()

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

    train_loss_total = []
    train_acc_total = []
    test_loss_total = []
    test_acc_total = []

    print('======defining model======')
    if args.mode == 0 :
        print('======Ours======')
        net = CNN()
    elif args.mode == 1 :
        print('======LeNet======')
        net = LeNet()
    elif args.mode == 2:
        print('======AlexNet======')
        net = AlexNet()
    elif args.mode == 3:
        print('======GoogleNet======')
        net = models.googlenet(pretrained=False)
    elif args.mode == 4:
        print('======Resnet18======')
        net = models.resnet18(pretrained=False)
    elif args.mode == 5:
        print('======Resnet34(Pretrained)======')
        net = models.resnet34(pretrained=True)
        # Substitute the FC output layer
        net.fc = torch.nn.Linear(net.fc.in_features, 10)
        torch.nn.init.xavier_uniform_(net.fc.weight)
    elif args.mode == 6:
        print('======DenseNet161(Pretrained)======')
        net = models.densenet161(pretrained=True)
    else:
        raise Exception("No such mode")
    
    net.to(device)
    
    start_epoch = 0

    # Resume Training
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        args.best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    

    Loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
    optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)


    for epoch in range(start_epoch, start_epoch+args.num_epochs):
        tic = time.time()
        loss, acc = train(net, trainloader, epoch, Loss, optimizer, device, classes)
        train_loss_total.append(loss)
        train_acc_total.append(acc)
        toc = time.time()
        print('Train Time Spent:', toc-tic)
        loss, acc = test(net, testloader, epoch, Loss, device, classes)
        test_loss_total.append(loss)
        test_acc_total.append(acc)
        toe = time.time()
        print('Test Time Spent:', toe-toc)

    plot(train_loss_total, train_acc_total, test_loss_total, test_acc_total, args.mode)


if __name__ == '__main__':
    make_print_to_file(path='./log')
    run()
