import torch
from torch import optim
import time


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
        net = LeNet()
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

    plot(train_loss_total, train_acc_total, test_loss_total, test_acc_total)


if __name__ == '__main__':
    make_print_to_file(path='./log')
    run()
