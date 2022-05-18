import torch
from torch import optim
import time
import matplotlib.pyplot as plt
import numpy as np

from src.dataset import create_dataset
from src.config import get_args
from src.cnn import *
from src.trainer import *
from utils.plot import make_print_to_file
 





def run():

    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Your device is: ', device)

    print('======creating dataset======')

    # Data Loader
    trainloader, testloader = create_dataset()

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

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
        train(net, trainloader, epoch, Loss, optimizer, device)
        toc = time.time()
        print('Train Time Spent:', toc-tic)
        test(net, testloader, epoch, Loss, device, classes)
        toe = time.time()
        print('Test Time Spent:', toe-toc)
    
    # 绘制损失和准确率随训练轮数变化的曲线。
    data_loss = np.loadtxt("loss_records.txt")
    data_accuracy = np.loadtxt("accurate_records.txt")
    x = data_loss[:, 0]
    y = data_loss[:, 1]
    x1 = data_accuracy[:, 0]
    y1 = data_accuracy[:, 1]

    fig, axes = plt.subplots(nrows=1, ncols=2)

    axes[0].plot(x, y, 'r-', label=u'training_loss')
    axes[0].set(title='loss in training',
                ylabel='training_loss', xlabel='epoch')

    axes[1].plot(x1, y1, 'b-', label=u'accurate')
    axes[1].set(title='accuracy in training',
                ylabel='accurate', xlabel='epoch')

    plt.show()

if __name__ == '__main__':
    make_print_to_file(path='./log')
    run()
