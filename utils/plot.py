import matplotlib.pyplot as plt
import numpy as np

def plot(train_loss_total, train_acc_total, test_loss_total, test_acc_total):
    # 绘制损失和准确率随训练轮数变化的曲线。
    fig, axes = plt.subplots(nrows=1, ncols=2)

    train_x = np.arange(len(train_loss_total))
    train_y = np.asarray(train_loss_total)
    test_x = np.arange(len(test_loss_total))
    test_y = np.asarray(test_loss_total)

    axes[0].plot(train_x, train_y, 'r-', label=u'train_loss')
    axes[0].plot(test_x, test_y, 'b-', label=u'test_loss')
    axes[0].set(title='loss in training',
                ylabel='training_loss', xlabel='epoch')
    axes[0].legend(loc='upper right')

    train_x = np.arange(len(train_acc_total))
    train_y = np.asarray(train_acc_total)
    test_x = np.arange(len(test_acc_total))
    test_y = np.asarray(test_acc_total)


    axes[1].plot(train_x, train_y, 'r-', label=u'train_acc')
    axes[1].plot(test_x, test_y, 'b-', label=u'test_acc')
    axes[1].set(title='accuracy in training',
                ylabel='accurate', xlabel='epoch')
    axes[1].legend(loc='upper left')
    plt.savefig('./fig/result.png')