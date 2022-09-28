import numpy as np
import matplotlib.pyplot as plt

groups = [3, 4, 5, 6, 7, 8, 9, 10]

no = 1
# plt.figure(figsize=(10, 10))
for g in groups:
    dist_acc = np.load('MNIST/plot/dist_acc_g{}.npy'.format(g))
    dist_loss = np.load('MNIST/plot/dist_loss_g{}.npy'.format(g))
    delta_acc = np.load('MNIST/plot/delta_acc_g{}.npy'.format(g))
    delta_loss = np.load('MNIST/plot/delta_loss_g{}.npy'.format(g))

    pltidx = np.arange(1, g + 1)

    plt.subplot(4, 4, no)
    plt.plot(pltidx, dist_acc, label='d acc')
    plt.plot(pltidx, delta_acc, label='Δd acc')
    plt.xlabel('groups')
    plt.ylabel('accuracy')
    plt.xticks(pltidx)
    plt.title('accuracy of {} groups'.format(g))
    plt.legend()
    
    plt.subplot(4, 4, no + 1)
    plt.plot(pltidx, dist_loss, label='d loss')
    plt.plot(pltidx, delta_loss, label='Δd loss')
    plt.xlabel('groups')
    plt.ylabel('loss')
    plt.xticks(pltidx)
    plt.title('loss of {} groups'.format(g))
    plt.legend()

    np.save('MNIST/plot/dist_loss_g{}'.format(g), dist_loss)
    np.save('MNIST/plot/dist_acc_g{}'.format(g), dist_acc)
    np.save('MNIST/plot/delta_loss_g{}'.format(g), delta_loss)
    np.save('MNIST/plot/delta_acc_g{}'.format(g), delta_acc)

    no += 2

plt.show()