import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader , random_split , Subset
import torch
import numpy as np
import random

torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True

class loader():

    def __init__(self, args):
        super(loader, self).__init__()

        mnist_transform = transforms.Compose([transforms.ToTensor()])
        download_root = 'D:/2020-2/비즈니스애널리틱스/논문리뷰/Stacked Convolutional Auto-Encoders for Hierarchical Feature Extraction/MNIST_DATASET'

        dataset = MNIST(download_root, transform=mnist_transform, train=True, download=True)
        normal_class_idx = dataset.targets != args.abnormal_class # args.abnormal_class is zero class
        dataset.targets = dataset.targets[normal_class_idx]
        dataset.data = dataset.data[normal_class_idx]
				'''train dataset은 1과 9 사이의 정상 데이터로만 구성한다.'''

        train_dataset , valid_dataset = random_split(dataset , [int(dataset.__len__()*0.8), dataset.__len__() - int(dataset.__len__()*0.8) ])
				'''train 80% , validation 20% split'''


        test_dataset = MNIST(download_root, transform=mnist_transform, train=False, download=True)
        normal_class_idx = torch.where(test_dataset.targets != args.abnormal_class)[0].numpy()
        novelty_class_idx = torch.where(test_dataset.targets == args.abnormal_class)[0].numpy()
        temp_idx = np.random.choice(normal_class_idx, size=novelty_class_idx.__len__())
        test_idx = np.concatenate([novelty_class_idx, temp_idx])
        '''test data는 비정상 클래스인 0과 정상 클래스인 1과 9 사이의 숫자로 구성된다. 이때, 비정상과 정상간의 클래스 비율은 50:50이다.'''
        test_dataset.targets = test_dataset.targets[test_idx]
        test_dataset.data = test_dataset.data[test_idx]

        self.batch_size = args.batch_size
        self.train_iter = DataLoader(dataset=train_dataset , batch_size=self.batch_size , shuffle=True)
        self.valid_iter = DataLoader(dataset=valid_dataset , batch_size=self.batch_size , shuffle=True)
        self.test_iter = DataLoader(dataset=test_dataset , batch_size=self.batch_size , shuffle=True)