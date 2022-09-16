from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from noise_build import dataset_split
from sklearn.metrics import f1_score


def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


def initial_data(dataset, r, noise_mode, root_dir, args):
    print('============ Initialize data')
    num_classes = None
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    if dataset == 'cifar10':
        num_classes = 10
        test_dic = unpickle('%s/test_batch' % root_dir)
        test_data = test_dic['data']
        test_data = test_data.reshape((10000, 3, 32, 32))
        test_data = test_data.transpose((0, 2, 3, 1))
        test_label = test_dic['labels']

        for n in range(1, 6):
            dpath = '%s/data_batch_%d' % (root_dir, n)
            data_dic = unpickle(dpath)
            train_data.append(data_dic['data'])
            train_label = train_label + data_dic['labels']
        train_data = np.concatenate(train_data)
    elif dataset =='cifar100':
        num_classes = 100
        test_dic = unpickle('%s/test' % root_dir)
        test_data = test_dic['data']
        test_data = test_data.reshape((10000, 3, 32, 32))
        test_data = test_data.transpose((0, 2, 3, 1))
        test_label = test_dic['fine_labels']

        train_dic = unpickle('%s/train' % root_dir)
        train_data = train_dic['data']
        train_label = train_dic['fine_labels']
    train_data = train_data.reshape((50000, 3, 32, 32))
    train_data = train_data.transpose((0, 2, 3, 1))

    # build noise
    noise_label = dataset_split(train_images=train_data, train_labels=train_label,
                                noise_rate=r, noise_type=noise_mode,
                                random_seed=args.seed, num_classes=num_classes)

    print('============ Actual clean samples number: ', sum(np.array(noise_label) == np.array(train_label)))
    return train_data, train_label, noise_label, test_data, test_label


class cifar_dataset(Dataset):
    def __init__(self, data, real_label, label, transform, mode, strong_transform=None, pred=[], probability=[], test_log=None, id_list=None):
        self.data = None
        self.label = None
        self.transform = transform
        self.strong_aug = transform
        self.mode = mode
        self.pred = pred
        self.probability = None
        self.real_label = real_label
        self.id_list = id_list

        if self.mode == 'all' or self.mode == 'test':
            self.data = data
            self.label = label
        else:
            if self.mode == 'labeled':
                pred_idx = self.pred.nonzero()[0]
                self.probability = [probability[i] for i in pred_idx]

            elif self.mode == 'unlabeled':
                pred_idx = (1 - pred).nonzero()[0]

            self.data = data[pred_idx]
            self.label = [label[i] for i in pred_idx]
            self.id_list = pred_idx
            true_label = [self.real_label[i] for i in pred_idx]
            f1_ = f1_score(true_label, self.label, average='micro')

            print("%s data has a size of %d, f-score: %f" % (self.mode, len(pred_idx), f1_))
            test_log.write("%s data has a size of %d, f-score: %f" % (self.mode, len(pred_idx), f1_))

    def __getitem__(self, index):
        if self.mode == 'labeled':
            img, target, prob = self.data[index], self.label[index], self.probability[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            img2 = self.strong_aug(img)
            return img1, img2, target, prob
        elif self.mode == 'unlabeled':
            img = self.data[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            img2 = self.strong_aug(img)
            return img1, img2
        elif self.mode == 'all':
            img, target = self.data[index], self.label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target, index
        elif self.mode == 'test':
            img, target = self.data[index], self.label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target

    def __len__(self):
        return len(self.data)


class cifar_dataloader():
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, root_dir, args, noise_file=''):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.noise_file = noise_file
        self.args = args

        self.train_data, self.train_label, self.noise_label, self.test_data, self.test_label = initial_data(self.dataset, self.r, self.noise_mode, self.root_dir, args)

        if self.dataset == 'cifar10':
            self.transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        elif self.dataset == 'cifar100':
            self.transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ])
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ])

    def run(self, mode, pred=[], prob=[], test_log=None):
        if mode == 'warmup':
            all_dataset = cifar_dataset(self.train_data, self.train_label, self.noise_label, self.transform_train, mode='all', strong_transform=None,
                                            pred=pred, probability=prob, test_log=test_log)
            trainloader = DataLoader(dataset=all_dataset, batch_size=128, shuffle=True, num_workers=self.num_workers)
            return trainloader

        elif mode == 'train':
            labeled_dataset = cifar_dataset(self.train_data, self.train_label, self.noise_label, self.transform_train, mode='labeled',
                                            strong_transform=None, pred=pred, probability=prob, test_log=test_log)
            labeled_trainloader = DataLoader(dataset=labeled_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
            return labeled_trainloader

        elif mode == 'test':
            test_dataset = cifar_dataset(self.test_data, self.train_label, self.test_label, self.transform_train, mode='test',
                                         strong_transform=None, pred=pred, probability=prob)
            test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
            return test_loader

        elif mode == 'eval_train':
            eval_dataset = cifar_dataset(self.train_data, self.train_label, self.noise_label, self.transform_train, mode='all',
                                         strong_transform=None, pred=pred, probability=prob)
            eval_loader = DataLoader(dataset=eval_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
            return eval_loader