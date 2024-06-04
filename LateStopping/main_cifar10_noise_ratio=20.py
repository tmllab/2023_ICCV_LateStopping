import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import csv
from noise_build import dataset_split


root_dir = './cifar-10'
EPOCH = 300
BATCH_SIZE = 32
LR = 0.01
r = 0.2 # noise ratio
argsseed = 1
noise_mode = 'sym'
#Instance-dependent label noise: 'instance'
#Symmetric label noise: 'sym'
#Pairflip label noise: 'pair'
num_classes = 10
model_type = 'resnet18'
WARM_UP_EPOCHS = 6
file_name = f'Results_ce_{noise_mode}{r}_cifar10_lr{LR}_bs{BATCH_SIZE}'
print(file_name)
remove_rate_1 = 0.98
remove_rate_2 = 0.98
remove_rate_3 = 0.98
I_rate_1 = 4
I_rate_2 = 4
I_rate_3 = 4

with open(file_name + '.csv', "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['round_num', 'epoch', 'acc_etrain', 'acc_test', 'FkL-Examples', 'Noise samples in FkL-Examples'])


orignal_train_data = None
orignal_train_label = None
orignal_noise_label = None
orignal_test_data = None
orignal_test_label = None
orignal_val_label = None
orignal_val_data = None
global FkLexample_dataset_global
FkLexample_dataset_global = []


class cifar_dataset(Dataset):
    def __init__(self, data, real_label, label, roundindex, transform, mode, strong_transform=None, pred=[],
                 probability=[], test_log=None, id_list=None):
        self.data = data
        self.label = label
        self.transform = transform
        self.strong_aug = transform
        self.mode = mode
        self.pred = pred
        self.probability = probability
        self.real_label = real_label
        self.id_list = id_list
        self.roundindex = roundindex

    def __getitem__(self, index):
        if self.mode == 'all':
            img, target = self.data[index], self.label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target, index
        elif self.mode == 'roundtrain':
            img, target = self.data[index], self.label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            roundindex1 = self.roundindex[index]
            return img, target, roundindex1
        elif self.mode == 'test':
            img, target = self.data[index], self.label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target
        elif self.mode == 'val':
            img, target = self.data[index], self.label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target

    def __len__(self):
        return len(self.data)


class cifar_dataloader():
    def __init__(self, r, noise_mode, batch_size, num_workers, random_seed):
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_seed = random_seed

        self.train_data, self.train_label, self.noise_label, self.test_data, self.test_label, self.val_set, self.val_labels = self.initial_data()
        self.roundindex = []

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

    def unpickle(self, file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def initial_data(self):
        print('============ Initialize data')
        train_data = []
        train_label = []
        test_data = []
        test_label = []
        noise_label = []
        num_classes = 10
        test_dic = self.unpickle('%s/test_batch' % root_dir)
        test_data = test_dic[b'data']
        test_data = test_data.reshape((10000, 3, 32, 32))
        test_data = test_data.transpose((0, 2, 3, 1))
        test_label = test_dic[b'labels']

        for n in range(1, 6):
            dpath = '%s/data_batch_%d' % (root_dir, n)
            data_dic = self.unpickle(dpath)
            train_data.append(data_dic[b'data'])
            train_label = train_label + data_dic[b'labels']

        train_data = np.concatenate(train_data)
        train_data = train_data.reshape((50000, 3, 32, 32))
        train_data = train_data.transpose((0, 2, 3, 1))

        noise_label = dataset_split(train_images=train_data,
                                    train_labels=train_label,
                                    noise_rate=self.r,
                                    noise_type=self.noise_mode,
                                    random_seed=self.random_seed,
                                    num_classes=num_classes)

        print('============ Actual clean samples number: ', sum(np.array(noise_label) == np.array(train_label)))

        num_samples = int(noise_label.shape[0])
        np.random.seed(self.random_seed)
        train_set_index = np.random.choice(num_samples, int(num_samples * 0.9), replace=False)
        index = np.arange(train_data.shape[0])
        val_set_index = np.delete(index, train_set_index)

        train_set = train_data[train_set_index, :]
        val_set = train_data[val_set_index, :]
        train_labels = noise_label[train_set_index]
        val_labels = noise_label[val_set_index]
        train_clean_labels = np.array(train_label)[train_set_index]
        val_clean_labels = np.array(train_label)[val_set_index]

        global orignal_train_data, orignal_train_label, orignal_noise_label, orignal_test_data, orignal_test_label, orignal_val_label, orignal_val_data
        orignal_train_data = train_set
        orignal_train_label = train_clean_labels
        orignal_noise_label = train_labels
        orignal_test_data = test_data
        orignal_test_label = test_label
        orignal_val_label = val_labels
        orignal_val_data = val_set

        return train_set, train_clean_labels, train_labels, test_data, test_label, val_set, val_labels

    def run(self, mode, pred=[], prob=[], test_log=None):
        if mode == 'train':
            train_dataset = cifar_dataset(self.train_data, self.train_label, self.noise_label, self.roundindex,
                                          self.transform_train, mode='all', strong_transform=None, pred=pred,
                                          probability=prob, test_log=test_log)
            train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True,
                                      num_workers=self.num_workers)
            return train_loader

        elif mode == 'etrain':
            etrain_dataset = cifar_dataset(self.train_data, self.train_label, self.noise_label, self.roundindex,
                                           self.transform_train, mode='all', strong_transform=None, pred=pred,
                                           probability=prob, test_log=test_log)
            etrain_loader = DataLoader(dataset=etrain_dataset, batch_size=1000, shuffle=False,
                                       num_workers=self.num_workers)
            return etrain_loader

        elif mode == 'test':
            test_dataset = cifar_dataset(self.test_data, self.train_label, self.test_label, self.roundindex,
                                         self.transform_train, mode='test', strong_transform=None, pred=pred,
                                         probability=prob)
            test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False, num_workers=self.num_workers)
            return test_loader

        elif mode == 'val':
            val_dataset = cifar_dataset(self.val_set, self.train_label, self.val_labels, self.roundindex,
                                        self.transform_train, mode='test', strong_transform=None, pred=pred,
                                        probability=prob)
            val_loader = DataLoader(dataset=val_dataset, batch_size=1000, shuffle=False, num_workers=self.num_workers)
            return val_loader


class cifar_dataloader1():
    def __init__(self, r, noise_mode, batch_size, num_workers, random_seed, round_num):
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_seed = random_seed
        self.round_num = round_num

        global orignal_train_label, orignal_train_data, orignal_noise_label, orignal_test_data, orignal_test_label
        self.train_data = orignal_train_data
        self.train_label = orignal_train_label
        self.noise_label = orignal_noise_label
        self.test_data = orignal_test_data
        self.test_label = orignal_test_label

        self.train_data_1, self.train_label_1, self.noise_label_1, self.roundindex = self.remove_round1(self.train_data,
                                                                                                        self.round_num)

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

    def remove_round1(self, train_data, round_num):
        global orignal_train_label, orignal_noise_label
        train_data_1 = []
        train_label_1 = []
        noise_label_1 = []
        roundindex = []

        global FkLexample_dataset_global

        length = len(FkLexample_dataset_global)
        for epoch_r in range(length):
            aka = int(FkLexample_dataset_global[epoch_r])
            train_data_1.append(train_data[aka])
            train_label_1.append(orignal_train_label[aka])
            noise_label_1.append(orignal_noise_label[aka])
            roundindex.append(aka)

        return train_data_1, train_label_1, noise_label_1, roundindex

    def run(self, mode, pred=[], prob=[], test_log=None):
        if mode == 'train1':
            labeled_dataset1 = cifar_dataset(self.train_data_1, self.train_label_1, self.noise_label_1, self.roundindex,
                                             self.transform_train, mode='roundtrain', strong_transform=None, pred=pred,
                                             probability=prob, test_log=test_log)
            train_loader1 = DataLoader(dataset=labeled_dataset1, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.num_workers)
            return train_loader1

        elif mode == 'etrain1':
            labeled_dataset1 = cifar_dataset(self.train_data_1, self.train_label_1, self.noise_label_1, self.roundindex,
                                             self.transform_train, mode='roundtrain', strong_transform=None, pred=pred,
                                             probability=prob, test_log=test_log)
            etrain_loader1 = DataLoader(dataset=labeled_dataset1, batch_size=1000, shuffle=False,
                                        num_workers=self.num_workers)
            return etrain_loader1


def build_model():
    if model_type == 'resnet18':
        from resnetnew import ResNet18
        model = ResNet18(num_classes)
        print('===================Use Resnet18===================')
    elif model_type == 'resnet34':
        from resnetnew import ResNet34
        model = ResNet34(num_classes)
        print('===================Use Resnet34===================')
    model = model.cuda()
    return model


def count_mislabeled_FkL_examples(FkLexample_dataset, orignal_noise_label, orignal_train_label):
    mislabeled_count = 0
    for example_index in FkLexample_dataset:
        if orignal_noise_label[example_index] != orignal_train_label[example_index]:
            mislabeled_count += 1
    return mislabeled_count


loader = cifar_dataloader(r=r, noise_mode=noise_mode, batch_size=BATCH_SIZE, num_workers=24, random_seed=argsseed)

test_loader = loader.run('test')
train_loader_0 = loader.run('train')
etrain_loader_0 = loader.run('etrain')
val_loader = loader.run('val')

epoch_data = {}

I_rate = I_rate_1 + I_rate_2 + I_rate_3 + 1

print("===================Main===================")
print("Start Training!")

for ite in range(I_rate):
    round_num = int(ite)
    print('round_num:', round_num)

    net = build_model()

    if round_num >= int(I_rate - 1):
        LR = 0.02
        BATCH_SIZE = 32

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

    if round_num >= int(I_rate - 1):
        sch_lr = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20, 40, 60, 80], 0.5)

    if round_num >= 1:
        loader1 = cifar_dataloader1(r=r, noise_mode=noise_mode, batch_size=BATCH_SIZE, num_workers=24,
                                    random_seed=argsseed, round_num=round_num)
        train_loader = loader1.run('train1')
        etrain_loader = loader1.run('etrain1')
    else:
        train_loader = train_loader_0
        etrain_loader = etrain_loader_0

    print('===================New Round===================')

    for epoch in range(EPOCH):
        print('\nEpoch: %d' % (epoch + 1))
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0

        for batch_idx, (inputs, labels, path) in enumerate(train_loader):
            length = len(train_loader)
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()

        if round_num >= int(I_rate - 1):
            sch_lr.step()
        print("Waiting etrain!")

        with torch.no_grad():
            correct = 0
            total = 0
            sample_counter = 0
            etrain_results = []

            for batch_idx, (inputs, labels, path) in enumerate(etrain_loader):
                net.eval()
                images, labels = inputs, labels
                images, labels = images.cuda(), labels.cuda()
                outputs = net(images)
                _, predicted1 = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted1 == labels).sum().item()

                for i in range(len(labels)):
                    t = sample_counter
                    sample_counter += 1
                    predicted_np1 = int(predicted1[i].cpu().detach().numpy())
                    labels_np1 = int(labels[i].cpu().detach().numpy())
                    o_index = int(path[i].cpu().detach().numpy())
                    q = (t, labels_np1, predicted_np1, o_index)
                    etrain_results.append(q)

            epoch_data[(round_num, epoch)] = etrain_results
            print('etrain_Acc：%.3f%%' % (100 * correct / total))
            acc_etrain = 100 * correct / total

        print("Waiting Test!")

        with torch.no_grad():
            correct = 0
            total = 0

            for batch_idx, (inputs, labels) in enumerate(test_loader):
                net.eval()
                images, labels = inputs, labels
                images, labels = images.cuda(), labels.cuda()
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()

            print('Test_Acc：%.3f%%' % (100 * correct / total))
            acc_test = 100 * correct / total
            acc_test = acc_test.item()

        print("Waiting Val!")

        with torch.no_grad():
            correct = 0
            total = 0

            for batch_idx, (inputs, labels) in enumerate(val_loader):
                net.eval()
                images, labels = inputs, labels
                images, labels = images.cuda(), labels.cuda()
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()

            print('Val_Acc：%.3f%%' % (100 * correct / total))
            acc_val = 100 * correct / total
            acc_val = acc_val.item()

        counterright = [0] * 45000
        sum_FkLexample_dataset = 0
        FkLexample_dataset = []
        rn = round_num + 1
        list = [101] * 45000
        lista = [101] * 45000
        Noiserate = 0

        if round_num == I_rate - 1:
            print("Final round!")
        else:
            for index in range(epoch):
                if index > WARM_UP_EPOCHS:
                    reader = epoch_data[(round_num, index)]

                    if rn <= int(I_rate_1):
                        FkL = 0
                        for t, label, prediction, path in reader:
                            if label == prediction:
                                counterright[t] = int(counterright[t]) + 1
                                if counterright[t] == 1:
                                    FkL += 1
                                    FkLexample_dataset.append(path)
                        sum_FkLexample_dataset += FkL

                    elif rn <= int(I_rate_2 + I_rate_1):
                        FkL = 0
                        for t, label, prediction, path in reader:
                            if label == prediction:
                                if prediction == list[t]:
                                    counterright[t] = int(counterright[t]) + 1
                                    if counterright[t] == 1:
                                        FkL += 1
                                        FkLexample_dataset.append(path)
                            list[t] = prediction
                        sum_FkLexample_dataset += FkL

                    else:
                        FkL = 0
                        for t, label, prediction, path in reader:
                            if label == prediction:
                                if prediction == list[t] and prediction == lista[t]:
                                    counterright[t] = int(counterright[t]) + 1
                                    if counterright[t] == 1:
                                        FkL += 1
                                        FkLexample_dataset.append(path)
                            lista[t] = list[t]
                            list[t] = prediction
                        sum_FkLexample_dataset += FkL

        Threshold = 0

        if rn > I_rate_1:
            if rn > int(I_rate_2 + I_rate_1):
                if rn > int(I_rate_2 + I_rate_1 + I_rate_3):
                    if epoch > 998:
                        print('Training Finished!!!')
                        break
                    else:
                        Threshold = 45000
                else:
                    Threshold = int(45000 * (((remove_rate_1) ** (I_rate_1)) * ((remove_rate_2) ** (I_rate_2)) * (
                                (remove_rate_3) ** (rn - (I_rate_1 + I_rate_2)))))
            else:
                Threshold = int(45000 * (((remove_rate_1) ** (I_rate_1)) * ((remove_rate_2) ** (rn - I_rate_1))))
        else:
            Threshold = int(45000 * ((remove_rate_1) ** (rn)))

        print("Threshold:", Threshold)

        if sum_FkLexample_dataset >= int(Threshold):
            print("Round Training Finished, TotalEPOCH=%d, FkL-Examples=%d" % (epoch, sum_FkLexample_dataset))
            FkLexample_dataset_global = FkLexample_dataset
            break
        else:
            print('Round Training continue, FkL-Examples = ', sum_FkLexample_dataset)
            mislabeled_count = count_mislabeled_FkL_examples(FkLexample_dataset, orignal_noise_label,
                                                             orignal_train_label)
            print(f"Mislabeled FkL Examples: {mislabeled_count}")
            print(f"Clean FkL Examples: {(sum_FkLexample_dataset - mislabeled_count)}")

            if sum_FkLexample_dataset != 0:
                Noiserate = mislabeled_count
                print(f"Noise rate: {Noiserate}")

        with open(file_name + '.csv', "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                [round_num, epoch, f"{acc_etrain:.2f}", f"{acc_test:.2f}", sum_FkLexample_dataset, Noiserate])