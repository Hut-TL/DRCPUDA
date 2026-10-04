import random
import torch
import numpy as np
import scipy.io as scio
from torch.utils.data import Dataset, DataLoader


class TrainSet(Dataset):
    def __init__(self, X, Y):
        self.X, self.Y = X, Y

    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]

        return x.reshape(1, 2048), y
        # return x, y

    def __len__(self):
        return len(self.X)


def load_Matdata_to_Txt(root, dataFileNames):
    """
    root :
    datanames :
    """
    for i in range(len(dataFileNames)):
        # 将每个故障的drive end 数据保存在文件中
        data = scio.loadmat(root + dataFileNames[i])
        # 要拿到这个数据 太抽象了 只能一层一层访问，找到具体的位置
        print(data.keys())
        position_sign = list(data.keys())[3]
        print(position_sign)
        print(len(data[position_sign][0][0][2][0][6][2][0]))
        file_data = data[position_sign][0][0][2][0][6][2][0]
        file_data = list(np.ravel(file_data))
        file_data = str(file_data)
        file_data = file_data[1:-1]
        print(type(file_data))
        file_data = file_data.replace(", ", "\n")
        f = open(root + str(i) + "_" + position_sign + ".txt", "w")
        f.writelines(file_data)
        f.close()


def set_MyDataset(root, dataset_txt_fileNames, separator, fixed_label=None, unlabel=None):
    """
    :param root:
    :param DatasetFileNames:
    :return:
    """
    data = []
    label = []
    for i, l in zip(range(len(dataset_txt_fileNames)), fixed_label):
        if separator == '\n':
            file_data = np.loadtxt(str(root) + dataset_txt_fileNames[i])
        else:
            file_data = np.loadtxt(str(root) + dataset_txt_fileNames[i], delimiter=separator)
        file_data = list(file_data)
        # 进行数据集切块
        j = 0
        counnt = 0
        if fixed_label == None:
            while j < len(file_data):
                if counnt == 600:
                    break
                if len(file_data[j:j + 2048]) == 2048:
                    data.append(file_data[j:j + 2048])
                    label.append(i)
                j += 400
                counnt += 1
        else:
            while j < len(file_data):
                if counnt == 600:
                    break
                if len(file_data[j:j + 2048]) == 2048:
                    data.append(file_data[j:j + 2048])
                if l >= unlabel:
                    label.append(unlabel)
                else:
                    label.append(l)
                j += 400
                counnt += 1

    return data, label


def get_dataloader(root, data_file_name, ratio, batch_size, shuffle=True, separator="\n", fixed_label=None,
                   unlabel=None):
    """
    :param data_file_name:
    :param label_file_name:
    :param ratio:
    :param batch_size:
    :return: dataloader
    """
    # data = np.loadtxt(data_file_name, delimiter=separator)
    # label = np.loadtxt(label_file_name, delimiter=separator)
    data, label = set_MyDataset(root, data_file_name, separator, fixed_label, unlabel)
    print(f"len data {len(data)} len label {len(label)}")
    # exit()
    all_data = []
    all_label = []
    train_data = []
    train_label = []
    test_data = []
    test_label = []

    i = 0
    temp = list(zip(data, label))
    random.shuffle(temp)
    all_data, all_label = zip(*temp)

    train_data = all_data[:int(len(all_data) * ratio[0])]
    train_label = all_label[:int(len(all_data) * ratio[0])]
    test_data = all_data[int(len(all_data) * ratio[0]):]
    test_label = all_label[int(len(all_data) * ratio[0]):]

    train_data_tensor = torch.tensor(np.array(train_data), dtype=torch.float)
    train_label_tensor = torch.tensor(np.array(train_label), dtype=torch.int64)
    test_data_tensor = torch.tensor(np.array(test_data), dtype=torch.float)
    test_label_tensor = torch.tensor(np.array(test_label), dtype=torch.int64)

    train_dataset = TrainSet(train_data_tensor, train_label_tensor)
    test_dataset = TrainSet(test_data_tensor, test_label_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)

    return train_loader, test_loader
