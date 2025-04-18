from torch.utils.data import Dataset, DataLoader
import os
import json
import math
import torch
from torch.utils.data.sampler import RandomSampler, Sampler
from torch.utils.data.dataset import ConcatDataset


def load_json(fp):
    if not os.path.exists(fp):
        return dict()

    with open(fp, 'r', encoding='utf8') as f:
        return json.load(f)
"""
class PreDatasets(Dataset):
    def __init__(self, n_fp, an_fp):
        with open(n_fp, 'r') as f:
            self.n_data = f.readlines()
        with open(an_fp, 'r') as f:
            self.an_data = f.readlines()

    def __len__(self):
        return min(len(self.n_data), len(self.an_data))
    
    def deal_data(self, sent):
        sent = sent.split(':')
        label_list = sent[0].split(' ')
        src = []
        label = []
        for item in label_list: label.append(item)
        instance_list = sent[1].split(';')
        for instance in instance_list: src.append(instance.split(' '))
        return src, label          

    def __getitem__(self, index):
        n_sent = self.n_data[index].replace('\n', '')
        an_sent = self.an_data[index].replace('\n', '')
        n_src, n_label = self.deal_data(n_sent)
        an_src, an_label = self.deal_data(an_sent)
        return n_src, n_label, an_src, an_label        

class SingleDatasets(Dataset):
    def __init__(self, n_fp):
        with open(n_fp, 'r') as f:
            self.n_data = f.readlines()

    def __len__(self):
        return len(self.n_data)
    
    def deal_data(self, sent):
        sent = sent.split(':')
        label_list = sent[0].split(' ')
        src = []
        label = []
        for item in label_list: label.append(item)
        instance_list = sent[1].split(';')
        for instance in instance_list: src.append(instance.split(' '))
        return src, label          

    def __getitem__(self, index):
        n_sent = self.n_data[index].replace('\n', '')
        n_src, n_label = self.deal_data(n_sent)
        return n_src, n_label

def get_loader(n_fp, batch_size = 4, shuffle = True, num_workers = 0):
    #dataset = PreDatasets(n_fp, an_fp)
    dataset = SingleDatasets(n_fp)
    loader = DataLoader(dataset, batch_size= batch_size, shuffle = shuffle, num_workers= num_workers)
    return loader



class CustomDataset1(Dataset):
    def __init__(self, n_fp):
        with open(n_fp, 'r') as f:
            self.data1 = f.readlines()

    def __len__(self):
        return len(self.data1)

    def deal_data(self, sent):
        sent = sent.split(':')
        label_list = sent[0].split(' ')
        src = []
        label = []
        for item in label_list: label.append(item)
        instance_list = sent[1].split(';')
        for instance in instance_list: src.append(instance.split(' '))
        return src, label

    def __getitem__(self, index):
        return self.data1[index]
        n_sent = self.data1[index].replace('\n', '')
        n_src, n_label = self.deal_data(n_sent)
        return n_src, n_label
    
def combine_batches(batch1, batch2):
    # 在这里编写将两个批次拼接的逻辑
    # 假设 batch1 和 batch2 是由 DataLoader 返回的批次数据
    n_sent = batch1
    an_sent = batch2
    #combined_batch = torch.cat([batch1, batch2], dim=0)
    return n_sent, an_sent 

def combined_dataloader(dataloader1, dataloader2):
    # 创建一个自定义的数据加载函数，它会独立地加载两个数据集
    for batch1, batch2 in zip(dataloader1, dataloader2):
        yield combine_batches(batch1, batch2)


class MyDataset(Dataset):
    def __init__(self, generator, length):
        self.generator = generator
        self.len = length

    def __len__(self):
        # 请根据实际情况返回数据集的长度
        return self.len

    def deal_data(self, sent):
        sent = sent.split(':')
        label_list = sent[0].split(' ')
        src = []
        label = []
        for item in label_list: label.append(item)
        instance_list = sent[1].split(';')
        for instance in instance_list: src.append(instance.split(' '))
        return src, label

    def __getitem__(self, index):
        # 获取生成器的下一个值
        n_src_list = []
        an_src_list = []
        n_label_list = []
        an_label_list = []

        n_sent_list, an_sent_list = self.generator.__next__()
        for n_sent in n_sent_list:
            n_sent = n_sent.replace('\n', '')
            n_src, n_label = self.deal_data(n_sent)
            n_src_list.append(n_src)
            n_label_list.append(n_label)
        for an_sent in an_sent_list:
            an_sent = an_sent.replace('\n', '')
            an_src, an_label = self.deal_data(an_sent)
            an_src_list.append(an_src)
            an_label_list.append(an_label)
        return n_src_list, n_label_list, an_src_list, an_label_list
        #return self.generator.__next__()

def get_loader(n_fp, an_fp, batch_size = 4, shuffle = True, num_workers = 0):
    dataset1 = CustomDataset1(n_fp)
    dataset2 = CustomDataset1(an_fp)
    loader1 = DataLoader(dataset1, batch_size= batch_size, shuffle = shuffle, num_workers= num_workers)
    loader2 = DataLoader(dataset2, batch_size= batch_size, shuffle = shuffle, num_workers= num_workers)
    generator = combined_dataloader(loader1, loader2)
    #return generator
    my_dataset = MyDataset(generator, min(dataset1.__len__(), dataset2.__len__()))
    loader = DataLoader(my_dataset, batch_size= 1, shuffle = shuffle, num_workers= num_workers)
    return loader
"""
          


class CustomDataset1(Dataset):
    def __init__(self, n_fp):
        with open(n_fp, 'r') as f:
            self.data1 = f.readlines()

    def __len__(self):
        return len(self.data1)

    def slide_window(self, src, window_size):
        res_src = []
        res_label = []
        for i in range(len(src) - window_size):
            window = []
            for j in range(window_size): window.append(src[i + j])
            res_src.append(window)
            res_label.append(src[i + window_size])
        return res_src, res_label
    
    def slide_window_bidirectional(self, src, window_size):
        res_src = []
        res_label = []
        for i in range(len(src) - window_size):
            window = []
            bi_window = []
            for j in range(window_size): window.append(src[i + j])
            for j in range(window_size): bi_window.append(src[len(src) - i - j - 1])
            res_src.append(window)
            res_src.append(bi_window)
            res_label.append(src[i + window_size])
            res_label.append(src[len(src) - i - window_size - 1])
        return res_src, res_label

    def deal_data(self, sent):
        sent = sent.split(':')[1]
        src = sent.split(' ')
        src, label = self.slide_window(src, window_size = 10)
        return src, label


    def __getitem__(self, index):
        n_sent = self.data1[index].replace('\n', '')
        n_src, n_label = self.deal_data(n_sent)
        return n_src, n_label
   


class RCAdatasets(Dataset):
    def __init__(self, fp):
        with open(fp, 'r') as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sent = self.data[index].replace('\n', '')
        sent = sent.split(':')
        label = sent[0]
        src = sent[1].split(' ')
        rca = sent[-1].split(' ')
        return src, label, rca
       

class CustomSampler(Sampler):
    def __init__(self, dataset1, dataset2, batch_size):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.batch_size = batch_size
        self.num_samples = len(dataset1)

    def __iter__(self):
        indices1 = torch.randperm(len(self.dataset1)).tolist()
        indices2 = torch.randperm(len(self.dataset2)).tolist()

        for i in range(0, len(self.dataset1), self.batch_size):
            batch_indices1 = indices1[i:i + self.batch_size]
            batch_indices2 = indices2[i:i + self.batch_size]
            for i in range(len(batch_indices2)):
                batch_indices2[i] += len(self.dataset1)
            batch_indices = batch_indices1 + batch_indices2

            yield batch_indices

    def __len__(self):
        if self.num_samples // self.batch_size == self.num_samples / self.batch_size:
            return self.num_samples // self.batch_size
        else: return self.num_samples // self.batch_size + 1


def get_loader(n_fp, an_fp, batch_size = 4, shuffle = True, num_workers = 0):
    dataset1 = RCAdatasets(n_fp)
    dataset2 = RCAdatasets(an_fp)
    dataset = ConcatDataset([dataset1, dataset2])
    custom_sampler = CustomSampler(dataset1, dataset2, batch_size = batch_size)
    loader = DataLoader(dataset, batch_sampler = custom_sampler, num_workers= num_workers)
    return loader


def get_eval_loader(n_fp, batch_size = 4, shuffle = True, num_workers = 0):
    dataset = RCAdatasets(n_fp)
    loader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, num_workers= num_workers)
    return loader

