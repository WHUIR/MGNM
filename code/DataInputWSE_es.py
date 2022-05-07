import numpy as np
import random
import pickle
from elasticUtil import ESClient
from tqdm import tqdm

random.seed(123)

def gen_se_pool(data):
    u2i = {}
    for t in tqdm(data):
        if t[0] in u2i:
            if len(u2i[t[0]]) < len(t[1]):
                u2i[t[0]] = list(t[1])
        else:
            u2i[t[0]] = list(t[1])
    print('filtering...')
    tmp = [li for li in list(u2i.values()) if len(li) > 1]
    tmp_u = [list(u2i.keys())[idx] for idx, ii in enumerate(list(u2i.values())) if len(ii) > 1]
    print('filter finished...')
    return tmp_u, tmp


class DataInput:
    def __init__(self, data, batch_size, max_len, neg_num, item_num, train_pool, se_num=5, init_es=False):

        self.batch_size = batch_size
        self.data = data  # shuffle数据
        self.neg_num = neg_num
        self.max_len = max_len
        self.item_num = item_num
        self.epoch_size = len(self.data) // self.batch_size - 1
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0
        self.se_num = se_num
        self.seclient = ESClient()
        if init_es:
            se_user, se_pool = gen_se_pool(train_pool)
            self.seclient.add_data(se_pool, se_user)

    def __iter__(self):
        return self

    def __next__(self):
        if self.i == self.epoch_size:
            raise StopIteration

        ts = self.data[self.i * self.batch_size: min((self.i + 1) * self.batch_size,
                                                     len(self.data))]

        while len(ts) < self.batch_size:
            print('__next__ padding...')
            tmp = ts[: self.batch_size - len(ts)]
            ts.extend(tmp)

        self.i += 1

        u, i, y, sl = [], [], [], []
        for t in ts:
            u.append(t[0])
            i.append(t[2])
            y.append(int(t[3]))
        max_sl = self.max_len

        hist_i = list(np.zeros([len(ts), max_sl+self.se_num], np.int32))
        mask_i = list(np.zeros([len(ts), max_sl+self.se_num], np.float32))
        k = 0
        for t in ts:
            for l in range(len(t[1][-max_sl:])):
                l += 1
                hist_i[k][-l] = t[1][-l]
                mask_i[k][-l] = 1.
                # query
                if self.se_num != 0:
                    se_item_list = list(self.seclient.search(t[1][-max_sl:]) - set(t[1]))
                    for idx in range(min(self.se_num, len(se_item_list))):
                        hist_i[k][idx] = se_item_list[idx]
                        mask_i[k][idx] = 1.
            # query = t[1]
            # self.se.top1(t[1], 5)
            # rel_his = self.se.list
            # tmp_se_set = set()
            # for lis in rel_his:
            #     tmp_se_set.update(self.se_pool[lis])
            # tmp_se_set = list(tmp_se_set - set(t[1]))
            # for idx in range(min(self.se_num, len(tmp_se_set))):
            #     se_items[k][idx] = tmp_se_set[idx]
            #     se_mask[k][idx] = 1.
            k += 1

        if self.neg_num:
            length = len(u)
            for idx in range(length):
                tmp = self.neg_num
                # """
                while tmp > 0:
                    neg_item = random.randint(1, self.item_num-2)
                    if neg_item in hist_i[idx] or neg_item == i[idx]:
                        continue
                    u.append(u[idx])
                    i.append(neg_item)
                    y.append(0)
                    hist_i.append(hist_i[idx])
                    mask_i.append(mask_i[idx])
                    tmp -= 1
        return u, i, y, hist_i, mask_i


if __name__ == '__main__':
    test_set = pickle.load(open('./clear_data/Musical_Instruments_train_set.pkl', 'rb'))
    item2id = pickle.load(open('./clear_data/Musical_Instruments_item2id.pkl', 'rb'))
    user2id = pickle.load(open('./clear_data/Musical_Instruments_user2id.pkl', 'rb'))
    data_input = DataInput(test_set, 64, 20, 4, len(item2id.keys()), train_pool=test_set)

    for idx, data in enumerate(data_input):
        print(data)
