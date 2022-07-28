import pickle as pkl
import random
import math

class BatchBucket():
    def __init__(self, max_h, max_w, max_l, max_img_size, max_batch_size,
                 feature_file, lable_file, shuffle=True):
        self._max_img_size = max_img_size
        self._max_batch_size = max_batch_size
        self._shuffle = shuffle

        with open(feature_file, 'rb') as fp:
            self._features = pkl.load(fp)
        with open(lable_file, 'rb') as fp:
            self._labels = pkl.load(fp)
        self._data_info_list = [(uid, fea.shape[1], fea.shape[2], len(self._labels[uid])) 
                             for uid, fea in self._features.items()]
        
        self._keys = self._calc_keys(max_h, max_w, max_l)

    def _calc_keys(self, max_h, max_w, max_l):
        # 计算出真实的最大值
        _, h_info, w_info, l_info = zip(*self._data_info_list)
        mh, mw, ml = max(h_info), max(w_info), max(l_info)
        max_h, max_w, max_l = min(max_h, mh), min(max_w, mw), min(max_l, ml)

        # 根据真实的最大值切分网络
        keys = []
        init_h = 100 if 100 < max_h else max_h
        init_w = 100 if 100 < max_w else max_w
        # init_l = 100 if 100 < max_l else max_l
        init_l = max_l
        #网格的切分间距
        h_step = 50
        w_step = 100
        l_step = 20
        h = init_h
        while h <= max_h:
            w = init_w
            while w <= max_w:
                l = init_l
                while l <= max_l:
                    keys.append([h, w, l, h * w * l, 0, []])
                    if l < max_l and l + l_step > max_l:
                        l = max_l
                    else:
                        l += l_step
                if w < max_w and w + max(int((w*0.3 // 10) * 10), w_step) > max_w:
                    w = max_w
                else:
                    w = w + max(int((w*0.3 // 10) * 10), w_step)
            if h < max_h and h + max(int((h*0.5 // 10) * 10), h_step) > max_h:
                h = max_h
            else:
                h = h + max(int((h*0.5 // 10) * 10), h_step)
        keys = sorted(keys, key=lambda area:area[3])

        # 把每个数据分配到想对应的网格中
        # 统计每个网格中落下的样本数量
        unused_num = 0
        for uid, h, w, l in self._data_info_list:
            flag = False
            for i, key in enumerate(keys):
                hh, ww, ll, _, _, subset = key
                if h <= hh and w <= ww and l <= ll:
                    keys[i][-2] += 1
                    subset.append(uid)
                    flag = True
                    break
            if flag == False:
                print(uid, h, w, l)
                unused_num += 1
        print(f'The number of all samples: {len(self._data_info_list)}')
        print(f'The number of unused samples: {unused_num}')
        # 过滤所有网格中少于某个阈值的网格
        keys = list(filter(lambda temp_k:temp_k[-2]>0, keys))
        # 计算每个子网格的batch大小
        # total_batch, total_sample = 0, 0
        for i, key in enumerate(keys):
            h, w, l, _, sample_num, _ = key
            batch_size = int(self._max_img_size / (h * w))
            batch_size = max(1, min(batch_size, self._max_batch_size))
            keys[i][3] = batch_size
            print(f'bucket [{h}, {w}, {l}], batch={batch_size}, sample={sample_num}')
        # 返回网格的key值
        return keys

    def _reset_batches(self):
        self._batches = []
        #打乱每个子网格中的样本顺序
        for _, _, _, batch_size, sample_num, uids in self._keys:
            if self._shuffle:
                random.shuffle(uids)
            batch_num = math.ceil(sample_num / batch_size)
            for i in range(batch_num):
                start = i * batch_size
                end = start + batch_size if start + batch_size < sample_num else sample_num
                self._batches.append(uids[start:end])
        if self._shuffle:
            random.shuffle(self._batches)

    def get_batches(self):
        batches = []
        self._reset_batches()
        for uid_batch in self._batches:
            fea_batch = [self._features[uid] for uid in uid_batch]
            label_batch = [self._labels[uid] for uid in uid_batch]
            batches.append((fea_batch, label_batch, uid_batch))
        print(f'The number of Bucket(subset) {len(self._keys)}')
        print(f'The number of Batches {len(batches)}')
        print(f'The number of Samples {len(self._data_info_list)}')

        return batches

if __name__ == '__main__':
    data_path       = 'data/'
    train_datasets = ['train_images.pkl', 'train_labels.pkl', 'train_relations.pkl']
    train_data_iterator = BatchBucket(600, 2100, 200, 800000, 16,
                                      data_path+train_datasets[0], data_path+train_datasets[1])
    train_batches = train_data_iterator.get_batches()