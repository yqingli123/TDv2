import time
import numpy as np 
import pickle as pkl 
import torch
from torch import optim, nn
from utils.utils import load_dict, prepare_data, weight_init, cmp_result
from model.encoder_decoder import Encoder_Decoder
from utils.data_iterator import BatchBucket
from utils.latex2gtd_v2_2 import list2node, tree2latex, relation2gtd

# configurations
init_param_flag = True  # whether init params
reload_flag     = False # whether relaod params
data_path       = 'data/'
work_path       = 'results/'
dictionaries = ['dictionary_object.txt', 'dictionary_relation_noend.txt']
train_datasets = ['train_images.pkl', 'train_labels.pkl', 'train_relations.pkl']
valid_datasets = ['valid_images.pkl', 'valid_labels.pkl', 'valid_relations.pkl']
valid_outputs = ['valid_results.txt']
model_params = ['SimTree_best_params.pkl', 'SimTree_last_params.pkl']

#train Settings
maxlen = 200
max_epochs = 5000
lrate = 1
my_eps = 1e-6
decay_c = 1e-4
clip_c = 100.

# early stop
estop = False
halfLrFlag = 0
bad_counter = 0
patience = 15
validStart = 0 # 模型未学习好，解码容易溢出，浪费时间
finish_after = 10000000

# model architecture 
# 这部分应该转移到模型代码中
params = {}
params['n'] = 256
params['m'] = 256
params['re_m'] = 64
params['dim_attention'] = 512
params['D'] = 936
params['K'] = 107
params['Kre'] = 9
params['mre'] = 256
params['maxlen'] = maxlen

params['growthRate'] = 24
params['reduction'] = 0.5
params['bottleneck'] = True
params['use_dropout'] = True
params['input_channels'] = 1

params['lc_lambda'] = 1.
params['lr_lambda'] = 1.
params['lc_lambda_pix'] = 1.

symbol2id = load_dict(data_path+dictionaries[0])
print('total chars', len(symbol2id))
id2symbol = {}
for symbol, symbol_id in symbol2id.items():
    id2symbol[symbol_id] = symbol

relation2id = load_dict(data_path+dictionaries[1])
print('total relations', len(relation2id))
id2relation =  {}
for relation, relation_id in relation2id.items():
    id2relation[relation_id] = relation

train_data_iterator = BatchBucket(600, 2100, 200, 800000, 16,
    data_path+train_datasets[0], data_path+train_datasets[1])
valid_data_iterator = BatchBucket(9999, 9999, 9999, 999999, 1,
    data_path+valid_datasets[0], data_path+valid_datasets[1])
valid = valid_data_iterator.get_batches()
with open(data_path + valid_datasets[1], 'rb') as fp:
    valid_gtds = pkl.load(fp)

# display
uidx = 0
object_loss_s = 0.
relation_loss_s = 0. 
loss_s = 0. 

ud_s = 0
validFreq = -1
saveFreq = -1
sampleFreq = -1
dispFreq = 100
WER = 100

# inititalize model
SimTree_model = Encoder_Decoder(params)
if init_param_flag:
    SimTree_model.apply(weight_init)
if reload_flag:
    print('Loading pretrained model ...')
    SimTree_model.load_state_dict(torch.load(work_path+model_params[1], map_location=lambda storage, loc:storage))
SimTree_model.cuda()

optimizer = optim.Adadelta(SimTree_model.parameters(), lr=lrate, eps=my_eps, weight_decay=decay_c)
print('Optimization')

# statistics
history_errs = []
for eidx in range(max_epochs):
    n_samples = 0
    ud_epoch = time.time()
    train = train_data_iterator.get_batches()
    if validFreq == -1:
        validFreq = len(train)
    if saveFreq == -1:
        saveFreq = len(train)
    if sampleFreq == -1:
        sampleFreq = len(train)
    
    for x, y, key in train:
        SimTree_model.train()
        ud_start = time.time()
        n_samples += len(x)
        uidx += 1
        x, x_mask, C_y, y_mask, P_y, P_re, C_re, lp, rp = \
            prepare_data(params, x, y, key, symbol2id, relation2id, shuffle=False)
        
        length  = C_y.shape[0]
        x       = torch.from_numpy(x).cuda()
        x_mask  = torch.from_numpy(x_mask).cuda()
        C_y     = torch.from_numpy(C_y).to(torch.long).cuda()
        y_mask  = torch.from_numpy(y_mask).cuda()
        P_y     = torch.from_numpy(P_y).to(torch.long).cuda()
        P_re    = torch.from_numpy(P_re).to(torch.long).cuda()
        P_position = torch.from_numpy(rp).to(torch.long).cuda()
        C_re    = torch.from_numpy(C_re).cuda()

        loss, object_loss, relation_loss = SimTree_model(params, x, x_mask,
            C_y, P_y, C_re, P_re, P_position, y_mask, y_mask, length)

        object_loss_s += object_loss.item()
        relation_loss_s += relation_loss.item()
        loss_s = loss.item()

        # backward
        optimizer.zero_grad()
        loss.backward()
        if clip_c > 0.:
            torch.nn.utils.clip_grad_norm_(SimTree_model.parameters(), clip_c)
        
        # updata
        optimizer.step()

        # display
        ud = time.time() - ud_start
        ud_s += ud

        if np.mod(uidx, dispFreq) == 0:
            ud_s /= 60. 
            loss_s /= dispFreq
            object_loss_s /= dispFreq
            relation_loss_s /= dispFreq
            print(f'Epoch {eidx}, Update {uidx} Cost_object {object_loss_s:.7}', end='')
            print(f'Cost_relation {relation_loss_s}, UD {ud_s:.3} lrate {lrate} eps {my_eps} bad_counter {bad_counter}')
            ud_s = 0
            loss_s = 0.
            object_loss_s = 0. 
            relation_loss_s = 0. 

        if np.mod(uidx, saveFreq) == 0:
            print('Saving latest model params ... ')
            torch.save(SimTree_model.state_dict(), work_path+model_params[1])

        # validation
        if np.mod(uidx, sampleFreq) == 0 and (eidx % 2) == 0:
            number_right = 0
            total_distance = 0
            total_length = 0
            latex_right = 0
            total_latex_distance = 0
            total_latex_length = 0
            total_number = 0

            print('begin sampling')
            ud_epoch_train = (time.time() - ud_epoch) / 60.
            print('epoch training cost time ...', ud_epoch_train)
            SimTree_model.eval()

            fp_results = open(work_path+valid_outputs[0], 'w') 
            with torch.no_grad():
                valid_count_idx = 0
                for x, y, valid_key in valid:
                    x, x_mask, C_y, y_mask, P_y, P_re, C_re, lp, rp = \
                        prepare_data(params, x, y, valid_key, symbol2id, relation2id)
                    
                    L, B = C_y.shape[:2]
                    x = torch.from_numpy(x).cuda()
                    x_mask = torch.from_numpy(x_mask).cuda()
                    lengths_gt = (y_mask > 0.5).sum(0)
                    y_mask = torch.from_numpy(y_mask).cuda()
                    P_y = torch.from_numpy(P_y).to(torch.long).cuda()
                    P_re = torch.from_numpy(P_re).to(torch.long).cuda()

                    object_predicts, P_masks, relation_table_static, _ \
                        = SimTree_model.greedy_inference(x, x_mask, L+1, P_y[0], P_re[0], y_mask[0])
                    object_predicts, P_masks = object_predicts.cpu().numpy(), P_masks.cpu().numpy()
                    relation_table_static = relation_table_static.numpy()
                    for bi in range(B):
                        length_predict = min((P_masks[bi, :] > 0.5).sum(), P_masks.shape[1])
                        object_predict = object_predicts[:int(length_predict), bi]
                        relation_predict = relation_table_static[bi, :int(length_predict), :]
                        gtd = relation2gtd(object_predict, relation_predict, id2symbol, id2relation)
                        latex = tree2latex(list2node(gtd))

                        uid = valid_key[bi]
                        groud_truth_gtd = valid_gtds[uid]
                        groud_truth_latex = tree2latex(list2node(groud_truth_gtd))

                        child = [symbol2id[g[0]] for g in groud_truth_gtd if g[0] != '</s>']
                        distance, length = cmp_result(object_predict, child)
                        total_number += 1

                        if distance == 0:
                            number_right += 1
                            fp_results.write(uid + '\tObject True\t')
                        else:
                            fp_results.write(uid + '\tObject False\t')

                        latex_distance, latex_length = cmp_result(groud_truth_latex, latex)
                        if latex_distance == 0:
                            latex_right += 1
                            fp_results.write('Latex True\n')
                        else:
                            fp_results.write('Latex False\n')

                        total_distance += distance
                        total_length += length
                        total_latex_distance += latex_distance
                        total_latex_length  += latex_length

                        fp_results.write(' '.join(groud_truth_latex) + '\n')
                        fp_results.write(' '.join(latex) + '\n')
                        
                        for c in child:
                            fp_results.write(id2symbol[c] + ' ')
                        fp_results.write('\n')

                        for ob_p in object_predict:
                            fp_results.write(id2symbol[ob_p] + ' ')
                        fp_results.write('\n')

            wer = total_distance / total_length * 100
            sacc = number_right / total_number * 100
            latex_wer = total_latex_distance / total_latex_length * 100
            latex_acc = latex_right / total_number * 100
            fp_results.close()

            ud_epoch = (time.time() - ud_epoch) / 60.
            print(f'valid set decode done, epoch cost time: {ud_epoch} min')
            print(f'WER {wer} SACC {sacc} Latex WER {latex_wer} Latex SACC {latex_acc}')

            if latex_wer <= WER:
                WER = latex_wer
                bad_counter = 0
                print('Saving best model params ... ')
                torch.save(SimTree_model.state_dict(), work_path+model_params[0])
            else:
                bad_counter += 1
                if bad_counter > patience:
                    if halfLrFlag == 2:
                        print('Early Stop!')
                        estop = True
                        break
                    else:
                        print('Lr decay and retrain!')
                        bad_counter = 0
                        lrate = lrate / 10.
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lrate
                        halfLrFlag += 1
        if uidx >= finish_after:
            print(f'Finishing after {uidx} iterations!')
            estop = True
            break

    print(f'Seen {n_samples} samples')

    if estop:
        break