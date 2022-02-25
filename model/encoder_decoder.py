import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import DenseNet
from .decoder import Decoder

class Encoder_Decoder(nn.Module):
    def __init__(self, params):
        super(Encoder_Decoder, self).__init__()
        self.encoder = DenseNet(growthRate=params['growthRate'],
                                reduction=params['reduction'],
                                bottleneck=params['bottleneck'],
                                use_dropout=params['use_dropout'])
        self.init_context = nn.Linear(params['D'], params['n'])
        self.decoder = Decoder(params)
        self.object_criterion = nn.CrossEntropyLoss(reduction='none')
        self.relation_criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.object_pix_criterion = nn.NLLLoss(reduction='none')
        self.param_n = params['n']
    

    def forward(self, params, x, x_mask, C_y,
                P_y, C_re, P_re, P_position, y_mask, re_mask, length):

        #encoder
        ctx, ctx_mask = self.encoder(x, x_mask)
        ctx_mean = (ctx * ctx_mask[:, None, :, :]).sum(3).sum(2) \
                    / ctx_mask.sum(2).sum(1)[:, None]
        init_state = torch.tanh(self.init_context(ctx_mean))

        #decoder
        predict_objects, predict_relations,predict_objects_pix = self.decoder(ctx, ctx_mask,
            C_y, P_y, y_mask, P_re, P_position, init_state, length)

        #loss
        predict_objects = predict_objects.view(-1, predict_objects.shape[2])
        object_loss = self.object_criterion(predict_objects, C_y.view(-1))
        object_loss = object_loss.view(C_y.shape[0], C_y.shape[1])
        object_loss = ((object_loss * y_mask).sum(0) / y_mask.sum(0)).mean()

        predict_objects_pix = predict_objects_pix.view(-1, predict_objects_pix.shape[2])
        object_pix_loss = self.object_pix_criterion(predict_objects_pix, C_y.view(-1))
        object_pix_loss = object_pix_loss.view(C_y.shape[0], C_y.shape[1])
        object_pix_loss = ((object_pix_loss * y_mask).sum(0) / y_mask.sum(0)).mean()

        relation_loss = predict_relations.view(-1, predict_relations.shape[2])
        relation_loss = self.relation_criterion(relation_loss, C_re.view(-1, C_re.shape[2]))
        relation_loss = relation_loss.view(C_re.shape[0], C_re.shape[1], C_re.shape[2])
        relation_loss = (relation_loss * re_mask[:, :, None]).sum(2).sum(0) / re_mask.sum(0)
        relation_loss = relation_loss.mean()
        
        loss = params['lc_lambda'] * object_loss + \
               params['lr_lambda'] * relation_loss + \
               params['lc_lambda_pix'] * object_pix_loss

        return loss, object_loss, relation_loss

    def greedy_inference(self, x, x_mask, max_length, p_y, p_re, p_mask):

        ctx, ctx_mask = self.encoder(x, x_mask)
        ctx_mean = (ctx * ctx_mask[:, None, :, :]).sum(3).sum(2) \
                   / ctx_mask.sum(2).sum(1)[:, None]
        init_state = torch.tanh(self.init_context(ctx_mean))

        B, H, W = ctx_mask.shape
        attention_past = torch.zeros(B, 1, H, W).cuda()

        ctx_key_object = self.decoder.conv_key_object(ctx).permute(0, 2, 3, 1)
        ctx_key_relation = self.decoder.conv_key_relation(ctx).permute(0, 2, 3, 1)

        relation_table = torch.zeros(B, max_length, 9).to(torch.long)
        relation_table_static = torch.zeros(B, max_length, 9).to(torch.long)
        predict_relation_static = torch.zeros(B, max_length, 9)
        P_masks = torch.zeros(B, max_length).cuda()
        predict_childs       = torch.zeros(max_length, B).to(torch.long).cuda()
      
        ht = init_state
        parent_ht = init_state
        for i in range(max_length):
            predict_child, ht, attention, ct = self.decoder.get_child(ctx, ctx_key_object, ctx_mask, 
            attention_past, p_y, p_mask, p_re, ht)
            predict_childs[i]  = torch.argmax(predict_child, dim=1)
            predict_childs[i] *= p_mask.to(torch.long)
            attention_past = attention[:, None, :, :] + attention_past

            predict_relation, ht_relation = self.decoder.get_relation(ctx, ctx_key_relation, ctx_mask,
                predict_childs[i], ht_relation, ct)
            
            P_masks[:, i] = p_mask
            
            predict_relation_static[:, i, :] = predict_relation
            relation_table[:, i, :] = (predict_relation > 0)
            relation_table_static[:, i, :] = (predict_relation > 0)

            relation_table[:, :, 8] = relation_table[:, :, :8].sum(2)

            find_number = 0
            for ii in range(B):
                if p_mask[ii] < 0.5:
                    continue
                ji = i
                find_flag = 0
                while ji >= 0:
                    if relation_table[ii, ji, 8] > 0:
                        for iii in range(9):
                            if relation_table[ii, ji, iii] != 0:
                                p_re[ii] = iii
                                p_y[ii] = predict_childs[ji, ii]
                                relation_table[ii, ji, iii] = 0
                                relation_table[ii, ji, 8] -= 1
                                find_flag = 1
                                break
                        if find_flag:
                            break
                    ji -= 1
                find_number += find_flag
                if not find_flag:
                    p_mask[ii] = 0.

            if find_number == 0:
                break
        return predict_childs, P_masks, relation_table_static, predict_relation_static

