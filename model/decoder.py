import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.object_embedding = nn.Embedding(params['K'], params['m'])
        self.relation_embedding = nn.Embedding(params['Kre'], params['re_m'])
        
        self.gru00 = nn.GRUCell(params['m']+params['re_m'], params['n'])
        self.gru01 = nn.GRUCell(params['D'], params['n'])
        self.gru10 = nn.GRUCell(params['D'], params['n'])
        self.gru11 = nn.GRUCell(params['D'], params['n'])

        self.conv_key_object = nn.Conv2d(params['D'], params['dim_attention'], kernel_size=1)
        self.object_attention = CoverageAttention(params)
        self.object_probility = ObjectProbility(params)

        self.conv_key_relation = nn.Conv2d(params['D'], params['dim_attention'], kernel_size=1)
        self.relation_attention = Attention(params)
        self.relation_probility = RelationProbility(params)

        self.pix_object_probility = PixObjectProbility(params)

        self.param_K = params['K']
        self.param_Kre = params['Kre']
        self.param_n = params['n']
        
    def forward(self, ctx_val, ctx_mask, C_ys, P_ys, P_y_masks, 
                P_res, P_positions, init_state, length):

        B, H, W = ctx_mask.shape
        attention_past = torch.zeros(B, 1, H, W).cuda()
        predict_childs = torch.zeros(length, B, self.param_K).cuda()
        predict_childs_pix = torch.zeros(length, B, self.param_K).cuda()
        predict_relations = torch.zeros(length, B, self.param_Kre).cuda()
        
        ht = init_state
        ht_relation = init_state

        ctx_key_object   = self.conv_key_object(ctx_val).permute(0, 2, 3, 1)
        ctx_key_relation = self.conv_key_relation(ctx_val).permute(0, 2, 3, 1)

        predict_features = self.pix_object_probility(ctx_val)
        for i in range(length):
            predict_childs[i], ht, attention, ct = self.get_child(ctx_val, ctx_key_object, ctx_mask,
                            attention_past, P_ys[i], P_y_masks[i], P_res[i], ht)
            attention_past = attention[:, None, :, :] + attention_past

            predict_childs_pix[i] = torch.log((attention.detach()[:, :, :, None] * predict_features).sum(2).sum(1))
            predict_relations[i], ht_relation = self.get_relation(ctx_val, ctx_key_relation,
                ctx_mask, C_ys[i], ht_relation, ct)
        
        return predict_childs, predict_relations, predict_childs_pix

    def get_child(self, ctx_val, ctx_key, ctx_mask, attention_past, p_y, p_y_mask, p_re, ht):
        p_y = self.object_embedding(p_y)
        p_re = self.relation_embedding(p_re)
        p = torch.cat([p_y, p_re], dim=1)
        ht_hat = self.gru00(p, ht)
        ht_hat = p_y_mask[:, None] * ht_hat + (1 - p_y_mask)[:, None] * ht

        ct, attention = self.object_attention(ctx_val, ctx_key, ctx_mask, attention_past, ht_hat)

        ht = self.gru01(ct, ht_hat)
        ht = p_y_mask[:, None] * ht + (1 - p_y_mask)[:, None] * ht_hat

        predict_child = self.object_probility(ct, ht, p_y, p_re)

        return predict_child, ht, attention, ct

    def get_relation(self, ctx_val, ctx_key, ctx_mask, c_y, ht, ct):
        c_y = self.object_embedding(c_y)
        ht_query = self.gru10(ct, ht)
        ct, _ = self.relation_attention(ctx_val, ctx_key, ctx_mask, ht_query)
        ht = self.gru11(ct, ht_query)

        predict_relation = self.relation_probility(ct, c_y, ht)
        return predict_relation, ht
    
class CoverageAttention(nn.Module):
    def __init__(self, params):
        super(CoverageAttention, self).__init__()
        self.fc_query = nn.Linear(params['n'], params['dim_attention'], bias=False)
        self.conv_att_past = nn.Conv2d(1, 512, kernel_size=11, bias=False, padding=5)
        self.fc_att_past = nn.Linear(512, params['dim_attention'])
        self.fc_attention = nn.Linear(params['dim_attention'], 1)
    
    def forward(self, ctx_val, ctx_key, ctx_mask, attention_past, ht_query):

        ht_query = self.fc_query(ht_query)

        attention_past = self.conv_att_past(attention_past).permute(0, 2, 3, 1)
        attention_past = self.fc_att_past(attention_past) #(batch, H, W, dim_att)

        attention_score = torch.tanh(ctx_key + ht_query[:, None, None, :] + attention_past)
        attention_score = self.fc_attention(attention_score).squeeze(3)

        attention_score = attention_score - attention_score.max()
        attention_score = torch.exp(attention_score) * ctx_mask
        attention_score = attention_score / (attention_score.sum(2).sum(1)[:, None, None] + 1e-10)

        ct = (ctx_val * attention_score[:, None, :, :]).sum(3).sum(2)

        return ct, attention_score

class Attention(nn.Module):
    def __init__(self, params):
        super(Attention, self).__init__()
        self.fc_query = nn.Linear(params['n'], params['dim_attention'], bias=False)
        self.fc_attention = nn.Linear(params['dim_attention'], 1)
    
    def forward(self, ctx_val, ctx_key, ctx_mask, ht_query):

        ht_query = self.fc_query(ht_query)

        attention_score = torch.tanh(ctx_key + ht_query[:, None, None, :])
        attention_score = self.fc_attention(attention_score).squeeze(3)

        attention_score = attention_score - attention_score.max()
        attention_score = torch.exp(attention_score) * ctx_mask
        attention_score = attention_score / (attention_score.sum(2).sum(1)[:, None, None] + 1e-10)

        ct = (ctx_val * attention_score[:, None, :, :]).sum(3).sum(2)

        return ct, attention_score

class ObjectProbility(nn.Module):
    def __init__(self, params):
        super(ObjectProbility, self).__init__()
        self.fc_ct = nn.Linear(params['D'], params['m'])
        self.fc_ht = nn.Linear(params['n'], params['m'])
        self.fc_p_y = nn.Linear(params['m'], params['m'])
        self.fc_p_re = nn.Linear(params['re_m'], params['m'])
        self.dropout = nn.Dropout(p=0.2)
        self.fc_probility = nn.Linear(int(params['m']/2), params['K'])

    def forward(self, ct, ht, p_y, p_re):
        out = self.fc_ct(ct) + self.fc_ht(ht) + self.fc_p_y(p_y) + self.fc_p_re(p_re)
        #maxout
        out = out.view(out.shape[0], -1, 2)
        out = out.max(2)[0]
        out = self.dropout(out)
        out = self.fc_probility(out)
        return out

class PixObjectProbility(nn.Module):
    def __init__(self, params):
        super(ObjectProbility, self).__init__()
        self.fc_ct = nn.Linear(params['D'], params['m'])
        self.dropout = nn.Dropout(p=0.2)
        self.fc_probility = nn.Linear(int(params['m']/2), params['K'])

    def forward(self, ctx):
        out = self.fc_ct(ctx.permute(0, 2, 3, 1)) # (B, H, W, D) --> (B, H, W, m)
        #maxout
        out = out.view(out.shape[0], out.shape[1], out.shape[2], -1, 2) #(B, H, W, m/2, 2)
        out = out.max(4)[0]     #(B, H, W, m/2)
        out = self.dropout(out)
        out = self.fc_probility(out) #(B, H, W, k)
        out = F.softmax(out, dim=3)
        return out

class RelationProbility(nn.Module):
    def __init__(self, params):
        super(RelationProbility, self).__init__()
        self.fc_ct = nn.Linear(params['D'], params['mre'])
        self.fc_c_y = nn.Linear(params['m'], params['mre'])
        self.fc_ht = nn.Linear(params['n'], params['mre'])
        self.dropout = nn.Dropout(p=0.2)
        self.fc_probility = nn.Linear(int(params['mre']/2), params['Kre'])

    def forward(self, ct, c_y, ht):
        out = self.fc_ct(ct) + self.fc_c_y(c_y) + self.fc_ht(ht)
        #maxout
        out = out.view(out.shape[0], -1, 2)
        out = out.max(2)[0]
        out = self.dropout(out)
        out = self.fc_probility(out)

        return out
