# python3
# Create Date: 2023-04-25
# Author: Scc_hy
# Func: 进行序列预测
# ==========================================================================================

import torch
from torch import nn
from torch.nn import functional as F
from argparse import Namespace
import math
from torch.autograd import Variable


cfg = Namespace(
    wheather_emb_dim=4,
    week_emb_dim=3,
    
    peak_hidden_dim=2,
    lstm_hidden_size=4,
    predict_len=2,
    
    device='cpu'
)



def qkv_attention(Q, K, V, mask=None):
    dim_k = K.size(-1)
    A = Q @ K.transpose(1,2) / math.sqrt(dim_k)
    if mask is not None:
        A = A.masked_fill(mask == 0, float("-inf"))
    A = F.softmax(A, dim=-1)
    return A @ V


# config = cfg

class peakPredict(nn.Module):
    def __init__(self, config):
        super(peakPredict, self).__init__()
        assert config.wheather_emb_dim == config.lstm_hidden_size, "wheather_emb_dim need equal to lstm_hidden_size"
        self.wheather_emb = nn.Embedding(6, config.wheather_emb_dim)
        self.work_rest_emb = nn.Embedding(3, config.wheather_emb_dim)
        
        self.index_emb = nn.Embedding(28, config.week_emb_dim)
        self.week_emb = nn.Embedding(10, config.week_emb_dim)
        
        self.peak_nn = nn.Linear(1, config.peak_hidden_dim)
        self.encoder = nn.LSTM(config.peak_hidden_dim + config.week_emb_dim, config.lstm_hidden_size, batch_first=True)
        self.decoder = nn.LSTM(config.lstm_hidden_size + config.lstm_hidden_size, config.lstm_hidden_size, batch_first=True)
        self.tgt_len = config.predict_len
        self.device = torch.device(config.device)
        self.attention_fc1 = nn.Linear(2 * config.lstm_hidden_size, 2 * config.lstm_hidden_size)
        self.attention_fc2 = nn.Linear(2 * config.lstm_hidden_size, 1)
        
        self.tgt_nn = nn.Linear(config.wheather_emb_dim + config.week_emb_dim, config.lstm_hidden_size)
        self.lstm_hidden_size = config.lstm_hidden_size
        
        self.head_pred = nn.Sequential(
            nn.Linear(self.lstm_hidden_size*3, self.tgt_len),
            nn.ReLU(),
            nn.Linear(self.tgt_len, 1)
        )
    
    def forward(self, peak_pt ,wheather_pt, work_rest_pt, index_pt, week_pt, 
                tgt_wheather_pt, tgt_work_rest_pt, tgt_index_pt, tgt_week_pt, train_flag=True):
        
        pred_res = []
        for i in range(peak_pt.size(0)):
            res_b = self.one_batch_forward(peak_pt[i, ...] ,wheather_pt[i, ...], work_rest_pt[i, ...], index_pt[i, ...], week_pt[i, ...], 
                tgt_wheather_pt[i, ...], tgt_work_rest_pt[i, ...], tgt_index_pt[i, ...], tgt_week_pt[i, ...], train_flag)
            pred_res.append(res_b)
            
        pred_res = torch.cat(pred_res).to(self.device)
        return pred_res

    def one_batch_forward(self, peak_pt ,wheather_pt, work_rest_pt, index_pt, week_pt, 
                tgt_wheather_pt, tgt_work_rest_pt, tgt_index_pt, tgt_week_pt, train_flag=True):
        wheather_rest_emb = self.wheather_emb(wheather_pt) + self.work_rest_emb(work_rest_pt)
        week_index_emb = self.index_emb(index_pt) + self.week_emb(week_pt)
        
        tgt_wheather_rest_emb = self.wheather_emb(tgt_wheather_pt) + self.work_rest_emb(tgt_work_rest_pt)
        tgt_week_index_emb = self.index_emb(tgt_index_pt) + self.week_emb(tgt_week_pt)
        tgt_data_embd = torch.cat([tgt_week_index_emb, tgt_wheather_rest_emb], dim=-1)
        # batch, len, lstm_hidden_size * len
        tgt_data_embd = self.tgt_nn(tgt_data_embd).unsqueeze(0)
        
        rnn_in = torch.cat([week_index_emb, self.peak_nn(peak_pt)], dim=-1).unsqueeze(0)
        print('rnn_in.shape=', rnn_in.shape)
        enc_out, (h, c) = self.encoder(rnn_in)
        print('wheather_rest_emb.shape=', wheather_rest_emb.shape)
        # weather_rest_emn self-attention
        enc_out = qkv_attention(enc_out, wheather_rest_emb.unsqueeze(0), wheather_rest_emb.unsqueeze(0))
        print('enc_out.shape=', enc_out.shape)
        dec_prev_hidden_tuple = (h, c)
        print('h.shape=', h.shape)
        self.atten_opt = Variable(torch.zeros(
            enc_out.shape[0], self.tgt_len, self.lstm_hidden_size
        )).to(self.device)
        # dec_opt  -> (batch, length, lstm_hidden_size)
        self.dec_opt = Variable(torch.zeros(
            enc_out.shape[0], self.tgt_len, self.lstm_hidden_size
        )).to(self.device)
        
        if train_flag:
            for i in range(self.tgt_len):
                ipt_emb = tgt_data_embd[:, i, :].unsqueeze(1)
                atten_opt, dec_opt, dec_hidden = self.attention_forward(ipt_emb, dec_prev_hidden_tuple, enc_out)
                self.atten_opt[:, i] = atten_opt.squeeze()
                self.dec_opt[:, i] = dec_opt.squeeze()
                dec_prev_hidden_tuple = dec_hidden
            # b, tgt_len, 
            pred_ipt = torch.cat([tgt_data_embd, self.atten_opt, self.dec_opt], dim=2)
            outs = self.head_pred(pred_ipt).reshape(-1, self.tgt_len)
        else:
            outs = []
            for i in range(self.tgt_len):
                atten_opt, dec_opt, dec_hidden = self.attention_forward(ipt_emb, dec_prev_hidden_tuple, enc_out)
                pred_ipt = torch.cat([ipt_emb, atten_opt, dec_opt], dim=2)
                pred = self.head_pred(pred_ipt).reshape(-1, self.tgt_len)
                outs.append(pred.squeeze().cpu().numpy())
                dec_prev_hidden_tuple = dec_hidden

            outs = torch.cat(outs).to(self.device)
        return outs
        
    def attention_forward(self, ipt_emb, decoder_prev_hidden, enc_output):
        # ipt_emb                   ->   (batch, 1, emb_size)
        # decoder_prev_hidden       ->  [(batch, lstm_hidden_size), (batch, lstm_hidden_size)]
        # enc_output                ->  (batch, length, lstm_hidden_size * 2)
        prev_dec_h = decoder_prev_hidden[0].repeat(1, enc_output.size(1), 1)
        # 这部分attention和一般的 A = QK 有点不一样
        # 一般 A = QK = (WkI_dec)T(WqI_enc) 
        # 这部分 A = F2(ReLu(F1( I_enc, I_dec )))
        # atten_ipt -> (batch, tgt_len, lstm_hidden_size * 3)
        atten_ipt = torch.cat([enc_output, prev_dec_h], dim=-1)
        # tten_w  ->  (batch, tgt_len,  1)
        atten_w = self.attention_fc2(F.relu(self.attention_fc1(atten_ipt)))
        atten_w = F.softmax(atten_w, dim=1)
        
        # O = AV 
        # 一般 V = WvI_enc
        # 这里的V 没有做线性变换
        # atten_opt ->  (batch, 1, lstm_hidden_size)
        atten_opt = torch.sum(atten_w * enc_output, dim=1).unsqueeze(1)
        # dec_lstm_ipt   ->  (batch, 1, emb_size + lstm_hidden_size * 2)
        dec_lstm_ipt = torch.cat([ipt_emb, atten_opt], dim=2)
        # dec_opt        ->  (batch, 1, lstm_hidden_size)
        # dec_h | dec_c  ->  (1, batch, lstm_hidden_size)
        dec_opt, (dec_h, dec_c) = self.decoder(dec_lstm_ipt)
        return atten_opt, dec_opt, (dec_h, dec_c) 


peak_model = peakPredict(cfg)
peak_pt = torch.rand(1, 4, 1)
wheather_pt = torch.randint(0, 3, (1, 4)).long()
work_rest_pt  = torch.randint(0, 2, (1, 4)).long()
index_pt = torch.arange(4).long().unsqueeze(0)
week_pt = torch.randint(0, 7, (1, 4)).long()
tgt_wheather_pt = torch.randint(0, 3, (1, 2)).long()
tgt_work_rest_pt = torch.randint(0, 2, (1, 2)).long()
tgt_index_pt = torch.arange(2).long().unsqueeze(0)
tgt_week_pt = torch.randint(0, 7, (1, 2)).long()
res = peak_model(peak_pt ,wheather_pt, work_rest_pt, index_pt, week_pt, 
                tgt_wheather_pt, tgt_work_rest_pt, tgt_index_pt, tgt_week_pt, train_flag=True)

print(res)
