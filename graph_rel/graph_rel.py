import torch as T
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from gcn import GCN

NUM_POS = 10
ARCH = '2p'

class Model_GraphRel(nn.Module):
    def __init__(self, mxl, num_rel,hid_size=256, rnn_layer=2,gcn_layer=2,dp=0.5):
        self.mxl=mxl
        self.num_rel=num_rel
        self.hid_size = hid_size
        self.rnn_layer = rnn_layer
        self.gcn_layer=gcn_layer
        self.dp = dp
        
        #num pos needs to be defined
        self.emb_pos = nn.Embedding(NUM_POS,15)

        self.rnn= nn.GRU(300+15, self.hid_size, num_layers=self.rnn_layer,batch_first=True, dropout=dp, bidirectional=True)
        #meed GCN module
        self.gcn_fw = nn.ModuleList([GCN(self.hid_size*1) for _ in range(self.gcn_layer)])
        self.gcn_bw = nn.ModuleList([GCN(self.hid_size*1) for _ in range(self.gcn_layer)])

        self.rnn_ne = nn.GRU(self.hid_size*2,self.hid_size,batch_first=True)
        self.fc_ne = nn.Linear(self.hid_size,5)

        self.trs0_rel = nn.Linear(self.hid_size*2, self.hid_size)
        self.trs1_rel = nn.Linear(self.hid_size*2, self.hid_size)
        self.fc_rel = nn.Linear(self.hid_size*2,self.num_rel)

        if ARCH == '2p':
            self.gcn2p_fw = nn.ModuleList([GCN(self.hid_size*2) for _ in range(self.num_rel)])
            self.gcn2p_bw = nn.ModuleList([GCN(self.hid_size*2) for _ in range(self.num_rel)])
        self.dp = nn.Dropout(dp)
    
    def output(self,feat):
        out_ne,_=self.rnn_ne(feat)
        out_ne = self.dp(out_ne)
        out_ne = self.fc_ne(out_ne)

        trs0 = nn.functional.relu(self.trs0_rel(feat))
        trs0 = self.dp(trs0)
        trs1 = nn.functional.relu(self.trs1_rel(feat))
        trs1 = self.dp(trs1)
        
        trs0 = trs0.view((trs0.shape[0], trs0.shape[1], 1, trs0.shape[2]))
        trs0 = trs0.expand((trs0.shape[0], trs0.shape[1], trs0.shape[1], trs0.shape[3]))
        trs1 = trs1.view((trs1.shape[0], 1, trs1.shape[1], trs1.shape[2]))
        trs1 = trs1.expand((trs1.shape[0], trs1.shape[2], trs1.shape[2], trs1.shape[3]))
        trs = T.cat([trs0, trs1], dim=3)
        
        out_rel = self.fc_rel(trs)
        
        return out_ne, out_rel

    def forward(self, inp, pos, dep_fw, dep_bw):
        pos = self.emb_pos(pos)
        inp = T.cat([inp, pos], dim=2)
        inp = self.dp(inp)
        
        out, _ = self.rnn(inp)
        
        for i in range(self.gcn_layer):
            out_fw = self.gcn_fw[i](out, dep_fw)
            out_bw = self.gcn_bw[i](out, dep_bw)
            
            out = T.cat([out_fw, out_bw], dim=2)
            out = self.dp(out)
        
        feat_1p = out
        out_ne, out_rel = self.output(feat_1p)
        
        if ARCH=='1p':
            return out_ne, out_rel
        
        # 2p
        out_ne1, out_rel1 = out_ne, out_rel
        
        dep_fw = nn.functional.softmax(out_rel, dim=3)
        dep_bw = dep_fw.transpose(1, 2)
        
        outs = []
        for i in range(self.num_rel):
            out_fw = self.gcn2p_fw[i](feat_1p, dep_fw[:, :, :, i])
            out_bw = self.gcn2p_bw[i](feat_1p, dep_bw[:, :, :, i])
            
            outs.append(self.dp(T.cat([out_fw, out_bw], dim=2)))
        
        feat_2p = feat_1p
        for i in range(self.num_rel):
            feat_2p = feat_2p+outs[i]
        
        out_ne2, out_rel2 = self.output(feat_2p)
        
        return out_ne1, out_rel1, out_ne2, out_rel2

