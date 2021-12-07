import json
import torch as T
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
from tqdm import tqdm_notebook as tqdm

from dataset import DS
from gcn import GCN
from graph_rel import Model_GraphRel


tr = json.loads(open("nyttest.json","r").read())
NUM_REL = 0
MXL = 0
ARCH = '2p'
DATASET = 'nyt'
for d in tr:
    sent = d["sentText"]
    MXL = max(len(sent)+4, MXL)
    
    rels = d["relationMentions"]
    NUM_REL = max(len(rels),NUM_REL)

NUM_REL = NUM_REL + 1 # 0 for NA
print('Num of relation: %d' % (NUM_REL))
print('Max length: %d' % (MXL))


import spacy

ld_tr = DataLoader(DS(tr), batch_size=1)

#ld_tr = DataLoader(DS(tr), batch_size=32, shuffle=True)

for x in ld_tr:
    print(x)
    break
gcn = GCN().cuda()
print(gcn)



model = nn.DataParallel(Model_GraphRel(mxl=MXL, num_rel=NUM_REL)).cuda()

if ARCH=='1p':
    out_ne, out_rel = model(inp.cuda(), pos.cuda(), dep_fw.cuda(), dep_bw.cuda())
    print(out_ne1.shape, out_rel1.shape)
    
else:
    out_ne1, out_rel1, out_ne2, out_rel2 = model(inp.cuda(), pos.cuda(), dep_fw.cuda(), dep_bw.cuda())
    print(out_ne1.shape, out_rel1.shape)
def post_proc(dat, idx, out_ne, out_rel):    
    out_ne = np.argmax(out_ne.detach().cpu().numpy(), axis=1)
    out_rel = np.argmax(out_rel.detach().cpu().numpy(), axis=2)
    
    nes = dict()
    el = -1
    for i, v in enumerate(out_ne):
        if v==4:
            nes[i] = [i, i]
            el = -1
            
        elif v==1:
            el = i
            
        elif v==3:
            if not el==-1:
                for p in range(el, i+1):
                    nes[p] = [el, i]
                
        elif v==2:
            pass
        
        elif v==0:
            el = -1
    
    rels = []
    for i in range(MXL):
        for j in range(MXL):
            if not out_rel[i][j]==0 and i in nes and j in nes:
                rels.append([nes[i][1], nes[j][1], out_rel[i][j]])
    
    cl = []
    for rel in rels:
        if not rel in cl:
            cl.append(rel)
    rels = cl
    
    ans = []
    for tmp in dat[idx][1]:
        ans.append([tmp[0][1], tmp[1][1], tmp[2]])
    
    cl = []
    for rel in ans:
        if not rel in cl:
            cl.append(rel)
    ans = cl
    
    return rels, ans

class F1:
    def __init__(self):
        self.P = [0, 0]
        self.R = [0, 0]
    
    def get(self):
        try:
            P = self.P[0]/self.P[1]
        except:
            P = 0
        
        try:
            R = self.R[0]/self.R[1]
        except:
            R = 0
            
        try: 
            F = 2*P*R/(P+R)
        except:
            F = 0
        
        return P, R, F
    
    def add(self, ro, ra):
        self.P[1] += len(ro)
        self.R[1] += len(ra)
        
        for r in ro:
            if r in ra:
                self.P[0] += 1
        
        for r in ra:
            if r in ro:
                self.R[0] += 1
EPOCHS = 200
LR = 0.0008
DECAY = 0.98

W_NE = 2
W_REL = 2
ALP = 3

loss_func = nn.CrossEntropyLoss(reduction='none').cuda()
optim = T.optim.Adam(model.parameters(), lr=LR)

def ls(out_ne, wgt_ne, out_rel, wgt_rel):
    ls_ne = loss_func(out_ne.view((-1, 5)), ans_ne.view((-1, )).cuda()).view(ans_ne.shape)
    ls_ne = (ls_ne*wgt_ne.cuda()).sum() / (wgt_ne>0).sum().cuda()
    
    ls_rel = loss_func(out_rel.view((-1, NUM_REL)), ans_rel.view((-1, )).cuda()).view(ans_rel.shape)
    ls_rel = (ls_rel*wgt_rel.cuda()).sum() / (wgt_rel>0).sum().cuda()
    
    return ls_ne, ls_rel

for e in tqdm(range(EPOCHS)):
    ls_ep_ne1, ls_ep_rel1 = 0, 0
    ls_ep_ne2, ls_ep_rel2 = 0, 0
    
    model.train()
    with tqdm(ld_tr) as TQ:
        for i, (idx, inp, pos, dep_fw, dep_bw, ans_ne, wgt_ne, ans_rel, wgt_rel) in enumerate(TQ):
            
            wgt_ne.masked_fill_(wgt_ne==1, W_NE)
            wgt_ne.masked_fill_(wgt_ne==0, 1)
            wgt_ne.masked_fill_(wgt_ne==-1, 0)
            
            wgt_rel.masked_fill_(wgt_rel==1, W_REL)
            wgt_rel.masked_fill_(wgt_rel==0, 1)
            wgt_rel.masked_fill_(wgt_rel==-1, 0)
            
            out_ne1, out_rel1, out_ne2, out_rel2 = model(inp.cuda(), pos.cuda(), dep_fw.cuda(), dep_bw.cuda())
            
            ls_ne1, ls_rel1 = ls(out_ne1, wgt_ne, out_rel1, wgt_rel)
            ls_ne2, ls_rel2 = ls(out_ne2, wgt_ne, out_rel2, wgt_rel)
            
            optim.zero_grad()
            ((ls_ne1+ls_rel1) + ALP*(ls_ne2+ls_rel2)).backward()
            optim.step()
            
            ls_ne1 = ls_ne1.detach().cpu().numpy()
            ls_rel1 = ls_rel1.detach().cpu().numpy()
            ls_ep_ne1 += ls_ne1
            ls_ep_rel1 += ls_rel1
            
            ls_ne2 = ls_ne2.detach().cpu().numpy()
            ls_rel2 = ls_rel2.detach().cpu().numpy()
            ls_ep_ne2 += ls_ne2
            ls_ep_rel2 += ls_rel2
            
            TQ.set_postfix(ls_ne1='%.3f'%(ls_ne1), ls_rel1='%.3f'%(ls_rel1), 
                           ls_ne2='%.3f'%(ls_ne2), ls_rel2='%.3f'%(ls_rel2))
            
            if i%100==0:
                for pg in optim.param_groups:
                    pg['lr'] *= DECAY
            
        ls_ep_ne1 /= len(TQ)
        ls_ep_rel1 /= len(TQ)
        
        ls_ep_ne2 /= len(TQ)
        ls_ep_rel2 /= len(TQ)
        
        print('Ep %d: ne1: %.4f, rel1: %.4f, ne2: %.4f, rel2: %.4f' % (e+1, ls_ep_ne1, ls_ep_rel1, 
                                                                       ls_ep_ne2, ls_ep_rel2))
        T.save(model.state_dict(), 'Model/%s_%s_%d.pt' % (DATASET, ARCH, e+1))
    
 #   f1 = F1()
 #   model.eval()
 #   with tqdm(ld_vl) as TQ:
 #       for idx, inp, pos, dep_fw, dep_bw, ans_ne, wgt_ne, ans_rel, wgt_rel in TQ:
 #           _, _, out_ne, out_rel = model(inp.cuda(), pos.cuda(), dep_fw.cuda(), dep_bw.cuda())
 #           
 #           for i in range(idx.shape[0]):
 #               rels, ans = post_proc(vl, idx[i], out_ne[i], out_rel[i])
 #               f1.add(rels, ans)
 #       
 #       p, r, f = f1.get()
 #       print('P: %.4f%%, R: %.4f%%, F: %.4f%%' % (100*p, 100*r, 100*f))

