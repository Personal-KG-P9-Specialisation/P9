import math
import torch.nn as nn
import torch as T

class GCN(nn.Module):
    def __init__(self, hid_size=256):
        super(GCN, self).__init__()
        
        self.hid_size = hid_size
        
        self.W = nn.Parameter(T.FloatTensor(self.hid_size, self.hid_size//2).cuda())
        self.b = nn.Parameter(T.FloatTensor(self.hid_size//2, ).cuda())
        
        self.init()
    
    def init(self):
        stdv = 1/math.sqrt(self.hid_size//2)
        
        self.W.data.uniform_(-stdv, stdv)
        self.b.data.uniform_(-stdv, stdv)
    
    def forward(self, inp, adj, is_relu=True):
        out = T.matmul(inp, self.W)+self.b
        out = T.matmul(adj, out)
        
        if is_relu==True:
            out = nn.functional.relu(out)
        
        return out
    
    def __repr__(self):
        return self.__class__.__name__+'(hid_size=%d)'%(self.hid_size)
