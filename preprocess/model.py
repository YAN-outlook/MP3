import math
from re import M
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys

import torch
import torch.nn as nn



class TemporalConvProjector(nn.Module):
    def __init__(self, D, C, kernel_size=3):
        super().__init__()
        self.proj = nn.Conv2d(1, C, kernel_size=(kernel_size, D),
                              padding=(kernel_size//2, 0))
        self.bn = nn.BatchNorm2d(C)
        self.act = nn.ReLU()

    def forward(self, x):
        # x: [B, N, T, D]
        B,N,T,D=x.shape
        x = x.unsqueeze(1)
        # print(x.shape)
        x = x.reshape(B*N, 1, T, D)
        # print(x.shape)
        out = self.proj(x)  # [B*N, C, T', 1]
        # print("out1",out.shape)
        out = self.bn(out)
        out = self.act(out)
        out = out.mean(dim=2)
        # print("out2",out.shape)
        out = out.reshape(B, N, -1)  # [B, N, C]
        return out
        
class PeriodInceptionEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, pool_size=(1,1), init_weight=True):
        """
        """
        super(PeriodInceptionEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        self.pool_size = pool_size

        kernels = []
        for i in range(self.num_kernels // 2):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[1, 2 * i + 3], padding=[0, i + 1]))
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[2 * i + 3, 1], padding=[i + 1, 0]))
        kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        self.kernels = nn.ModuleList(kernels)

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(pool_size)

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        x: [B, N, L//period, period, D]
        return: [B, N, m, n, out_channels]
        """
        B, N, H, W, D = x.shape

        x = x.permute(0,1,4,2,3).reshape(B*N, D, H, W)  # [B*N, D, H, W]

        res_list = []
        for i in range(self.num_kernels // 2 * 2 + 1):
            res_list.append(self.kernels[i](x))  # [B*N, out_channels, H, W]
        res = torch.stack(res_list, dim=-1).mean(-1)
        # BatchNorm + ReLU + Pooling
        res = self.bn(res)
        res = self.relu(res)
        res = self.pool(res)  # [B*N, out_channels, m, n]
        m, n = res.shape[2], res.shape[3]
        res = res.reshape(B, N, self.out_channels, m, n).permute(0,1,3,4,2).contiguous()#(B,N,hid,m,n)
        return res



class SparseGCN(nn.Module):
    def __init__(self, in_channels, out_channels,adj):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.adj=adj
        self.norm = nn.LayerNorm(out_channels)
    def forward(self, x,node_index=None):
        """
        return: [B, N, P, out_channels]
        """
        #[B*Lp, N, hid_dim//4, P]
        B, N, P, C = x.shape
        if node_index is None:
            node_index = torch.arange(N)
        # print("N",N)    
        # print(x.shape)

        # print(adj.shape)
        if self.adj.is_sparse:
            adj_dense = self.adj.to_dense()
            adj_sub = adj_dense[node_index][:, node_index]
            adj = adj_sub.to_sparse()
            x_agg = []
            for b in range(B):
                x_b = x[b]  # [N, P, C]
                x_b = x_b.reshape(N, -1)  # [N, P*C]
                # print("x_b",x_b.shape)
                x_b = torch.sparse.mm(adj, x_b)  # [N, P*C]
                x_b = x_b.reshape(N, P, C)
                x_agg.append(x_b)
            x = torch.stack(x_agg, dim=0)  # [B, N, P, C]
        else:
            x_agg = torch.einsum("ij,bjpc->bipc", adj, x)
            x = x_agg
        # print(x.shape) 
        # x=x.transpose(2,3)   
        x = self.linear(x)  # [B, N, P,C]
        x=self.norm(x)
        return x
class MemoryBank(nn.Module):
    def __init__(self, N, C, momentum=0.9):
        super().__init__()
        self.register_buffer('memory', torch.zeros(N, C))
        self.momentum = momentum
        
        self.proj = nn.Linear(2 * C, C)

    def forward(self, x):
        B, N, P, C = x.shape
        current_global = x.mean(dim=(0, 2))
        with torch.no_grad():
            self.memory.mul_(self.momentum).add_((1 - self.momentum) * current_global)
        global_context = self.memory.unsqueeze(0).unsqueeze(2).expand(B, N, P, C) 
        x = torch.cat([x, global_context], dim=-1)  # [B, N, P, 2C]  
        x = self.proj(x)  # [B, N, P, C]
        
        return x
class IEBlock(nn.Module):
    def __init__(self, hid_dim, output_dim, d_model, adj, momentum=0.9):
        super().__init__()
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.momentum = momentum
        
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(
                in_channels=self.hid_dim, 
                out_channels=hid_dim // 4, 
                kernel_size=(1, 3),
                padding=(0, 1)
            ),
            nn.BatchNorm2d(hid_dim // 4), 
            nn.LeakyReLU()
        )

        self.graph_conv = SparseGCN(hid_dim // 4, hid_dim // 4, adj)
        self.memory_banks = {}  
        self.output_proj = nn.Linear(hid_dim // 4, output_dim)
        self.Norm = nn.LayerNorm(hid_dim // 4)

    def forward(self, x, node_index):
        B, N, Lp, P, D = x.shape

        x = x.reshape(B * N, Lp, P, D)  # [B*N, Lp, P, D]
        
        x = x.permute(0, 3, 1, 2).contiguous()  # [B*N, D, Lp, P]

        x = self.spatial_proj(x)

        C = x.shape[1]
        x = x.permute(0, 2, 3, 1).contiguous()  # [B*N, Lp, P, C]
        x = x.reshape(B, N, Lp, P, C)           # [B, N, Lp, P, C]
        x = x.reshape(B, N, Lp * P, C)          # [B, N, T_total, C] (T_total = Lp*P)

        x = self.graph_conv(x, node_index)  # [B, N, T_total, C]
        
        if N not in self.memory_banks:
            self.memory_banks[N] = MemoryBank(N, x.size(-1), self.momentum).to(x.device)
        memory_bank = self.memory_banks[N]
        x = memory_bank(x)  # [B, N, T_total, C]

        x = self.Norm(x)
        x = self.output_proj(x)  # [B, N, T_total, output_dim]

        return x


class PeriodAttentionBlock(nn.Module):
    """
  
    """
    def __init__(self, hidden_dim, k=3, period_aware_module=None):
        super(PeriodAttentionBlock, self).__init__()
        self.k = k
        self.hidden_dim = hidden_dim
        self.period_aware = PeriodInceptionEncoder(hidden_dim, hidden_dim, num_kernels=3, pool_size=(13,8))
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        
    def FFT_for_Period(self, x, k=3):
        B, T, N, D = x.shape
        xf = torch.fft.rfft(x, dim=1)
        amplitude = torch.abs(xf)  # [B, T//2+1, N, D]
        freq_energy = amplitude.mean(dim=(0, 2, 3))  # [T//2+1]
        sorted_idx = torch.argsort(freq_energy, descending=True)
        period_list = []
        freq_list = []
        for idx in sorted_idx:
            period = int(T // idx) if idx != 0 else 0
            if period > 0 and period not in period_list:
                period_list.append(period)
                freq_list.append(idx)
            if len(period_list) >= k:
                break
        period_list = torch.tensor(period_list, device=x.device)
        freq_list = torch.tensor(freq_list, device=x.device)
        freq_amp = amplitude[:, freq_list, :, :]  # [B, k, N, D]
        return period_list, freq_amp

    def divide_period(self, x, period_list, B, N, D):
        T = x.shape[1]
        period = period_list
        # print(period)
        # padding
        period = int(period.item()) if torch.is_tensor(period) else int(period)
        # print(T)
        # print(period)
        if T % period != 0:
            length = ((T // period) + 1) * period
            padding = torch.zeros([x.shape[0], (length - T), x.shape[2], D]).to(x.device)
            out = torch.cat([x, padding], dim=1)
        else:
            length = T
            out = x
        # reshape
        out = out.reshape(B, length // period, period, N, D).permute(0, 3, 1, 2, 4).contiguous()
        return out  # out (B,N,len//period,period,d_model)
    
    def forward(self, x):
        """
        x: [B, T, N, D]
        return: [B, N, T, D]
        """
        B, T, N, D = x.size()
        
        period_list, period_weight = self.FFT_for_Period(x, self.k)
        period_weight = F.softmax(period_weight, dim=1)
        
        agg = torch.zeros(B, N, T, D, device=x.device)
        
        for i, p in enumerate(period_list):
            x_p = self.divide_period(x, p, B, N, self.hidden_dim)
            x_period_inter = self.period_aware(x_p)
            #[B,N,m,n,D]
            x_period_inter = x_period_inter.reshape(B, N, T, D).contiguous()
            weight = period_weight[:, i, :, :].unsqueeze(2)  # -> [B, N, 1, D]
            agg = agg + weight * x_period_inter
        agg = self.layer_norm(agg)
        return agg

class PeriodAttentionBlock_new(nn.Module):
    def __init__(self, hidden_dim, adj, k=3, period_aware_module=None):
        super().__init__()
        self.k = k
        self.hidden_dim = hidden_dim
        self.period_aware = PeriodInceptionEncoder(hidden_dim, hidden_dim, num_kernels=3, pool_size=(13,8))
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.IEBlock = IEBlock(hidden_dim, hidden_dim, hidden_dim, adj)
        
    def FFT_for_Period(self, x,k=3):
        B, T, N, D = x.shape
        xf = torch.fft.rfft(x, dim=1)
        amplitude = torch.abs(xf)  # [B, T//2+1, N, D]
        freq_energy = amplitude.mean(dim=(0, 2, 3))  # [T//2+1]
        sorted_idx = torch.argsort(freq_energy, descending=True)
        period_list = []
        freq_list = []
        for idx in sorted_idx:
            period = int(T // idx) if idx != 0 else 0
            if period > 0 and period not in period_list:
                period_list.append(period)
                freq_list.append(idx)
            if len(period_list) >= k:
                break
        period_list = torch.tensor(period_list, device=x.device)
        freq_list = torch.tensor(freq_list, device=x.device)
        freq_amp = amplitude[:, freq_list, :, :]  # [B, k, N, D]
        return period_list, freq_amp

    def divide_period(self, x, period_list, B, N, D):
        T = x.shape[1]
        period = period_list
        period = int(period.item()) if torch.is_tensor(period) else int(period)
        if T % period != 0:
            length = ((T // period) + 1) * period
            padding = torch.zeros([x.shape[0], (length - T), x.shape[2], D]).to(x.device)
            out = torch.cat([x, padding], dim=1)
        else:
            length = T
            out = x
        out = out.reshape(B, length // period, period, N, D).permute(0, 3, 1, 2, 4).contiguous()
        return out  # out (B,N,len//period,period,d_model)
    
    def forward(self, x, node_index):
        B, T, N, D = x.size()
        
        period_list, period_weight = self.FFT_for_Period(x, self.k)
        
        agg = torch.zeros(self.k, B, N, T, D, device=x.device)
        
        for i, p in enumerate(period_list):
            x_p = self.divide_period(x, p, B, N, self.hidden_dim)
            x_period_inter = self.period_aware(x_p)
            x_period_inter = self.layer_norm(x_period_inter)
            
            x_period_inter = self.IEBlock(x_period_inter, node_index)
            
            agg[i] = x_period_inter
            
        agg = self.layer_norm(agg)
        return agg

import torch
import torch.nn as nn
import math

class PeriodGraphCausalBlock(nn.Module):
    def __init__(self, K, d_model, use_soft_mask=True, soft_mask_init=0.1):
        super().__init__()
        self.K = K
        self.d_model = d_model
        self.use_soft_mask = use_soft_mask

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

        if use_soft_mask:
            self.causal_mask = nn.Parameter(
                torch.triu(torch.ones(K, K) * soft_mask_init, diagonal=1)
            )
        else:
            self.register_buffer(
                "causal_mask", 
                torch.triu(torch.ones(K, K, dtype=torch.bool), diagonal=1)
            )

        self.proj_out = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        return: [B, T, N, D]
        """
        K, B, N, T, D = x.shape
        assert K == self.K, f"输入周期数{K}与初始化{self.K}不匹配"

        q = self.Wq(x)  # [K, B, N, T, D]
        k = self.Wk(x)  # [K, B, N, T, D]
        v = self.Wv(x)  # [K, B, N, T, D]

        sim = torch.einsum("kbntd, mbntd -> kmb", q, k)
        # scale = 1.0 / math.sqrt(N * T * D)
        # sim = sim * scale
        
        if self.use_soft_mask:
            sim = sim - self.causal_mask.unsqueeze(-1)  # [K,K,B]
        else:
            sim = sim.masked_fill(self.causal_mask.unsqueeze(-1), -1e9)
        sim = (sim - sim.mean(dim=1, keepdim=True)) / (sim.std(dim=1, keepdim=True) + 1e-6)
        # print(sim.shape)
        mean_val = sim.mean()
        var_val = sim.var()
      
        print("平均值:", mean_val)
        print("方差:", var_val)
        attn = torch.softmax(sim, dim=1)
        
        print(attn)
        # print(attn)
        out = torch.einsum("kmb, mbntd -> kbntd", attn, v)

        out = out.sum(dim=0)  # [B, N, T, D]
        out = out.permute(0, 2, 1, 3)  # [B, T, N, D]

        out = self.proj_out(out)

        return out
        
class linear_transformer(nn.Module):
    def __init__(self, input_length, output_length, in_dim, num_nodes, nhid,adj):
        super(linear_transformer, self).__init__()

        self.k=3
        self.alpha=0.8
        self.hidden_dim=32
        self.seq_len=1248//12
        self.pred_len=output_length
        
        self.period_aware = PeriodInceptionEncoder(in_channels=self.hidden_dim, 
                                                  out_channels=self.hidden_dim,
                                                  num_kernels=6, 
                                                  pool_size=(13,8))
        self.period_block1 = PeriodAttentionBlock(self.hidden_dim, k=self.k, period_aware_module=self.period_aware)
        self.period_block2 = PeriodAttentionBlock(self.hidden_dim, k=self.k, period_aware_module=self.period_aware)
        self.period_block3 = PeriodAttentionBlock_new(self.hidden_dim,adj,k=self.k, period_aware_module=self.period_aware)
        
        self.temporal_embedding = nn.Parameter(torch.empty(int(input_length/12), self.hidden_dim), requires_grad=True) # (C, nhid)
        nn.init.xavier_uniform_(self.temporal_embedding)
        # self.temperal_projection = TemporalConvProjector(self.hidden_dim, nhid)
        
        self.regression_layer = nn.Linear(nhid, output_length)
        self.context_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.hidden_dim, kernel_size=(12, 1), stride=(12, 1))
        self.predict_linear = nn.Linear(self.seq_len, self.pred_len)
        self.projection = nn.Linear(self.hidden_dim, 12)
        # self.IEBlock = IEBlock(self.hidden_dim, self.hidden_dim,self.hidden_dim,adj)
        self.cross_period_block = PeriodGraphCausalBlock(self.k, self.hidden_dim)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        # self.layer_norm_2 = nn.LayerNorm(self.hidden_dim)
        self.feat_predict = nn.Linear(self.hidden_dim, 1)

    def forward(self, x,node_index):
        # input: (1, 9638, 2016, 3) (B, N, T, D)
        B, N, T, D = x.size()
        x = x.reshape(B*N, T, D)
        x = x.permute(0, 2, 1).unsqueeze(-1) # (B*N, T, D) -> (B*N, D, T, 1)
        x = self.context_conv(x) # (B*N, D, T, 1) -> (B*N, nhid, T/12, 1)
        x = x.squeeze(-1) # (B*N, nhid, T/12)
        # temporal embedding layer
        x = x.permute(0, 2, 1) # (B*N, T/12, nhid)

        pe = self.temporal_embedding.unsqueeze(0).expand(B*N, -1, -1) # (B*N, T/12, nhid)
        x = x+pe # (B*N, T/12, nhid)
        x = x.reshape(B, N, T//12, self.hidden_dim)
        x = x.transpose(1, 2)  # [B, T, N, D]
        
        agg1 = self.period_block1(x)
        var_over_T = agg1.var(dim=1)
        if (var_over_T == 0).any():
            print(f"⚠️ 警告：输入张量 agg1 在 T 维度存在方差为 0 的元素！")
        x2 = agg1.transpose(1, 2)+x  # [B, N, T, D] -> [B, T, N, D]
        agg2 = self.period_block2(x2)
        var_over_T = agg2.var(dim=1)
        # if (var_over_T == 0).any():
        x3 = agg2.transpose(1, 2)+x2  # [B, N, T, D] -> [B, T, N, D]
        agg3 = self.period_block3(x3,node_index) #[k,B,N,T,D]
        var_over_T = agg3.var(dim=3)
        # if (var_over_T == 0).any():
        agg = self.cross_period_block(agg3) #[B,T,N,D]
        agg = self.layer_norm(agg) + x3
        # print(agg.shape)
        agg_reshaped = agg.permute(0,2,3,1).contiguous()  # [B, N, D, T]
        agg_reshaped = self.predict_linear(agg_reshaped)  # [B, N, D, pred_len]
        agg_reshaped = agg_reshaped.permute(0, 1, 3, 2)  # [B, N, pred_len, D]
        # agg_reshaped = self.layer_norm_2(agg_reshaped)
        
        x = self.projection(agg_reshaped)  # [B, N, pred_len, 1]
        x = x.squeeze(-1).transpose(1,2)  # [B, N, pred_len]
        feat=agg_reshaped.transpose(1,2)
        # feat = self.temperal_projection(agg_reshaped)
        
        return x, feat