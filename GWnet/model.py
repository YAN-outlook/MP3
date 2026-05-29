import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# class TemporalConvProjector(nn.Module):
#     def __init__(self, D, C, kernel_size=3):
#         super().__init__()
#         self.proj = nn.Conv2d(1, C, kernel_size=(kernel_size, D),
#                               padding=(kernel_size//2, 0))
#         self.bn = nn.BatchNorm2d(C)
#         self.act = nn.ReLU()

#     def forward(self, x):
#         # x: [B, N, T, D]
#         B, N, T, D = x.shape
#         x = x.reshape(B * N, 1, T, D)
#         out = self.proj(x)           # [B*N, C, T, 1]
#         out = self.bn(out)
#         out = self.act(out)
#         out = out.mean(dim=2)        # [B*N, C]
#         out = out.reshape(B, N, -1)  # [B, N, C]
#         return out


# class FusionModule(nn.Module):
#     def __init__(self, in_dim, skip_dim, kernel_size=3):
#         super().__init__()
#         self.proj = TemporalConvProjector(D=in_dim, C=skip_dim, kernel_size=kernel_size)
#         self.ln = nn.LayerNorm(skip_dim)
#         self.gate_conv = nn.Conv2d(skip_dim, skip_dim, kernel_size=1)

#     def forward(self, feat, skip):
#         """
#         """
#         p = self.proj(feat)              # [B, N, Cskip]
#         p = self.ln(p)
#         p = p.transpose(1, 2).unsqueeze(-1)  # [B, Cskip, N, 1]

#         gate = torch.sigmoid(self.gate_conv(skip))  # [B, Cskip, N, 1]

#         out = skip + gate * (self.alpha * p)
#         return out


# class MP3(nn.Module):
#     def __init__(self, args, supports=None, edge_indices=None):
#         super(MP3, self).__init__()
#         self.tau = args.tau
#         self.num_layers = args.num_layers
#         self.random_feature_dim = args.random_feature_dim
        
#         self.use_residual = args.use_residual
#         self.use_bn = args.use_bn
#         self.use_spatial = args.use_spatial
#         self.use_long = args.use_long
        
#         self.dropout = args.dropout
#         self.activation = nn.ReLU()
#         self.supports = supports
#         self.edge_indices = edge_indices
        
#         self.time_num = args.time_num
#         self.week_num = args.week_num
        
#         # node embedding layer
#         self.node_emb_layer = nn.Parameter(torch.empty(args.num_nodes, args.node_dim))
#         nn.init.xavier_uniform_(self.node_emb_layer)
        
#         # time embedding layer
#         self.time_emb_layer = nn.Parameter(torch.empty(self.time_num, args.time_dim))
#         nn.init.xavier_uniform_(self.time_emb_layer)
#         self.week_emb_layer = nn.Parameter(torch.empty(self.week_num, args.time_dim))
#         nn.init.xavier_uniform_(self.week_emb_layer)

#         # embedding layer
#         self.input_emb_layer = nn.Conv2d(args.output_length*args.in_dim, args.hid_dim, kernel_size=(1, 1), bias=True)
        
#         self.W_1 = nn.Conv2d(args.node_dim+args.time_dim*2, args.hid_dim, kernel_size=(1, 1), bias=True)
#         self.W_2 = nn.Conv2d(args.node_dim+args.time_dim*2, args.hid_dim, kernel_size=(1, 1), bias=True)
        
#         self.linear_conv = nn.ModuleList()
#         self.bn = nn.ModuleList()
        
#         self.supports_len = 0
#         if supports is not None:
#             self.supports_len += len(supports)
        
#         for i in range(self.num_layers):
#             self.linear_conv.append(linearized_conv(args.hid_dim*4, args.hid_dim*4, self.dropout, self.tau, self.random_feature_dim))
#             self.bn.append(nn.LayerNorm(args.hid_dim*4))
        
#         if self.use_long:
#             self.regression_layer = nn.Conv2d(args.hid_dim*4*2+args.hid_dim+args.output_length, args.output_length, kernel_size=(1, 1), bias=True)
#         else:
#             self.regression_layer = nn.Conv2d(args.hid_dim*4*2, args.output_length, kernel_size=(1, 1), bias=True)

#     def forward(self, x, feat=None):
#         # input: (B, N, T, D)
#         B, N, T, D = x.size()
        
#         time_emb = self.time_emb_layer[(x[:, :, -1, 1]*self.time_num).type(torch.LongTensor)]
#         week_emb = self.week_emb_layer[(x[:, :, -1, 2]).type(torch.LongTensor)]
        
#         # input embedding
#         x = x.contiguous().view(B, N, -1).transpose(1, 2).unsqueeze(-1) # (B, D*T, N, 1)
#         input_emb = self.input_emb_layer(x)

#         # node embeddings
#         node_emb = self.node_emb_layer.unsqueeze(0).expand(B, -1, -1).transpose(1, 2).unsqueeze(-1) # (B, dim, N, 1)

#         # time embeddings
#         time_emb = time_emb.transpose(1, 2).unsqueeze(-1) # (B, dim, N, 1)
#         week_emb = week_emb.transpose(1, 2).unsqueeze(-1) # (B, dim, N, 1)
        
#         x_g = torch.cat([node_emb, time_emb, week_emb], dim=1) # (B, dim*4, N, 1)
#         x = torch.cat([input_emb, node_emb, time_emb, week_emb], dim=1) # (B, dim*4, N, 1)

#         # linearized spatial convolution
#         x_pool = [x] # (B, dim*4, N, 1)
#         node_vec1 = self.W_1(x_g) # (B, dim, N, 1)
#         node_vec2 = self.W_2(x_g) # (B, dim, N, 1)
#         node_vec1 = node_vec1.permute(0, 2, 3, 1) # (B, N, 1, dim)
#         node_vec2 = node_vec2.permute(0, 2, 3, 1) # (B, N, 1, dim)
#         for i in range(self.num_layers):
#             if self.use_residual:
#                 residual = x
#             x, node_vec1_prime, node_vec2_prime = self.linear_conv[i](x, node_vec1, node_vec2)
            
#             if self.use_residual:
#                 x = x+residual 
                
#             if self.use_bn:
#                 x = x.permute(0, 2, 3, 1) # (B, N, 1, dim*4)
#                 x = self.bn[i](x)
#                 x = x.permute(0, 3, 1, 2)

#         x_pool.append(x)
#         x = torch.cat(x_pool, dim=1) # (B, dim*4, N, 1)
        
#         x = self.activation(x) # (B, dim*4, N, 1)
        
#         if self.use_long:
#             feat = feat.permute(0, 2, 1).unsqueeze(-1) # (B, F, N, 1)
#             x = torch.cat([x, feat], dim=1)
#             x = self.regression_layer(x) # (B, N, T)
#             x = x.squeeze(-1).permute(0, 2, 1)
#         else:
#             x = self.regression_layer(x) # (B, N, T)
#             x = x.squeeze(-1).permute(0, 2, 1)
        
#         if self.use_spatial:
#             s_loss = spatial_loss(node_vec1_prime, node_vec2_prime, self.supports, self.edge_indices)
#             return x, s_loss
#         else:
#             return x, 0


# def create_products_of_givens_rotations(dim, seed):
#     nb_givens_rotations = dim * int(math.ceil(math.log(float(dim))))
#     q = np.eye(dim, dim)
#     np.random.seed(seed)
#     for _ in range(nb_givens_rotations):
#         random_angle = math.pi * np.random.uniform()
#         random_indices = np.random.choice(dim, 2)
#         index_i = min(random_indices[0], random_indices[1])
#         index_j = max(random_indices[0], random_indices[1])
#         slice_i = q[index_i]
#         slice_j = q[index_j]
#         new_slice_i = math.cos(random_angle) * slice_i + math.cos(random_angle) * slice_j
#         new_slice_j = -math.sin(random_angle) * slice_i + math.cos(random_angle) * slice_j
#         q[index_i] = new_slice_i
#         q[index_j] = new_slice_j
#     return torch.tensor(q, dtype=torch.float32)


# def create_random_matrix(m, d, seed=0, scaling=0, struct_mode=False):
#     nb_full_blocks = int(m/d)
#     block_list = []
#     current_seed = seed
#     for _ in range(nb_full_blocks):
#         torch.manual_seed(current_seed)
#         if struct_mode:
#             q = create_products_of_givens_rotations(d, current_seed)
#         else:
#             unstructured_block = torch.randn((d, d))
#             q, _ = torch.qr(unstructured_block)
#             q = torch.t(q)
#         block_list.append(q)
#         current_seed += 1
#     remaining_rows = m - nb_full_blocks * d
#     if remaining_rows > 0:
#         torch.manual_seed(current_seed)
#         if struct_mode:
#             q = create_products_of_givens_rotations(d, current_seed)
#         else:
#             unstructured_block = torch.randn((d, d))
#             q, _ = torch.qr(unstructured_block)
#             q = torch.t(q)
#         block_list.append(q[0:remaining_rows])
#     final_matrix = torch.vstack(block_list)

#     current_seed += 1
#     torch.manual_seed(current_seed)
#     if scaling == 0:
#         multiplier = torch.norm(torch.randn((m, d)), dim=1)
#     elif scaling == 1:
#         multiplier = torch.sqrt(torch.tensor(float(d))) * torch.ones(m)
#     else:
#         raise ValueError("Scaling must be one of {0, 1}. Was %s" % scaling)

#     return torch.matmul(torch.diag(multiplier), final_matrix)


# def random_feature_map(data, is_query, projection_matrix=None, numerical_stabilizer=0.000001):
#     data_normalizer = 1.0 / torch.sqrt(torch.sqrt(torch.tensor(data.shape[-1], dtype=torch.float32)))
#     data = data_normalizer * data
#     ratio = 1.0 / torch.sqrt(torch.tensor(projection_matrix.shape[0], dtype=torch.float32))
#     data_dash = torch.einsum("bnhd,md->bnhm", data, projection_matrix)
#     diag_data = torch.square(data)
#     diag_data = torch.sum(diag_data, dim=len(data.shape)-1)
#     diag_data = diag_data / 2.0
#     diag_data = torch.unsqueeze(diag_data, dim=len(data.shape)-1)
#     last_dims_t = len(data_dash.shape) - 1
#     attention_dims_t = len(data_dash.shape) - 3
#     if is_query:
#         data_dash = ratio * (
#             torch.exp(data_dash - diag_data - torch.max(data_dash, dim=last_dims_t, keepdim=True)[0]) + numerical_stabilizer
#         )
#     else:
#         data_dash = ratio * (
#             torch.exp(data_dash - diag_data - torch.max(torch.max(data_dash, dim=last_dims_t, keepdim=True)[0],
#                     dim=attention_dims_t, keepdim=True)[0]) + numerical_stabilizer
#         )
#     return data_dash


# def linear_kernel(x, node_vec1, node_vec2):
#     # x: [B, N, 1, nhid] node_vec1: [B, N, 1, r], node_vec2: [B, N, 1, r]
#     node_vec1 = node_vec1.permute(1, 0, 2, 3) # [N, B, 1, r]
#     node_vec2 = node_vec2.permute(1, 0, 2, 3) # [N, B, 1, r]
#     x = x.permute(1, 0, 2, 3) # [N, B, 1, nhid]
    
#     v2x = torch.einsum("nbhm,nbhd->bhmd", node_vec2, x)
#     out1 = torch.einsum("nbhm,bhmd->nbhd", node_vec1, v2x) # [N, B, 1, nhid]
    
#     one_matrix = torch.ones([node_vec2.shape[0]]).to(node_vec1.device)
#     node_vec2_sum = torch.einsum("nbhm,n->bhm", node_vec2, one_matrix)
#     out2 = torch.einsum("nbhm,bhm->nbh", node_vec1, node_vec2_sum) # [N, 1]

#     out1 = out1.permute(1, 0, 2, 3)  # [B, N, 1, nhid]
#     out2 = out2.permute(1, 0, 2)
#     out2 = torch.unsqueeze(out2, len(out2.shape))
#     out = out1 / out2 # [B, N, 1, nhid]

#     return out

    
# def spatial_loss(node_vec1, node_vec2, supports, edge_indices):
#     B = node_vec1.size(0)
#     node_vec1 = node_vec1.permute(1, 0, 2, 3) # [N, B, 1, r]
#     node_vec2 = node_vec2.permute(1, 0, 2, 3) # [N, B, 1, r]
    
#     node_vec1_end, node_vec2_start = node_vec1[edge_indices[:, 0]], node_vec2[edge_indices[:, 1]] # [E, B, 1, r]
#     attn1 = torch.einsum("ebhm,ebhm->ebh", node_vec1_end, node_vec2_start) # [E, B, 1]
#     attn1 = attn1.permute(1, 0, 2) # [B, E, 1]

#     one_matrix = torch.ones([node_vec2.shape[0]]).to(node_vec1.device)
#     node_vec2_sum = torch.einsum("nbhm,n->bhm", node_vec2, one_matrix)
#     attn_norm = torch.einsum("nbhm,bhm->nbh", node_vec1, node_vec2_sum)
    
#     attn2 = attn_norm[edge_indices[:, 0]]  # [E, B, 1]
#     attn2 = attn2.permute(1, 0, 2) # [B, E, 1]
#     attn_score = attn1 / attn2 # [B, E, 1]
    
#     d_norm = supports[0][edge_indices[:, 0], edge_indices[:, 1]]
#     d_norm = d_norm.reshape(1, -1, 1).repeat(B, 1, attn_score.shape[-1])
#     spatial_loss = torch.mean(attn_score.log() * d_norm)
    
#     return spatial_loss

    
# class conv_approximation(nn.Module):
#     def __init__(self, dropout, tau, random_feature_dim):
#         super(conv_approximation, self).__init__()
#         self.tau = tau
#         self.random_feature_dim = random_feature_dim
#         self.activation = nn.ReLU()
#         self.dropout = dropout

#     def forward(self, x, node_vec1, node_vec2):
#         B = x.size(0) # (B, N, 1, nhid)
#         dim = node_vec1.shape[-1] # (N, 1, d)
        
#         random_seed = torch.ceil(torch.abs(torch.sum(node_vec1) * 1e8)).to(torch.int32)
#         random_matrix = create_random_matrix(self.random_feature_dim, dim, seed=random_seed).to(node_vec1.device) # (d, r)
        
#         node_vec1 = node_vec1 / math.sqrt(self.tau)
#         node_vec2 = node_vec2 / math.sqrt(self.tau)
#         node_vec1_prime = random_feature_map(node_vec1, True, random_matrix) # [B, N, 1, r]
#         node_vec2_prime = random_feature_map(node_vec2, False, random_matrix) # [B, N, 1, r]
        
#         x = linear_kernel(x, node_vec1_prime, node_vec2_prime)
        
#         return x, node_vec1_prime, node_vec2_prime


# class linearized_conv(nn.Module):
#     def __init__(self, in_dim, hid_dim, dropout, tau=1.0, random_feature_dim=64):
#         super(linearized_conv, self).__init__()
        
#         self.dropout = dropout
#         self.tau = tau
#         self.random_feature_dim = random_feature_dim
        
#         self.input_fc = nn.Conv2d(in_channels=in_dim, out_channels=hid_dim, kernel_size=(1, 1), bias=True)
#         self.output_fc = nn.Conv2d(in_channels=in_dim, out_channels=hid_dim, kernel_size=(1, 1), bias=True)
#         self.activation = nn.Sigmoid()
#         self.dropout_layer = nn.Dropout(p=dropout)
        
#         self.conv_app_layer = conv_approximation(self.dropout, self.tau, self.random_feature_dim)
        
#     def forward(self, input_data, node_vec1, node_vec2):
#         x = self.input_fc(input_data)
#         x = self.activation(x)*self.output_fc(input_data)
#         x = self.dropout_layer(x)
        
#         x = x.permute(0, 2, 3, 1) # (B, N, 1, dim*4)
#         x, node_vec1_prime, node_vec2_prime = self.conv_app_layer(x, node_vec1, node_vec2)
#         x = x.permute(0, 3, 1, 2) # (B, dim*4, N, 1)
        
#         return x, node_vec1_prime, node_vec2_prime


import torch
from torch import nn
import torch.nn.functional as F


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        A = A.to(x.device)
        if len(A.shape) == 3:
            x = torch.einsum('ncvl,nvw->ncwl',(x,A))
        else:
            x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class Fusion(nn.Module):
    def __init__(self, dim):
        super(Fusion, self).__init__()
        self.HS_fc = nn.Linear(dim, dim, bias=True)
        self.HT_fc = nn.Linear(dim, dim, bias=True)
        self.output_fc = nn.Linear(dim, dim, bias=True)

    def forward(self, flow_eb, time_eb):
        XS = self.HS_fc(flow_eb)
        XT = self.HT_fc(time_eb)
        z = torch.sigmoid(torch.add(XS, XT))
        H = torch.add(torch.multiply(z, flow_eb), torch.multiply(1 - z, time_eb))
        H = self.output_fc(H)
        return H

# class Fusion(nn.Module):
#     def __init__(self, dim, dropout=0.1):
#         super(Fusion, self).__init__()
#         self.HS_fc = nn.Linear(dim, dim, bias=True)
#         self.HT_fc = nn.Linear(dim, dim, bias=True)
#         self.gate_fc = nn.Linear(dim * 2, dim)
#         self.output_fc = nn.Linear(dim, dim, bias=True)
#         self.norm = nn.LayerNorm(dim)
#         self.dropout = nn.Dropout(dropout)
#         self.norm_feat = nn.LayerNorm(dim)

#     def forward(self, flow_eb, time_eb):
#         XS = self.HS_fc(flow_eb)
#         XT = self.HT_fc(time_eb)
#         XS = self.norm_feat(XS) 
#         z = torch.sigmoid(self.gate_fc(torch.cat([XS, XT], dim=-1)))
#         H = flow_eb + z * (time_eb - flow_eb)
#         H = self.output_fc(F.gelu(H))
#         H = self.norm(H)
#         H = self.dropout(H)
#         return H

class GraphWaveNet(nn.Module):
    """
        Paper: Graph WaveNet for Deep Spatial-Temporal Graph Modeling.
        Link: https://arxiv.org/abs/1906.00121
        Ref Official Code: https://github.com/nnzhan/Graph-WaveNet/blob/master/model.py
    """

    def __init__(self, num_nodes, supports=None, dropout=0.3, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2, out_dim=12, residual_channels=32, dilation_channels=32, skip_channels=256, end_channels=512, kernel_size=2, blocks=4, layers=2, use_plugin=True, **kwargs):

        super(GraphWaveNet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.use_plugin = use_plugin

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.in_dim = in_dim
        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1,1))
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10), requires_grad=True)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes), requires_grad=True)
                self.supports_len +=1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True)
                self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels, out_channels=dilation_channels, kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels, out_channels=dilation_channels, kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels, out_channels=residual_channels, kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels, out_channels=skip_channels, kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))

        self.end_conv_1_plugin = nn.Conv2d(in_channels=skip_channels, out_channels=end_channels, kernel_size=(1,1), bias=True)
        self.end_conv_1_base = nn.Conv2d(in_channels=skip_channels, out_channels=end_channels, kernel_size=(1,1), bias=True)
        self.end_conv_2_plugin = nn.Conv2d(in_channels=end_channels, out_channels=out_dim, kernel_size=(1,1), bias=True)
        self.end_conv_2_base = nn.Conv2d(in_channels=end_channels, out_channels=out_dim, kernel_size=(1,1), bias=True)
        self.receptive_field = receptive_field
        if self.use_plugin:
            self.lin_test = nn.Linear(3, 32)
            self.fusion = Fusion(32)

    def forward(self, input, feat=None):
        """feed forward of Graph WaveNet.
        Args:
            input (torch.Tensor): input history MTS with shape [B, L, N, C].
            His (torch.Tensor): the output of TSFormer of the last patch (segment) with shape [B, N, d].
        Returns:
            torch.Tensor: prediction with shape [B, N, L]
        """
        if self.use_plugin:
            input = input.transpose(1, 2)
            x_t1 = self.lin_test(input)
            input = self.fusion(feat, x_t1)
            input = input.transpose(1, 3)
        else:
            input = input.transpose(1, 2).transpose(1, 3)
        # feed forward
        input = nn.functional.pad(input, (1, 0, 0, 0))

        input = input[:, :self.in_dim, :, :]
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0


        # calculate the current adaptive adj matrix
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip


            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x,self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]


            x = self.bn[i](x)
        # hidden_states = feat.transpose(1,2)
        # #print("hidden_states", hidden_states.shape)
        # if hidden_states is not None:
        #     hidden_states_t = self.gate_fusion(hidden_states,skip)        # B, N, D
        #     # hidden_states_t = hidden_states_t.transpose(1, 2).unsqueeze(-1)  # B, D, N, 1
        #     # skip = skip+hidden_states_t


        x_plugin = F.relu(skip)
        x_plugin = F.relu(self.end_conv_1_plugin(x_plugin))
        x_plugin = self.end_conv_2_plugin(x_plugin)
        # reshape: [B, P, N, 1] -> [B, N, P]
        x_plugin = x_plugin.squeeze(-1).transpose(1, 2)

        # x_base = F.relu(skip_base)
        # x_base = F.relu(self.end_conv_1_base(x_base))
        # x_base = self.end_conv_2_base(x_base)
        # # reshape: [B, P, N, 1] -> [B, N, P]
        # x_base = x_base.squeeze(-1).transpose(1, 2)
    
        # return x_base, x_plugin
        return x_plugin




