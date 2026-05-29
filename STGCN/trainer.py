import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import metrics
from model import STGCN

class Trainer():
    def __init__(self, args, scaler, supports, edge_indices):
        dim_in = 32 if args.use_long else args.in_dim
        self.model = STGCN(
        device=args.device,
        dim_in=dim_in,
        dim_out=1,
        Ks=3,
        Kt=3,
        num_nodes=args.num_nodes,
        G=supports,
        blocks1=[64, 32, 128],
        drop_prob=0,
        outputl_ks=3,
        use_plugin=args.use_long
        )
        self.model.to(args.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, eps=1e-8)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.milestones, gamma=args.gamma, verbose=False)
        self.loss = metrics.masked_mae
        self.scaler = scaler
        self.use_spatial = args.use_spatial
        self.grad_clip = args.grad_clip
        


    def train(self, input, real_val, feat=None):
        self.model.train()
        self.optimizer.zero_grad()
        
        output_plugin = self.model(input, feat)

        # inverse transform
        real = self.scaler.inverse_transform(real_val)
        # predict_base = self.scaler.inverse_transform(output_base)
        predict_plugin = self.scaler.inverse_transform(output_plugin)

        loss_plugin = self.loss(predict_plugin, real, 0.0)
        # loss_base = self.loss(predict_base, real, 0.0)

        # align_loss = torch.nn.functional.mse_loss(predict_base, predict_plugin)

        λ_align = 0.3
        λ_base = 0.6
        loss = loss_plugin
        
        # backward
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        mape = metrics.masked_mape(predict_plugin, real, 0.0).item()
        rmse = metrics.masked_rmse(predict_plugin, real, 0.0).item()

        return loss.item(), mape, rmse
    def eval(self, input, real_val, feat=None, flag='overall'):
        if flag=='overall':
            self.model.eval()
            output= self.model(input, feat)
            real = self.scaler.inverse_transform(real_val)
            predict = self.scaler.inverse_transform(output)
            loss = self.loss(predict, real, 0.0)
            mape = metrics.masked_mape(predict,real,0.0).item()
            rmse = metrics.masked_rmse(predict,real,0.0).item()
            return loss.item(), mape, rmse
        elif flag=='horizon':
            self.model.eval()
            output= self.model(input, feat)
            real = self.scaler.inverse_transform(real_val)
            predict = self.scaler.inverse_transform(output)
            loss = []
            mape = []
            rmse = []
            for i in range(12):
                loss.append(self.loss(predict[..., i], real[..., i], 0.0).item())
                mape.append(metrics.masked_mape(predict[..., i], real[..., i], 0.0).item())
                rmse.append(metrics.masked_rmse(predict[..., i], real[..., i], 0.0).item())
            return loss, mape, rmse
    # def eval(self, input, real_val, feat=None, flag='overall'):
    #     self.model.eval()
    #     with torch.no_grad():
    #         output_base, output_plugin = self.model(input, feat)

    #     real = self.scaler.inverse_transform(real_val)
    #     predict_base = self.scaler.inverse_transform(output_base)
    #     predict_plugin = self.scaler.inverse_transform(output_plugin)

    #     if flag == 'overall':
    #         loss = self.loss(predict_plugin, real, 0.0)
    #         mape = metrics.masked_mape(predict_plugin, real, 0.0).item()
    #         rmse = metrics.masked_rmse(predict_plugin, real, 0.0).item()
    #         return loss.item(), mape, rmse

    #     elif flag == 'horizon':
    #         loss, mape, rmse = [], [], []
    #             loss.append(self.loss(predict_plugin[..., i], real[..., i], 0.0).item())
    #             mape.append(metrics.masked_mape(predict_plugin[..., i], real[..., i], 0.0).item())
    #             rmse.append(metrics.masked_rmse(predict_plugin[..., i], real[..., i], 0.0).item())
    #         return loss, mape, rmse
