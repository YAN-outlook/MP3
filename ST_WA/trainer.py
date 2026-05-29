import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import metrics
from model import STWA

class Trainer():
    def __init__(self, args, scaler, supports, edge_indices):
        dim_in = 32 if args.use_long else 1
        self.model = STWA(
            device=args.device,
            num_nodes=args.num_nodes,
            out_dim=1,
            channels=16,
            dynamic=True,
            horizon=12,
            lag=12,
            memory_size=16,
            supports=supports,
            dim_in=dim_in,
            use_plugin=args.use_long
        )       
        self.model.to(args.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, eps=1e-8)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.milestones, gamma=args.gamma, verbose=False)
        self.loss = torch.nn.SmoothL1Loss()
        self.test_loss=metrics.masked_mae
        self.scaler = scaler
        self.use_spatial = args.use_spatial
        self.grad_clip = args.grad_clip
        self.dynamic = args.dynamic

    def train(self, input, real_val, feat=None):
        self.model.train()
        self.optimizer.zero_grad()
        if not self.model.use_plugin:
            input = input[:,:,:,0].squeeze(-1)
        
        output_plugin, input_o = self.model(input, feat)

        # inverse transform
        real = self.scaler.inverse_transform(real_val)
        # predict_base = self.scaler.inverse_transform(output_base)
        predict_plugin = self.scaler.inverse_transform(output_plugin).transpose(2, 1).squeeze(-1)
        # print(predict_plugin.shape)
        # print(real.shape)
        loss_plugin = self.loss(predict_plugin, real)
        # loss_base = self.loss(predict_base, real, 0.0)
        if self.dynamic:
            mu = []
            logvar = []
            for layer in self.model.layers:
                mu.append(layer.mu)
                logvar.append(layer.logvar)
            logvar = torch.stack(logvar)
            mu = torch.stack(mu)

            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            mu = self.model.mu_estimator(input_o)
            logvar = self.model.logvar_estimator(input_o)

            data_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            KLD = KLD + data_KLD

        else:
            KLD = torch.Tensor([0.]).to(self.device)

        loss = loss_plugin + 0.0005 * KLD
        # align_loss = torch.nn.functional.mse_loss(predict_base, predict_plugin)

        
        # backward
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        mse =metrics.masked_mae(predict_plugin, real, 0.0).item()
        mape = metrics.masked_mape(predict_plugin, real, 0.0).item()
        rmse = metrics.masked_rmse(predict_plugin, real, 0.0).item()

        return loss.item(), mape, rmse
    def eval(self, input, real_val, feat=None, flag='overall'):
        if not self.model.use_plugin:
            input = input[:,:,:,0].squeeze(-1)
        if flag=='overall':
            self.model.eval()
            output, _ = self.model(input, feat)
            real = self.scaler.inverse_transform(real_val)
            predict = self.scaler.inverse_transform(output).transpose(2, 1).squeeze(-1)
            loss = self.loss(predict, real)
            mse = metrics.masked_mae(predict, real, 0.0).item()
            mape = metrics.masked_mape(predict,real,0.0).item()
            rmse = metrics.masked_rmse(predict,real,0.0).item()
            return  mse, mape, rmse
        elif flag=='horizon':
            self.model.eval()
            output, _ = self.model(input, feat)
            real = self.scaler.inverse_transform(real_val)
            predict = self.scaler.inverse_transform(output).transpose(2, 1).squeeze(-1)
            mse = []
            mape = []
            rmse = []
            for i in range(12):
                mse.append(metrics.masked_mae(predict[..., i], real[..., i], 0.0).item())
                mape.append(metrics.masked_mape(predict[..., i], real[..., i], 0.0).item())
                rmse.append(metrics.masked_rmse(predict[..., i], real[..., i], 0.0).item())
            return mse, mape, rmse
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
