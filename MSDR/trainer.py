import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import metrics
from model import GMSDRModel

class Trainer():
    def __init__(self, args, scaler, supports, edge_indices):
        input_dim = 32 if args.use_long else 1
        self.model_parameter = {
                'cl_decay_steps': 2000,
                'filter_type': 'dual_random_walk',
                'horizon': 12,
                'input_dim': input_dim,
                'max_diffusion_step': 1,
                'num_nodes': args.num_nodes,
                'num_rnn_layers': 2,
                'output_dim': 1,
                'rnn_units': 64,
                'seq_len': 12,
                'pre_k': 7,
                'pre_v': 1,
                'use_curriculum_learning': True,
                'construct_type': 'connectivity',
                'l2lambda': 1.0e-06
            }

        self.l2lambda = 1.0e-06
        self.model = GMSDRModel(device=args.device,
                                adj_mx=supports,
                                use_plugin=args.use_long,
                                **self.model_parameter)
        self.model.to(args.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            eps=1e-8
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=args.milestones, gamma=args.gamma
        )

        self.loss = metrics.masked_mae
        self.scaler = scaler
        self.grad_clip = args.grad_clip
        


    # ===========================================================
    # ===========================================================
    def train(self, input, real_val, inverse_y_true, batches_seen,feat=None):
        """
        """
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(input, real_val, batches_seen, feat)
        

        # inverse transform real
        real = self.scaler.inverse_transform(real_val)
        predict = self.scaler.inverse_transform(output)

        loss_plugin = self.loss(predict, real, 0.0)

        lossl2 = self.model.Loss_l2() * self.l2lambda
        loss = loss_plugin + lossl2

        # backward
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        mape = metrics.masked_mape(predict, real, 0.0).item()
        rmse = metrics.masked_rmse(predict, real, 0.0).item()

        return loss.item(), mape, rmse


    # ===========================================================
    # ===========================================================
    def eval(self, input, real_val, feat=None, flag='overall'):
        """
            y_hat = (1 + Δ) * x_last
        """
        self.model.eval()
        
        with torch.no_grad():
            output = self.model(input, None, None, feat)  
        # #[B,N,T]
        # output=output.permute(2,0,1).contiguous()
        # # inverse transform real y
        real = self.scaler.inverse_transform(real_val)

        # x = self.scaler.inverse_transform(input)
        
        # # input shape: (batch, seq_len, num_nodes, 1)
        # x_last = x[:,:,-1,0].squeeze(-1).squeeze(-1)  # (batch, N)
        # # print(x_last.shape)
        # # print(output.shape)
        # # output: (batch, horizon, N, 1)
        # predict = (1 + output) * x_last
        #[T,B,N]
        # predict=predict.permute(1,2,0).contiguous()
        predict = self.scaler.inverse_transform(output)

        # ---------------- overall ------------------
        if flag == 'overall':
            loss = self.loss(predict, real, 0.0)
            mape = metrics.masked_mape(predict, real, 0.0).item()
            rmse = metrics.masked_rmse(predict, real, 0.0).item()
            return loss.item(), mape, rmse

        # ---------------- horizon ------------------
        elif flag == 'horizon':
            mse, mape, rmse = [], [], []
            for i in range(12):
                mse.append(metrics.masked_mae(predict[..., i], real[..., i], 0.0).item())
                mape.append(metrics.masked_mape(predict[..., i], real[..., i], 0.0).item())
                rmse.append(metrics.masked_rmse(predict[..., i], real[..., i], 0.0).item())
            return mse, mape, rmse

            
