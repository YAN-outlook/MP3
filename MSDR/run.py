# -*- coding: utf-8 -*-
import torch
import numpy as np
import argparse
import time
import util
import copy
from trainer import Trainer
from preprocess.model import linear_transformer
from scipy.sparse.linalg import eigs
import random

DATASET_CONFIGS = {
    'PEMS03': {
        'data_path': 'data/PEMS03/short_term',
        'adjdata': 'data/PEMS03/adj_PEMS03.pkl',
        'in_dim': 1,
        'num_nodes': 358,
        'checkpoint': 'checkpoint_PEMS03_good/testA_best.pth',
    },
    'PEMS04': {
        'data_path': 'data/PEMS04/short_term',
        'adjdata': 'data/PEMS04/adj_PEMS04.pkl',
        'in_dim': 3,
        'num_nodes': 307,
        'checkpoint': 'checkpoint_PEMS04_good/testA_best.pth',
    },
    'PEMS07': {
        'data_path': 'data/PEMS07/short_term',
        'adjdata': 'data/PEMS07/adj_PEMS07.pkl',
        'in_dim': 1,
        'num_nodes': 883,
        'checkpoint': 'checkpoint_PEMS07_good/testA_best.pth',
    },
    'PEMS08': {
        'data_path': 'data/PEMS08/short_term',
        'adjdata': 'data/PEMS08/adj_PEMS08.pkl',
        'in_dim': 3,
        'num_nodes': 170,
        'checkpoint': 'checkpoint_PEMS08_good/testA_best.pth',
    },
    'CA': {
        'data_path': 'data/CA/short_term',
        'adjdata': 'data/CA/adj_pems.npy',
        'in_dim': 3,
        'num_nodes': 9638,
        'checkpoint': 'checkpoint_CA_good/testA_best.pth',
    },
}

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, choices=list(DATASET_CONFIGS.keys()), help='dataset name')
parser.add_argument('--device', type=str, default='cuda:1', help='')
parser.add_argument('--data_path', type=str, default=None, help='data path')
parser.add_argument('--adjdata', type=str, default=None, help='adj data path')
parser.add_argument('--input_length', type=int, default=1248, help='')
parser.add_argument('--output_length', type=int, default=12, help='')
parser.add_argument('--hid_dim', type=int, default=32, help='')
parser.add_argument('--in_dim', type=int, default=None, help='inputs dimension')
parser.add_argument('--num_nodes', type=int, default=None, help='number of nodes')
parser.add_argument('--num_layers', type=int, default=3, help='number of layers')
parser.add_argument('--tau', type=int, default=0.25, help='temperature coefficient')
parser.add_argument('--random_feature_dim', type=int, default=64, help='random feature dimension')
parser.add_argument('--node_dim', type=int, default=32, help='node embedding dimension')
parser.add_argument('--time_dim', type=int, default=32, help='time embedding dimension')
parser.add_argument('--time_num', type=int, default=288, help='time in day')
parser.add_argument('--week_num', type=int, default=7, help='day in week')
parser.add_argument('--use_residual', action='store_true', help='use residual connection')
parser.add_argument('--use_bn', action='store_true', help='use batch normalization')
parser.add_argument('--use_spatial', action='store_true', help='use spatial loss')
parser.add_argument('--use_long', action='store_true', help='use long-term preprocessed features (plugin mode)')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clip')
parser.add_argument('--learning_rate', type=float, default=0.0015, help='learning rate')
parser.add_argument('--milestones', type=list, default=[30,50,70,80], help='optimizer milestones')
parser.add_argument('--patience', type=int, default=30, help='early stopping')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=1.0e-3, help='weight decay rate')
parser.add_argument('--epochs', type=int, default=200, help='')
parser.add_argument('--print_every', type=int, default=50, help='')
parser.add_argument('--save', type=str, default='checkpoint/', help='save path')
parser.add_argument('--checkpoint', type=str, default=None, help='pretrained checkpoint path')
parser.add_argument('--expid', type=int, default=1, help='experiment id')
parser.add_argument('--seed', type=int, default=10, help='random seed')
parser.add_argument('--gamma', type=float, default=0.2, help='learning rate scheduler gamma')
parser.add_argument('--dynamic', type=bool, default=True, help='use dynamic attention')
args = parser.parse_args()

# Auto-fill dataset-specific configs
config = DATASET_CONFIGS[args.dataset]
for key, val in config.items():
    if getattr(args, key) is None:
        setattr(args, key, val)

print(args)


def main():
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Using random seed: {seed}")
    
    args.device = torch.device(args.device)
    adj_mx = util.load_pickle(args.adjdata)
    dataloader = util.load_dataset(args)
    scaler = dataloader['scaler']
    supports = [torch.tensor(adj_mx).to(args.device)]
    supports_1 = torch.tensor(adj_mx, dtype=torch.float32).to(args.device)
    N = args.num_nodes
    edge_indices = torch.nonzero(supports[0] > 0)
    values = torch.ones(edge_indices.size(0), device=args.device)
    adj_sparse = torch.sparse_coo_tensor(
        edge_indices.t(),
        values,
        (N, N),
        device=args.device
    )

    trainer = Trainer(args, scaler, supports_1, edge_indices)

    if args.use_long:
        feat_extractor = linear_transformer(args.input_length, args.output_length, args.in_dim,
                                             args.num_nodes, args.hid_dim, adj_sparse)
        feat_extractor.to(args.device)
        feat_extractor.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
        for param in feat_extractor.parameters():
            param.requires_grad = False
        feat_extractor.eval()
    
    print("start training...", flush=True)
    his_loss = []
    test_time = []
    val_time = []
    train_time = []
    batches_seen = 0
    wait = 0
    min_val_loss = np.inf
    
    for i in range(1, args.epochs+1):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        now_epoch = i
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(args.device)
            trainx = trainx.transpose(1, 2)
            trainy = torch.Tensor(y).to(args.device)
            trainy = trainy.transpose(1, 2)
            shorty = trainy
            y_true = shorty[..., [0]].unsqueeze(-1)
            
            if args.use_long:
                feat = []
                for j in range(trainx.shape[0]):
                    with torch.no_grad():
                        _, feat_sample = feat_extractor(trainx[[j],:,:,:], None)
                    feat.append(feat_sample)
                feat = torch.cat(feat, dim=0)
                metrics = trainer.train(trainx[:,:,-12:,:], trainy[:,:,:,0], y_true, batches_seen, feat)
            else:
                metrics = trainer.train(trainx[:,:,-12:,:], trainy[:,:,:,0], y_true, batches_seen)
            batches_seen += 1
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            t2 = time.time()
            train_time.append(t2-t1)
            
            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)

        trainer.scheduler.step()
        
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(args.device)
            testx = testx.transpose(1, 2)
            testy = torch.Tensor(y).to(args.device)
            testy = testy.transpose(1, 2)
            
            if args.use_long:
                feat = []
                for j in range(testx.shape[0]):
                    with torch.no_grad():
                        _, feat_sample = feat_extractor(testx[[j],:,:,:], None)
                    feat.append(feat_sample)
                feat = torch.cat(feat, dim=0)
                metrics = trainer.eval(testx[:,:,-12:,:], testy[:,:,:,0], feat)
            else:
                metrics = trainer.eval(testx[:,:,-12:,:], testy[:,:,:,0])
                
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Validation Inference Time: {:.4f} secs'
        print(log.format(now_epoch, (s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)

        if mvalid_loss < min_val_loss:
            wait = 0
            min_val_loss = mvalid_loss
            best_epoch = now_epoch
            best_state_dict = copy.deepcopy(trainer.model.state_dict())
        else:
            wait += 1
            if wait >= args.patience:
                break
        
        log = 'Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}'
        print(log.format(mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse), flush=True)
        print("best_epoch", best_epoch)

    trainer.model.load_state_dict(best_state_dict)
    test_loss = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': [], '10': [], '11': []}
    test_mape = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': [], '10': [], '11': []}
    test_rmse = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': [], '10': [], '11': []}
    s1 = time.time()
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(args.device)
        testx = testx.transpose(1, 2)
        testy = torch.Tensor(y).to(args.device)
        testy = testy.transpose(1, 2)
        
        if args.use_long:
            feat = []
            for j in range(testx.shape[0]):
                with torch.no_grad():
                    _, feat_sample = feat_extractor(testx[[j],:,:,:], None)
                feat.append(feat_sample)
            feat = torch.cat(feat, dim=0)
            metrics = trainer.eval(testx[:,:,-12:,:], testy[:,:,:,0], feat, flag='horizon')
        else:
            metrics = trainer.eval(testx[:,:,-12:,:], testy[:,:,:,0], flag='horizon')
        for k in range(12):
            test_loss[str(k)].append(metrics[0][k])
            test_mape[str(k)].append(metrics[1][k])
            test_rmse[str(k)].append(metrics[2][k])
    s2 = time.time()
    log = 'Epoch: {:03d}, Test Inference Time: {:.4f} secs'
    print(log.format(best_epoch, (s2-s1)))
    test_time.append(s2-s1)
    amae = []
    amape = []
    armse = []
    for k in range(12):
        amae.append(np.mean(test_loss[str(k)]))
        amape.append(np.mean(test_mape[str(k)]))
        armse.append(np.mean(test_rmse[str(k)]))
        log = 'Model performance for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(k+1, amae[-1], amape[-1], armse[-1]))

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
    batches_seen = 0

if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
