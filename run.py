import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import tqdm
import os
import csv
import datetime
import time
from tensorflow import keras
from model import APMGR


class Run():
    def __init__(self, config):
        self.use_cuda = config['use_cuda']
        self.base_model = config['base_model']
        self.root = config['root']
        self.ratio = config['ratio']
        self.task = config['task']
        self.seed = config['seed']
        self.results_csv = './result/' +config['mid'] + '/' + config['results_csv']

        self.variant_name = config.get('variant_name', 'APMGR')

        self.num_prototypes = config.get('num_prototypes')
        self.use_prototype = bool(config.get('use_prototype', 1))
        self.use_personalized = bool(config.get('use_personalized', 1))
        self.use_dynamicDelta = config['use_dynamicDelta']
        self.tgt_gating = config.get('tgt_gating', 1)

        self.save_pretrained = config.get('save_pretrained', 1)
        self.load_pretrained = config.get('load_pretrained', 1)

        # Dataset configurations
        self.src = config['src_tgt_pairs'][self.task]['src']
        self.tgt = config['src_tgt_pairs'][self.task]['tgt']
        self.uid_all = config['src_tgt_pairs'][self.task]['uid']
        self.iid_all = config['src_tgt_pairs'][self.task]['iid']

        self.batchsize_src = config['src_tgt_pairs'][self.task]['batchsize_src']
        self.batchsize_tgt = config['src_tgt_pairs'][self.task]['batchsize_tgt']
        self.batchsize_meta = config['src_tgt_pairs'][self.task]['batchsize_meta']
        self.batchsize_test = config['src_tgt_pairs'][self.task]['batchsize_test']

        # Training configurations
        self.epoch = config['epoch']
        self.emb_dim = config['emb_dim']
        self.meta_dim = config['meta_dim']
        self.num_fields = config['num_fields']
        self.lr = config['lr']
        self.wd = config['wd']

        # Paths
        self.ratio_str = str(int(self.ratio[0] * 10)) + '_' + str(int(self.ratio[1] * 10))
        self.input_root = self.root + 'ready/_' + self.ratio_str + '/tgt_' + self.tgt + '_src_' + self.src
        self.pretrained_path = config['pretrained_path']

        self.src_path = self.input_root + '/train_src.csv'
        self.tgt_path = self.input_root + '/train_tgt.csv'
        self.meta_path = self.input_root + '/train_meta.csv'
        self.test_path = self.input_root + '/test.csv'
        self.training_time_seconds = 0

        model_id_str_src = f"task{self.task}_{self.base_model}_{self.seed}"
        self.pretrained_src_path = os.path.join(self.pretrained_path, f"pretrained_{model_id_str_src}_src.pth")

        model_id_str_tgt = f"task{self.task}_{self.base_model}_{self.seed}_ratio{self.ratio_str}"
        self.pretrained_tgt_path = os.path.join(self.pretrained_path, f"pretrained_{model_id_str_tgt}_tgt.pth")

        self.results = {'cross_mae': 10, 'cross_rmse': 10}

    def seq_extractor(self, x):
        x = x.rstrip(']').lstrip('[').split(', ')
        for i in range(len(x)):
            try:
                x[i] = int(x[i])
            except:
                x[i] = self.iid_all
        return np.array(x)

    def read_log_data(self, path, batchsize, history=False):
        if not history:
            cols = ['uid', 'iid', 'y']
            x_col = ['uid', 'iid']
            y_col = ['y']
            data = pd.read_csv(path, header=None)
            data.columns = cols
            X = torch.tensor(data[x_col].values, dtype=torch.long)
            y = torch.tensor(data[y_col].values, dtype=torch.long)
            if self.use_cuda:
                X = X.cuda()
                y = y.cuda()
            dataset = TensorDataset(X, y)
            data_iter = DataLoader(dataset, batchsize, shuffle=True)
            return data_iter
        else:
            data = pd.read_csv(path, header=None)
            cols = ['uid', 'iid', 'y', 'pos_seq']
            x_col = ['uid', 'iid']
            y_col = ['y']
            data.columns = cols
            pos_seq = keras.preprocessing.sequence.pad_sequences(
                data.pos_seq.map(self.seq_extractor),
                maxlen=20,
                padding='post',
                value=self.iid_all
            )
            pos_seq = torch.tensor(pos_seq, dtype=torch.long)
            id_fea = torch.tensor(data[x_col].values, dtype=torch.long)
            X = torch.cat([id_fea, pos_seq], dim=1)
            y = torch.tensor(data[y_col].values, dtype=torch.long)
            if self.use_cuda:
                X = X.cuda()
                y = y.cuda()
            dataset = TensorDataset(X, y)
            data_iter = DataLoader(dataset, batchsize, shuffle=True)
            return data_iter

    def get_data(self):
        print('========Reading data========')
        data_src = self.read_log_data(self.src_path, self.batchsize_src)
        print(f'src {len(data_src.dataset)}')
        data_tgt = self.read_log_data(self.tgt_path, self.batchsize_tgt)
        print(f'tgt {len(data_tgt.dataset)}')
        data_meta = self.read_log_data(self.meta_path, self.batchsize_meta, history=True)
        print(f'meta {len(data_meta.dataset)}')
        data_test = self.read_log_data(self.test_path, self.batchsize_test, history=True)
        return data_src, data_tgt, data_meta, data_test

    def get_model(self):
        model = APMGR(
            uid_all=self.uid_all,
            iid_all=self.iid_all,
            emb_dim=self.emb_dim,
            meta_dim=self.meta_dim,
            num_prototypes=self.num_prototypes,
            base_model=self.base_model,
            use_personalized=self.use_personalized,
            use_prototype=self.use_prototype,
            tgt_gating=self.tgt_gating,
            use_dynamicDelta=self.use_dynamicDelta
        )


        return model.cuda() if self.use_cuda else model

    def get_optimizer(self, model):
        optimizer_src = torch.optim.Adam(params=model.src_model.parameters(), lr=self.lr, weight_decay=self.wd)
        optimizer_tgt = torch.optim.Adam(params=model.tgt_model.parameters(), lr=self.lr, weight_decay=self.wd)

        cross_params = []

        if self.use_personalized and model.personalized_bridge is not None:
            cross_params += list(model.personalized_bridge.parameters())

        if self.use_prototype and model.prototype_learner is not None:
            cross_params += list(model.prototype_learner.parameters())

        optimizer_cross = torch.optim.Adam(params=cross_params, lr=self.lr , weight_decay=self.wd)
        return optimizer_src, optimizer_tgt, optimizer_cross

    def eval_mae(self, model, data_loader, stage):
        model.eval()
        targets, predicts = list(), list()
        loss = torch.nn.L1Loss()
        mse_loss = torch.nn.MSELoss()
        with torch.no_grad():
            for X, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0, leave=False):
                pred = model(X, stage)
                if isinstance(pred, tuple):
                    pred = pred[0]
                targets.extend(y.squeeze(1).tolist())
                predicts.extend(pred.tolist())
        targets = torch.tensor(targets).float()
        predicts = torch.tensor(predicts)
        return loss(targets, predicts).item(), torch.sqrt(mse_loss(targets, predicts)).item()

    def train(self, data_loader, model, criterion, optimizer, epoch, stage):
        print(f'Training Epoch {epoch + 1}:')
        model.train()
        total_loss = 0
        num_batches = 0

        for X, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            pred = model(X, stage)
            if isinstance(pred, tuple):
                pred = pred[0]

            loss_main = criterion(pred, y.squeeze().float())

            total_loss_batch = loss_main

            model.zero_grad()
            total_loss_batch.backward()
            optimizer.step()

            total_loss += total_loss_batch.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f'  Average Loss: {avg_loss:.4f}')

    def update_results(self, mae, rmse, phase):
        if mae < self.results[phase + '_mae']:
            self.results[phase + '_mae'] = mae
        if rmse < self.results[phase + '_rmse']:
            self.results[phase + '_rmse'] = rmse

    def APMGR(self, model, data_src, data_tgt, data_meta, data_test, criterion,
                    optimizer_src, optimizer_tgt, optimizer_cross):
        src_loaded = False
        tgt_loaded = False
        print('\n[*] Checking Pretrained Weights (Optimized)...')

        if self.load_pretrained and os.path.exists(self.pretrained_src_path):
            try:
                model.src_model.load_state_dict(torch.load(self.pretrained_src_path))
                print(f'[+] Loaded Source Model: {os.path.basename(self.pretrained_src_path)}')
                src_loaded = True
            except Exception as e:
                print(f'[!] Source model load failed ({APMGR}). Will re-train.')
        else:
            print(f'[-] Source Model missing: {os.path.basename(self.pretrained_src_path)}')

        if self.load_pretrained and os.path.exists(self.pretrained_tgt_path):
            try:
                model.tgt_model.load_state_dict(torch.load(self.pretrained_tgt_path))
                print(f'[+] Loaded Target Model: {os.path.basename(self.pretrained_tgt_path)}')
                tgt_loaded = True
            except Exception as e:
                print(f'[!] Target model load failed ({APMGR}). Will re-train.')
        else:
            print(f'[-] Target Model missing: {os.path.basename(self.pretrained_tgt_path)}')

        # Pre-training phase
        if not src_loaded:
            print('\n' + '=' * 70)
            print('===== Source Domain Pretraining =====')
            for i in range(self.epoch):
                self.train(data_src, model, criterion, optimizer_src, i, stage='train_src')
            if self.save_pretrained:
                os.makedirs(self.pretrained_path, exist_ok=True)
                torch.save(model.src_model.state_dict(), self.pretrained_src_path)
                print(f'[Saved] Source model to {self.pretrained_src_path}')
        else:
            print('[*] Skipped Source Training (Loaded).')

        if not tgt_loaded:
            print('\n' + '=' * 70)
            print('===== Target Domain Pretraining =====')
            for i in range(self.epoch):
                self.train(data_tgt, model, criterion, optimizer_tgt, i, stage='train_tgt')
            if self.save_pretrained:
                os.makedirs(self.pretrained_path, exist_ok=True)
                torch.save(model.tgt_model.state_dict(), self.pretrained_tgt_path)
                print(f'[Saved] Target model to {self.pretrained_tgt_path}')
        else:
            print('[*] Skipped Target Training (Loaded).')

        # Cross-domain training with optimized single-scale

        print('\n' + '=' * 70)
        print(f'[*] Cross-Domain Training ({self.variant_name})')
        print('=' * 70)

        best_mae = float('inf')
        best_epoch = 0
        training_start_time = time.time()
        for i in range(self.epoch):
            print(f'\nEpoch {i + 1}/{self.epoch}')
            self.train(data_meta, model, criterion, optimizer_cross, i, stage='train_cross')
            mae, rmse = self.eval_mae(model, data_test, stage='test_cross')
            self.update_results(mae, rmse, 'cross')

            if mae < best_mae:
                best_mae = mae
            print(f'MAE: {mae:.4f} | RMSE: {rmse:.4f}')
        training_end_time = time.time()
        self.training_time_seconds = training_end_time - training_start_time

        print(f'\n Results: MAE {self.results["cross_mae"]:.4f}, RMSE {self.results["cross_rmse"]:.4f}')

    def save_results_to_csv(self):
        """Save results with optimized single-scale configuration"""
        csv_file = self.results_csv
        model_name = f"CDR_Optimized_{self.base_model}_task{self.task}_{self.ratio_str}_{self.seed}"

        # Add configuration info to model name
        model_name += f"_proto{self.num_prototypes}"

        best_mae = self.results["cross_mae"]
        best_rmse = self.results["cross_rmse"]
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        file_exists = os.path.exists(csv_file)

        try:
            with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['Variant', 'Model Name', 'MAE', 'RMSE','Training_Time_Seconds', 'Time',
                                     'Meta_Dim', 'Configuration'])

                config_info = f"Prototypes:{self.num_prototypes}"
                writer.writerow([self.variant_name, model_name, best_mae, best_rmse,self.training_time_seconds,
                                 current_time, self.meta_dim, config_info])

            print(f"[LOG] Results saved to {csv_file}")

        except Exception as e:
            print(f"[!] Error saving to CSV: {APMGR}")

    def main(self):
        model = self.get_model()
        data_src, data_tgt, data_meta, data_test = self.get_data()
        optimizer_src, optimizer_tgt, optimizer_cross = self.get_optimizer(model)
        criterion = torch.nn.MSELoss()

        self.APMGR(model, data_src, data_tgt, data_meta, data_test,
                         criterion, optimizer_src, optimizer_tgt, optimizer_cross)

        self.save_results_to_csv()
