""" Matplotlib backend configuration """
import matplotlib
matplotlib.use('PS')  # generate postscript output by default

""" Imports """
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import train_test_split, KFold
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn import preprocessing

import sys
import pandas as pd
import os
import pickle
from model import PainPredModel 

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


""" Custom Dataset """
class VisitSequenceWithLabelDataset(Dataset):
    def __init__(self, seqs_list, labels_list, reverse=True):
        """
        Args:
            seqs (list): list of patients (list) of visits (list) of codes (int) that contains visit sequences
            labels (list): list of labels (int)
            reverse (bool): If true, reverse the order of sequence 
        """

        if len(seqs_list) != len(labels_list):
            raise ValueError("Seqs and Labels have different lengths")

        self.seqs = []
        self.labels = []
        for seq, label in zip(seqs_list, labels_list):
            if reverse:
                sequence = list(reversed(seq))
            else:
                sequence = seq

            self.seqs.append(np.array(sequence, dtype=np.float32))
            self.labels.append(np.expand_dims(label, 0))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.seqs[index], self.labels[index]


""" Custom collate_fn for DataLoader"""
# @profile
def visit_collate_fn(batch):
    """
    DataLoaderIter call - self.collate_fn([self.dataset[i] for i in indices])
    Thus, 'batch' is a list [(seq1, label1), (seq2, label2), ... , (seqN, labelN)]
    where N is minibatch size, seq is a SparseFloatTensor, and label is a LongTensor

    :returns
        seqs
        labels
        lengths
    """
    batch_seq, batch_label = zip(*batch)

    num_features = batch_seq[0].shape[1]
    seq_lengths = list(map(lambda patient_tensor: patient_tensor.shape[0], batch_seq))
    max_length = max(seq_lengths)


    sorted_indices, sorted_lengths = zip(*sorted(enumerate(seq_lengths), key=lambda x: x[1], reverse=True))
    sorted_padded_seqs = []
    sorted_labels = []

    for i in sorted_indices:
        length = batch_seq[i].shape[0]

        if length < max_length:
            padded = np.concatenate(
                (batch_seq[i], np.zeros((max_length - length, num_features), dtype=np.float32)), axis=0)
        else:
            padded = batch_seq[i]

        sorted_padded_seqs.append(padded)
        sorted_labels.append(batch_label[i].flatten())

    seq_tensor = np.stack(sorted_padded_seqs, axis=0)

    return torch.FloatTensor(np.array(seq_tensor)), torch.FloatTensor(np.array(sorted_labels)), list(sorted_lengths), list(sorted_indices) #torch.from_numpy(seq_tensor)




class Model_trainer:
    def __init__(self, params):
        self.params = params
        if self.params['cuda']:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        # self.criterion = nn.BCELoss(reduction='mean')
        self.criterion = nn.MSELoss(reduction='mean')

        if self.params['cuda']:
            self.criterion = self.criterion.cuda()

    def _data_loading(self, train_seqs, train_labels, test_seqs, test_labels):
        if self.params['print']:
            print('===> Loading entire datasets')

            print("     ===> Construct train set")
        train_set = VisitSequenceWithLabelDataset(train_seqs, train_labels)
        if self.params['print']:
            print("     ===> Construct test set")
        test_set = VisitSequenceWithLabelDataset(test_seqs, test_labels)
        att_set = VisitSequenceWithLabelDataset(train_seqs, train_labels)

        self.train_loader = DataLoader(dataset=train_set, batch_size=self.params['batch_size'], shuffle=True,
                                  collate_fn=visit_collate_fn, num_workers=self.params['threads'])
        self.test_loader = DataLoader(dataset=test_set, batch_size=self.params['batch_size'], shuffle=False,
                                 collate_fn=visit_collate_fn, num_workers=self.params['threads'])
        self.att_loader = DataLoader(dataset=att_set, batch_size=1, shuffle=False,
                                 collate_fn=visit_collate_fn, num_workers=self.params['threads'])


    def _save_checkpoint(self, state, filename="checkpoint.pth.tar"):
        print("=> Saving checkpoint")
        torch.save(state, filename)

    def train(self, train_seqs, train_labels, test_seqs, test_labels):
        if self.params['threads'] == -1:
            self.params['threads'] = torch.multiprocessing.cpu_count() - 1 or 1
        print('===> Configuration')
        print(self.params)

        if self.params['cuda']:
            if torch.cuda.is_available():
                if self.params['print']:
                    print('===> {} GPUs are available'.format(torch.cuda.device_count()))
            else:
                raise Exception("No GPU found, please run with --no-cuda")

        # Fix the random seed for reproducibility
        np.random.seed(self.params['seed'])
        torch.manual_seed(self.params['seed'])
        if self.params['cuda']:
            torch.cuda.manual_seed(self.params['seed'])

        # Data loading
        self._data_loading(train_seqs, train_labels, test_seqs, test_labels)
        if self.params['print']:
            print('===> Dataset loaded!')
            # Create model
            print('===> Building a Model')

        self.model = PainPredModel(
            dim_input=self.params['input_dim'],
            dim_emb=self.params['emb_dim'],
            dim_alpha=self.params['rnn_dim'],
            dim_beta=self.params['rnn_dim'],
            dropout_context=self.params['drop_out'],
            dim_output=self.params['output_dim'])
        if self.params['cuda']:
            self.model = self.model.cuda()
        if self.params['print']:
            print(self.model)

        # for name, param in model.named_parameters():
        #    print("{}: {}".format(name, param.size()))
        if self.params['print']:
            print('===> Model built!')
        logFile = '../results/training.log'
        # Optimization
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['lr'],
                                     weight_decay=self.params['L2_norm'])  # , betas=(0.1, 0.001), eps=1e-8
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=params['lr']*0.005, momentum=0.9, weight_decay=params['L2_norm'])

        # best_valid_epoch = 0
        # best_model_wts = copy.deepcopy(self.model.state_dict())

        best_valid_mae = sys.float_info.max
        best_valid_r2 = 0
        best_checkpoint = {}
        #
        train_losses = []
        valid_losses = []


        for ei in range(self.params['epochs']):
            # Train
            train_y_true, train_y_pred, train_loss = self._epoch(self.train_loader, criterion=self.criterion,
                                                           optimizer=optimizer,
                                                           train=True)
            train_losses.append(train_loss)

            valid_y_true, valid_y_pred, valid_loss = self._epoch(self.test_loader, criterion=self.criterion, train=False)
            valid_losses.append(valid_loss)
            r2 = r2_score(valid_y_true, valid_y_pred)
            mae = mean_absolute_error(valid_y_true, valid_y_pred)
            if mae < best_valid_mae:
                best_valid_mae = mae
                best_valid_r2 = r2
                buf = "Epoch {} - Loss train: {:.4f}, test: {:.4f}, r2: {:.4f}, mae: {:.4f}".format(ei, train_loss, valid_loss, r2, mae)
                if self.params['print']:
                    print(buf)

                checkpoint = {
                    "state_dict": self.model.state_dict(),
                    "optimizer": optimizer.state_dict()
                }
                best_checkpoint = checkpoint


        return best_valid_mae, best_valid_r2, buf

    def _epoch(self, loader, criterion, optimizer=None, train=False):
        if train and not optimizer:
            raise AttributeError("Optimizer should be given for training")

        if train:
            self.model.train()
            mode = 'Train'
        else:
            self.model.eval()
            mode = 'Eval'

        losses = AverageMeter()
        labels = []
        outputs = []


        for batch in loader:
            inputs, targets, lengths, sorted_indice = batch
            if self.params['cuda']:
                inputs = inputs.cuda()
                targets = targets.cuda()

            output, att_dict= self.model(inputs, lengths)

            loss = criterion(output, targets)

            # compute gradient and do update step
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if self.params['cuda']:
                loss = loss.cpu()
                output = output.cpu()
                targets = targets.cpu()

            # sort back
            if not train:
                sort_back_indices, _ = zip(*sorted(enumerate(sorted_indice), key=lambda x: x[1]))
                sort_back_indices = list(sort_back_indices)
                output = output[sort_back_indices]
                targets = targets[sort_back_indices]

            outputs.append(output.detach())
            labels.append(targets.detach())

            # record loss
            losses.update(loss.detach().numpy(), inputs.size(0))
            # if train:
            #     break

        return torch.cat(labels, 0), torch.cat(outputs, 0), losses.avg

    def _epoch_att(self, loader):
        with torch.no_grad():
            attention_results = []
            alpha_results = []
            self.model.eval()
            for batch in loader:
                inputs, targets, lengths, sorted_indice = batch
                # if self.params['cuda']:
                #     inputs = inputs.cuda()
                #     targets = targets.cuda()
                output, att_dict= self.model(inputs, lengths)
                for k, v in att_dict.items():
                    att_dict[k] = v.detach()
                num_time = lengths[0]
                num_x = inputs.shape[-1]
                attention_pat = []
                alpha_att_pat = []
                for j in range(num_time):
                    attention_time = []
                    alpha_att_pat.append(att_dict['alpha'][:, j, :].item())
                    for k in range(num_x):
                        beta_j = att_dict['beta'][:, j, :]
                        w_emb_k = torch.unsqueeze(att_dict['w_emb'][:,k], dim=0)
                        alpha_j = att_dict['alpha'][:, j, :]
                        w_out = att_dict['w_out']
                        att_jk = torch.matmul(alpha_j * w_out, torch.transpose(beta_j * w_emb_k, 0, 1))
                        att_jk = torch.squeeze(att_jk).item()
                        x_jk = inputs[0, j, k].item()
                        attention_time.append(att_jk*x_jk)
                    attention_pat.append(attention_time)
                attention_results.append(np.array(attention_pat))
                alpha_results.append(np.array(alpha_att_pat))
            return attention_results, alpha_results

    # def _epoch_att_np(self, loader):
    #     with torch.no_grad():
    #         attention_results = []
    #         self.model.eval()
    #         for batch in loader:
    #             inputs, targets, lengths, sorted_indice = batch
    #             output, att_dict= self.model(inputs, lengths)
    #             for k, v in att_dict.items():
    #                 att_dict[k] = v.detach().numpy()
    #             num_time = lengths[0]
    #             num_x = inputs.shape[-1]
    #             attention_pat = []
    #             for j in range(num_time):
    #                 attention_time = []
    #                 for k in range(num_x):
    #                     beta_j = att_dict['beta'][:, j, :]
    #                     w_emb_k = np.expand_dims(att_dict['w_emb'][:,k], axis=0)
    #                     alpha_j = att_dict['alpha'][:, j, :]
    #                     w_out = att_dict['w_out']
    #                     att_jk = np.matmul(alpha_j * w_out, np.transpose(beta_j * w_emb_k))
    #                     att_jk = np.squeeze(att_jk)
    #                     attention_time.append(att_jk)
    #                 attention_pat.append(attention_time)
    #             attention_results.append(np.array(attention_pat))
    #         return attention_results

def print2file(buf, outFile):
    outfd = open(outFile, 'a')
    outfd.write(buf + '\n')
    outfd.close()


def data_convert(feature_df, label_df):
    seqs = []
    labels = []

    feature_dict = {}
    feature_set = list(feature_df.columns)
    feature_set.remove('record_id')
    feature_set.remove('redcap_event_name')
    for idx, row in feature_df.iterrows():
        feature_dict[(row['record_id'], row['redcap_event_name'])] = row[feature_set].to_numpy()

    timeline = ['baseline_arm_1', 'week_04_arm_1', 'week_08_arm_1', 'week_12_arm_1']
    for idx, row in label_df.iterrows():
        labels.append(row['pain_label'])
        feature_seq = []
        for i in range(0, timeline.index(row['redcap_event_name']) + 1):
            key = (row['record_id'], timeline[i])
            feature_seq.append(feature_dict[key])
        seqs.append(feature_seq)
    return seqs, labels

def normalization_df(feature_train, feature_test, exclude_fea):
    feature_set = list(feature_train.columns)
    include_fea = [x for x in feature_set if x not in exclude_fea]
    mm_scaler = preprocessing.MinMaxScaler()
    X_train_minmax = mm_scaler.fit_transform(feature_train[include_fea])
    X_test_minmax = mm_scaler.transform(feature_test[include_fea])
    X_train_minmax = pd.DataFrame(X_train_minmax, columns=include_fea)
    X_test_minmax = pd.DataFrame(X_test_minmax, columns=include_fea)
    X_train_minmax = pd.concat([feature_train[exclude_fea], X_train_minmax], axis=1)
    X_test_minmax = pd.concat([feature_test[exclude_fea], X_test_minmax], axis=1)
    return X_train_minmax, X_test_minmax

if __name__ == "__main__":
    feature_df = pd.read_csv('../data/format/regression/feature.csv')
    label_df = pd.read_csv('../data/format/regression/label.csv')

    pat_list = feature_df['record_id'].unique()

    # number of splits for cross-validation
    mae_lst_seed = []
    r2_lst_seed = []

    for seed in [123, 321, 45, 65, 52]:
        print(f'---------------------seed: {seed}--------------')

        n_splits = 5
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        mae_lst = []
        r2_lst = []
        for fold, (train_index, test_index) in enumerate(kf.split(pat_list)):
            print(f'Fold {fold + 1}/{n_splits}')
            train_pat = pat_list[train_index]
            test_pat = pat_list[test_index]

            feature_train = feature_df[feature_df['record_id'].isin(train_pat)].reset_index(drop=True)
            label_train = label_df[label_df['record_id'].isin(train_pat)].reset_index(drop=True)
            feature_test = feature_df[feature_df['record_id'].isin(test_pat)].reset_index(drop=True)
            label_test = label_df[label_df['record_id'].isin(test_pat)].reset_index(drop=True)

            feature_train, feature_test = normalization_df(feature_train, feature_test, ['record_id', 'redcap_event_name'])

            train_X, train_Y = data_convert(feature_train, label_train)
            test_X, test_Y = data_convert(feature_test, label_test)

            params = {}
            params['input_dim'] = train_X[0][0].shape[0]
            params['output_dim'] = 1
            params['rnn_dim'] = 32
            params['lr'] = 0.001
            params['batch_size'] = 8
            params['epochs'] = 300
            params['L2_norm'] = 0.6
            params['emb_dim'] = 32
            params['drop_out'] = 0.2
            params['seed'] = seed
            params['threads'] = 0
            params['cuda'] = True
            params['print'] = False
            if not torch.cuda.is_available():
                params['cuda'] = False

            model = Model_trainer(params)
            best_valid_mae, best_valid_r2, buf = model.train(train_X, train_Y, test_X, test_Y)
            mae_lst.append(best_valid_mae)
            r2_lst.append(best_valid_r2)
            print(buf)
            print(f'    best_mae = {best_valid_mae}, best_r2 = {best_valid_r2}')
            print('     ------------------------------------------------------')

        print()
        print(f'    avg_mae_seed:{seed} = {np.mean(np.array(mae_lst))}, std_r2_seed:{seed} = {np.std(np.array(mae_lst))}')
        print(f'    r2_mae_seed:{seed} = {np.mean(np.array(r2_lst))}, std_r2_seed:{seed} = {np.std(np.array(r2_lst))}')
        print()
        mae_lst_seed.append(np.mean(np.array(mae_lst)))
        r2_lst_seed.append(np.mean(np.array(r2_lst)))

    mae_arr = np.array(mae_lst_seed)
    mean_mae_arr = np.mean(mae_arr)

    r2_arr = np.array(r2_lst_seed)
    mean_r2_arr = np.mean(r2_arr)

    print(f'avg mae over all seeds:{mean_mae_arr}')
    print(f'r2 mae over all seeds:{mean_r2_arr}')


