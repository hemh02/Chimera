import argparse
import torch
import pytorch_lightning as pl
from src.dataset import get_loader, get_eval_loader
from src.models import Chimera
from src.utils import get_abs_path
from src.dataset import load_json
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import pandas as pd
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import time
from sklearn.metrics import ndcg_score


def deal_batch(res_src, res_label):
    src_embedding = []
    for batch in res_src:
        batch_embedding = []
        for seq in batch:
            seq_embedding = [int(word) for word in seq]
            batch_embedding.append(seq_embedding)
        src_embedding.append(batch_embedding)    

    src_embedding = np.array(src_embedding)
    src_embedding = torch.from_numpy(src_embedding).float()

    label_embedding = []
    for batch in res_label:
        batch_embedding = [int(word) for word in batch]
        label_embedding.append(batch_embedding)

    label_embedding = np.array(label_embedding)
    label_embedding = torch.from_numpy(label_embedding).float()

    src_embedding = src_embedding.permute(2, 0, 1)
    label_embedding = label_embedding.transpose(0, 1)
    return src_embedding, label_embedding



def gen_line(src, label):
    sent = '0:'
    for it in src: sent += str(int(it.item())) + ' '
    for it in label: sent += str(int(it.item())) + ' '
    sent = sent[:-1] + '\n'
    return sent



class MyEarlyStopping(EarlyStopping):
    def __init__(self, monitor: str = 'val_loss', patience: int = 3, verbose: bool = False, mode: str = 'min'):
        super().__init__(monitor=monitor, patience=patience, verbose=verbose, mode=mode)

    def on_validation_end(self, trainer, pl_module):
        # override this to disable early stopping at the end of val loop
        pass

    def on_epoch_end(self, trainer, pl_module):
        self._run_early_stopping_check(trainer, pl_module)
        # if pl_module.current_epoch == 69 and not pl_module.restart_epoch:   
        #     print("Restart triggered. Entering second training phase.") 
        #     self.wait_count = 0
        #     self.stopped_epoch = 0
        #     trainer.should_stop = False
        #     pl_module.optimizer.load_state_dict(pl_module.initial_optimizer_state_dict)
        #     pl_module.restart_epoch = pl_module.current_epoch + 1        
        # elif trainer.should_stop:
        #     if not pl_module.restart_epoch:
        #         print("Early restart triggered. Entering second training phase.")
        #         self.wait_count = 0
        #         self.stopped_epoch = 0
        #         trainer.should_stop = False
        #         pl_module.optimizer.load_state_dict(pl_module.initial_optimizer_state_dict)
        #         pl_module.restart_epoch = pl_module.current_epoch + 1
        #     else:
        #         print("Early stopping triggered. Training will be stopped.")




def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hard_device", default='cuda', type=str, help="cpu or cuda")
    parser.add_argument("--gpu_index", default=0, type=int, help='gpu index, one of [0,1,2,3,...]')
    parser.add_argument("--load_checkpoint", nargs='?', const=True, default=False, type=str2bool,
                        help="one of [t,f]")
    parser.add_argument('--model_save_path', default='checkpoint', type=str)
    parser.add_argument('--epochs', default=10, type=int, help='epochs')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--warmup_epochs', default=8, type=int, help='warmup')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--accumulate_grad_batches',
                        default=16,
                        type=int,
                        help='grad_batches')
    parser.add_argument('--mode', default='train', type=str,
                        help='one of [train, test, preproc]')
    parser.add_argument('--threshold', default=0.9, type=float,
                        help='inference threshold')
    parser.add_argument('--topk', default=1, type=int,
                        help='rca threshold')
    parser.add_argument('--dataset', default='BGL', type=str,
                        help='dataset name')

    arguments = parser.parse_args()
    if arguments.hard_device == 'cpu':
        arguments.device = torch.device(arguments.hard_device)
    else:
        arguments.device = torch.device(f'cuda:{arguments.gpu_index}')
    print(arguments)
    return arguments


def main():
    args = parse_args()
    embedding_dict = load_json('data/' + args.dataset + '/emd_dict.json')
    model = Chimera(args, embedding_dict).to(args.hard_device)


    train_loader = get_loader('data/' + args.dataset + '/n_train.txt',
                            'data/' + args.dataset + '/an_train.txt',
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        num_workers=4)

    valid_loader = get_loader('data/' + args.dataset + '/n_dev.txt',
                              'data/' + args.dataset + '/an_dev.txt',
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        num_workers=4)
      
    test_loader = get_loader('data/' + args.dataset + '/n_test.txt',
                             'data/' + args.dataset + '/n_test.txt',
                                       batch_size=args.batch_size,
                                       shuffle=False,
                                       num_workers=4)
    

    n_eval_loader = get_loader('data/' + args.dataset + '/n_test.txt',
                             'data/' + args.dataset + '/n_test.txt',
                                       batch_size=args.batch_size,
                                       shuffle=False,
                                       num_workers=4)
    
    an_eval_loader = get_loader('data/' + args.dataset + '/an_test.txt',
                             'data/' + args.dataset + '/an_test.txt',
                                       batch_size=args.batch_size,
                                       shuffle=False,
                                       num_workers=4)


    early_stop_callback = MyEarlyStopping(monitor = 'epoch_loss', patience = 10, mode = 'min')
    trainer = pl.Trainer(max_epochs=args.epochs,
                         gpus=None if args.hard_device == 'cpu' else [args.gpu_index],
                         accumulate_grad_batches=args.accumulate_grad_batches,  
                         progress_bar_refresh_rate = 0,
                         callbacks = [early_stop_callback])
    

    if args.load_checkpoint:
        model.load_state_dict(torch.load(get_abs_path('checkpoint', f'{model.__class__.__name__}_model.bin'),
                                         map_location=args.hard_device))    
        
    if args.mode == 'eval':
        with torch.no_grad():
            model.eval()
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            acc = 0
            pos = 0
            nums = 0

            start = time.time()

            for batch in n_eval_loader:
                _, pair = model(batch)
                out, score, ad_label, rca_label = pair
                for i in range(len(out)):
                    if out[i][1].item() >= out[i][0].item(): fp += 1
                    else: tn += 1

            for batch in an_eval_loader:
                _, pair = model(batch)
                out, score, ad_label, rca_label = pair
                for i in range(len(out)):
                    if out[i][1].item() >= out[i][0].item(): tp += 1
                    else: fn += 1

            print("tp:", tp)
            print("fn: ", fn)
            print("fp: ", fp) 
            print("tn: ", tn) 


            if tp != 0:
                p = tp / (tp + fp)
                r = tp / (tp + fn)
                f1 = 2 * p * r / (p + r)
                acc = (tp + tn) / (tp + tn + fp + fn)
            else:
                p = 0
                r = 0
                f1 = 0

            print("P:", p)
            print("R: ", r)
            print("F1: ", f1) 
            print("Acc: ", acc) 
            print()


            maps = 0
            for topk in range(1, 6):
                pos = 0
                nums = 0

                ndcg = 0
                counts = 0

                pr = 0
                faults = 0

                for batch in an_eval_loader:
                    _, pair = model(batch)
                    out, score, ad_label, rca_label = pair
                    for i in range(len(out)):
                        if torch.sum(rca_label[i], dim = -1).item() > 0: faults += 1
                        if out[i][1].item() >= out[i][0].item(): 
                            candidates = torch.topk(score[i], topk, dim = -1).indices
                            target = rca_label[i].unsqueeze(0)
                            res = target[torch.arange(target.size(0)).unsqueeze(1), candidates.unsqueeze(0)]
                            if torch.sum(res[0], dim = -1).item() > 0: 
                                pos  += 1
                                pr   += torch.sum(res[0], dim = -1).item() / min(topk, torch.sum(target[0], dim = -1).item()) 
                        nums += 1
                    ndcg += ndcg_score(rca_label.cpu(), score.cpu(), k = topk)
                    counts += 1
                print("HR" + str(topk) + ": ", pos / nums)
                print("NDCG" + str(topk) + ": ", ndcg / counts)
                print("PR" + str(topk) + ": ", pr / faults)
                maps += pr / faults
                print("MAP" + str(topk) + ": ", maps / topk)
                print()

            mrr = 0
            faults = 0
            for batch in an_eval_loader:
                _, pair = model(batch)
                out, score, ad_label, rca_label = pair
                for i in range(len(out)):
                    if torch.sum(rca_label[i], dim = -1).item() > 0: faults += 1
                    if out[i][1].item() >= out[i][0].item(): 
                        candidates = torch.topk(score[i], 20, dim = -1).indices
                        target = rca_label[i].unsqueeze(0)
                        res = target[torch.arange(target.size(0)).unsqueeze(1), candidates.unsqueeze(0)]
                        if torch.sum(res[0], dim = -1).item() > 0: 
                            mrr += 1 / ((res[0] > 0).nonzero(as_tuple=True)[0][0].item() + 1)
            print("MRR: ", mrr / faults)
            print()


            end = time.time()
            print('inference time: ', end - start)



    elif args.mode == 'train':
        start = time.time()
        trainer.fit(model, train_loader, valid_loader)
        end = time.time()
        print('training time: ', end - start)
        model.load_state_dict(
            torch.load(get_abs_path('checkpoint', f'{model.__class__.__name__}_model.bin'), map_location=args.hard_device))
    else:
        trainer.test(model, test_loader)


if __name__ == '__main__':
    main()
