import argparse
import time
import warnings

import torch

import utils
from data.preprocessing import toDataloader

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='FD')
parser.add_argument('--multi_group', type=list, default=[1, 2, 3],
                    help='Input list')  # group<=math.ceil(sqrt(seq_len))
parser.add_argument('--batch', type=int, default=16, help='Dataset batch')
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--emb_size', type=int, default=128)
parser.add_argument('--nhead', type=int, default=8)
parser.add_argument('--emb_size_c', type=int, default=128)
parser.add_argument('--masking_ratio', type=float, default=0.15)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--ratio_highest_attention', type=float, default=0.35)
parser.add_argument('--dropout', type=float, default=0.01)
parser.add_argument('--nhid', type=int, default=128)
parser.add_argument('--nhid_c', type=int, default=128)

args = parser.parse_args()
print(args)


def main():
    prop = utils.get_prop(args)
    prop['multi_group'] = [int(patch_index) for patch_index in prop['multi_group']]
    print('Data loading start...')
    X_train, y_train, X_test, y_test = utils.data_loader(args.dataset)

    prop['batch_true'] = X_train.shape[0]
    print('Data loading complete...')

    print('Data preprocessing start...')
    X_train_task, y_train_task, X_test, y_test = utils.preprocess(X_train, y_train, X_test, y_test, prop)
    print('After standered:', X_train_task.shape, y_train_task.shape, X_test.shape, y_test.shape)

    print('Data preprocessing complete...')

    # classes
    prop['nclasses'] = torch.max(y_train_task).item() + 1
    prop['dataset'], prop['seq_len'], prop['input_size'] = prop['dataset'], X_train_task.shape[1], X_train_task.shape[2]
    prop['device'] = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    # gain dataloader
    dataloader_train, dataloader_test = toDataloader(prop, X_train_task, y_train_task, X_test, y_test)

    print('Initializing model...')
    t = time.time()
    model, criterion, optimizer = utils.initialize_training(prop)
    print('Model intialized...')

    print('Training start...')
    utils.training(model, optimizer, criterion, dataloader_train, dataloader_test, prop)
    t = time.time() - t
    print(f"\nTraining time: {t / prop['epochs']}\n")

    print('Training complete...')


if __name__ == "__main__":
    main()
