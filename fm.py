import tqdm
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from model.criteo import CriteoDataset
from model.twitter import TwitterDataset
from model.FM import FactorizationMachineModel, DeepFactorizationMachineModel, FieldWeightedFactorizationMachineModel, DeepFieldWeightedFactorizationMachineModel
import numpy as np
from sklearn import metrics


def compute_prauc(gt, pred):
    prec, recall, thresh = metrics.precision_recall_curve(gt, pred)
    prauc = metrics.auc(recall, prec)
    return prauc


def calculate_ctr(gt):
    positive = len([x for x in gt if x == 1])
    ctr = positive / float(len(gt))
    return ctr


def compute_rce(gt, pred):
    cross_entropy = metrics.log_loss(gt, pred)
    data_ctr = calculate_ctr(gt)
    strawman_cross_entropy = metrics.log_loss(gt, [data_ctr for _ in range(len(gt))])
    return (1.0 - cross_entropy / strawman_cross_entropy) * 100.0


def cross_entropy(targets, predictions):
    N = predictions.shape[0]
    ce = -np.sum(targets * np.log(predictions)) / N
    return ce


def train(model, optimizer, data_loader, criterion, device, log_interval=100):
    model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (fields, target) in enumerate(tk0):
        fields, target = fields.to(device), target.to(device)
        y = model(fields)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0


def test(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    total_loss = 0
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            loss = criterion(y, target.float())
            total_loss += loss.item()

            predicts.extend(y.cpu().data.numpy().astype("float64"))
            targets.extend(target.cpu().data.numpy().astype("float64"))

    return total_loss / len(data_loader), roc_auc_score(targets, predicts), compute_prauc(targets, predicts), compute_rce(targets, predicts)

def inference_time(model, data_loader, device):
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (fields, target) in enumerate(tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)):
            fields, target = fields.to(device), target.to(device)
            if i < 100:  # warmup
                y = model(fields)
            else:
                with torch.autograd.profiler.profile(use_cuda=False) as prof:
                    y = model(fields)

        print(prof.key_averages().table(sort_by="self_cpu_time_total"))


if __name__ == '__main__':
    batch_size = 2048
    epochs = 1
    #dataset = CriteoDataset("G://dac//train_ssss.txt")
    dataset = TwitterDataset("G://dac//twitter_final_s.parquet", twitter_label='like')
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length))
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=0)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

    #device = torch.device('cuda:0')
    #torch.cuda.empty_cache()
    device = torch.device('cpu')

    field_dims = dataset.field_dims

    #model = DeepFactorizationMachineModel(field_dims=field_dims, embed_dim=10, mlp_dims=(16, 16), dropout=0.2).to(device)
    #model = DeepFieldWeightedFactorizationMachineModel(field_dims=field_dims, embed_dim=10, use_fwlw=True, use_lw=False, mlp_dims=(400, 400, 400), dropout=0.2).to(device)

    model = FactorizationMachineModel(field_dims=field_dims, embed_dim=10).to(device)
    #model = FieldWeightedFactorizationMachineModel(field_dims=field_dims, embed_dim=10, use_fwlw=True, use_lw=False).to(device)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001, weight_decay=1e-6)

    for epoch_i in range(epochs):
        train(model, optimizer, train_data_loader, criterion, device)
        loss, auc, prauc, rce = test(model, valid_data_loader, device)
        print('epoch:', epoch_i)
        print(f'valid loss: {loss:.6f} auc: {auc:.6f} prauc: {prauc:.4f} rce: {rce:.4f}')

    loss, auc, prauc, rce = test(model, test_data_loader, device)
    print(f'test loss: {loss:.6f} auc: {auc:.6f} prauc: {prauc:.4f} rce: {rce:.4f}')

    '''mini_dataset, _ = torch.utils.data.random_split(dataset, (300, len(dataset) - 300))
    mini_data_loader = DataLoader(mini_dataset, batch_size=1, num_workers=0)
    inference_time(model, mini_data_loader, torch.device('cpu'))'''