from torch.utils.data import Dataset, DataLoader


class myDataset(Dataset):
    def __init__(self, data, labels):
        super(myDataset, self).__init__()
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.data[item], self.labels[item]


def toDataloader(prop, X_train_task, y_train_task, X_test, y_test):
    # X_train_task
    dataset_train = myDataset(X_train_task, y_train_task)
    dataloader_train = DataLoader(dataset_train, batch_size=prop['batch'], shuffle=True)
    # X_test
    dataset_test = myDataset(X_test, y_test)
    dataloader_test = DataLoader(dataset_test, batch_size=prop['batch'])

    return dataloader_train, dataloader_test
