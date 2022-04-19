import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SequenceDataset(Dataset):

    def __init__(self, data_reader, data_per_iteration, input_size, scaler):
        self.data_reader = data_reader
        self.data_per_iteration = data_per_iteration
        self.input_size = input_size
        self.scaler = scaler

        self.raw_x = None
        self.x = None
        self.y = None
        self.samples_cnt = None
        self.raw_data = np.zeros(data_per_iteration + 1)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.samples_cnt

    def load_next(self):
        new_data = np.array(self.data_reader.get_numbers(self.data_per_iteration + 1))
        is_last = self.raw_data.shape != new_data.shape
        if is_last:
            # The last set of data is smaller, so we have to fill up, from the previous load too
            # It has an effect on the training process, but so small that the
            # random factor has a bigger role.
            self.raw_data = np.append(self.raw_data[len(new_data):], new_data)
        else:
            # Standard set of data, with the standard size
            self.raw_data = new_data

        # Changing the structure for the neural network
        self.prepare_data()
        return is_last

    def prepare_last_x(self):
        self.raw_data = self.raw_data[len(self.raw_data) - self.input_size:]
        try:
            data = self.scaler.transform(self.raw_data.reshape(-1, 1))
        except:
            data = self.scaler.fit_transform(self.raw_data.reshape(-1, 1))

        x = [data[:]]

        x = np.array(x, dtype=np.float32)

        x = np.reshape(x, (x.shape[0], 1, x.shape[1]))

        self.x = torch.from_numpy(x)
        self.y = []
        self.samples_cnt = 1
        return self.x

    def get_data_loader(self, batch_size):
        return DataLoader(dataset=self, batch_size=batch_size, shuffle=True)

    def prepare_data(self):
        try:
            data = self.scaler.transform(self.raw_data.reshape(-1, 1))
        except:
            data = self.scaler.fit_transform(self.raw_data.reshape(-1, 1))

        x = []
        y = []

        for i in range(self.input_size, len(data)):
            x.append(data[i - self.input_size:i])
            y.append(data[i])

        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        x = np.reshape(x, (x.shape[0], 1, x.shape[1]))

        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.samples_cnt = len(self.y)
