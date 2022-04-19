import torch.nn as nn
import torch.optim as optim


class NeuralNetwork:

    def __init__(self, model, learning_rate, accuracy_check_frequency, device, scaler):
        self.training_cnt = 0
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss(reduction='mean')
        self.accuracy_check_frequency = accuracy_check_frequency
        self.device = device
        self.scaler = scaler

    def train(self, epochs, data_loader, device):
        self.training_cnt += 1
        if self.training_cnt % self.accuracy_check_frequency == 0:
            self.check_accuracy(data_loader, device)

        for i in range(epochs):
            for batch_index, (x, y) in enumerate(data_loader):
                # Calculating on device
                x = x.to(device=device)
                y = y.to(device=device)

                # Making a prediction
                prediction = self.model(x)
                # Bringing the solution and the prediction to a same format
                prediction = prediction.reshape(prediction.shape[0])
                y = y.reshape(y.shape[0])

                # loss calculating
                loss = self.criterion(prediction, y)

                # Optimizing
                self.optimizer.zero_grad()
                # Optimize backward
                loss.backward()
                # Optimize forward
                self.optimizer.step()

    def predict(self, data_array, device):
        return self.scaler.inverse_transform(self.model(data_array).data)

    def check_accuracy(self, data_loader, device):
        accuracy_sum = 0
        cnt = 0
        self.model.eval()
        for batch_index, (x, y) in enumerate(data_loader):
            # Calculating on device
            x = x.to(device=device)
            y = y.to(device=device)

            # Making a prediction
            prediction = self.model(x)

            for i in range(len(prediction)):
                # Get one prediction and exact value
                y_value = y[i][0]
                pred_value = prediction[i][0]

                # Determining a number between 0 and 1 about how close the two numbers are
                # not objective since we don't know the limits of the sequence. But can be useful.
                accuracy_sum += 1 - (abs(y_value - pred_value) / (y_value + pred_value))
                cnt += 1

        if cnt != 0:
            print(f'{self.training_cnt}. tanítás kész,\n'
                  f'nagyjából {(accuracy_sum / cnt) * 100}%-os pontosságú.')
        else:
            print(f'Nem ismert a pontosság!')
