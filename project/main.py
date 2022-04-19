import torch
from sklearn.preprocessing import MinMaxScaler

from project.data_processing.data_reader import DataReader
from project.data_processing.sequence_dataset import SequenceDataset
from project.neural_network.lstm_model import LSTMModel
from project.neural_network.neural_network import NeuralNetwork

# constant variables
SEPARATOR = ";"
OUTPUT_PATH = "prediction.txt"
INPUT_SIZE = 50
LEARNING_RATE = 0.001
LOADING_SIZE = 250
ACCURACY_CHECK_CNT = 5
HIDDEN_SIZE = 128
NUM_LAYERS = 6
BATCH_SIZE = 4
EPOCH_CNT = 4


def read_data(minimal_value, maximum_value, standard_value, only_int):
    while True:
        try:
            variable = input()
            if variable == "":
                variable = standard_value
            if minimal_value == -1 or minimal_value < int(variable) and \
                    maximum_value == -1 or int(variable) < maximum_value:
                if only_int:
                    return int(variable)
                else:
                    return variable
        except:
            pass
        print("Kérem szám értéket adjon meg, a megfelelő intervallumban.")


if __name__ == '__main__':
    dr = DataReader(SEPARATOR)
    input_size = None
    learning_rate = None

    while not dr.is_file_opened:
        print("Kérem adja meg az adatfile elérési útvonalát.")
        data_path = input()
        dr.open_file(data_path)

    print(f'Kérem adja meg a háló inputjának méretét.\n'
          f'Ha nem ad meg, {INPUT_SIZE}-el számol az program.')
    input_size = read_data(0, -1, INPUT_SIZE, True)

    print(f'Kérem adja meg a tanulási rátát (nullánál nagyobb, egynél kisebb).\n'
          f'Ha nem ad meg, {LEARNING_RATE}-el számol az program.')
    learning_rate = read_data(0, 1, LEARNING_RATE, False)

    print(f'Kérem adja meg hány adat legyen egyszerre betöltve.\n'
          f'Nagyobbnak kell lennie, mint az input mérete {input_size}'
          f'Ha nem ad meg, {LOADING_SIZE}-el számol az program.')
    load_size = read_data(input_size, -1, LOADING_SIZE, True)

    print(f'Kérem adja meg hány tanítási körönként legyen ellenőrizve a pontosság.\n'
          f'Ha nem ad meg, {ACCURACY_CHECK_CNT}-el számol az program.')
    accuracy_check_cnt = read_data(0, -1, ACCURACY_CHECK_CNT, True)

    print(f'Kérem adja meg hány rejtett node legyen.\n'
          f'Ha nem ad meg, {HIDDEN_SIZE}-el számol az program.')
    hidden_size = read_data(0, -1, HIDDEN_SIZE, True)

    print(f'Kérem adja meg Hány LSTM réteg legyen.\n'
          f'Ha nem ad meg, {NUM_LAYERS}-el számol az program.')
    num_layers = read_data(0, -1, NUM_LAYERS, True)

    print(f'Kérem adja meg mekkora legyen egy batch.\n'
          f'Ha nem ad meg, {BATCH_SIZE}-el számol az program.')
    batch_size = read_data(0, -1, BATCH_SIZE, True)

    print(f'Kérem adja meg egy adathalmazon hányszor legyen tanítva a jáló.\n'
          f'Ha nem ad meg, {EPOCH_CNT}-el számol az program.')
    epoch_cnt = read_data(0, -1, EPOCH_CNT, True)

    # Finding device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Need to use the same scaler everywhere
    scaler = MinMaxScaler((0, 1))

    # Dataset for reading infinite rows of numbers
    data = SequenceDataset(dr, load_size, input_size, scaler)

    model = LSTMModel(input_size, hidden_size, num_layers, 1, device)
    network = NeuralNetwork(model, learning_rate, accuracy_check_cnt, device, scaler)

    is_last = False
    while not is_last:
        is_last = data.load_next()
        network.train(epoch_cnt, data.get_data_loader(batch_size), device)

    output_file = open(OUTPUT_PATH, "w")
    prediction = network.predict(data.prepare_last_x(), device)[0][0]
    prediction = round(prediction)
    print(f'A tippelt adat: {prediction}, a\n a file relatív helye: {OUTPUT_PATH}')
    output_file.write(f'{prediction}')
    output_file.close()
