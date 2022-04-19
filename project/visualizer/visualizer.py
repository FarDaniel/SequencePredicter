import matplotlib.pyplot as plt


def visualize_data(datasets):
    plt.figure(figsize=(20, 10))
    for dataset in datasets:
        plt.plot(dataset)
    plt.show()
