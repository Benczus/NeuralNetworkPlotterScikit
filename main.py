import csv
import matplotlib.pyplot as plt
import numpy as np
import sys
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    with open('data/dataset.csv', mode='r') as infile:
        reader = csv.reader(infile)
        mydict = {rows[0]: rows[1] for rows in reader}
        f_vector_s = list(mydict.keys())
        s_vector_s = list(mydict.values())
        input_vector = np.reshape([float(i) for i in f_vector_s], (-1, 1))
        output_vector = [float(i) for i in s_vector_s]
        print(np.asarray(output_vector).shape)
    model = MLPRegressor(hidden_layer_sizes=(50, 20, 30, 40, 50, 60),
                         activation='relu',
                         solver='lbfgs',
                         learning_rate='adaptive',
                         random_state=1,
                         max_iter=1000,
                         learning_rate_init=0.01,
                         alpha=0.01,
                         verbose=True)



    input_train, input_test, output_train, output_test = train_test_split(input_vector, output_vector, test_size=0.33,                                                                   random_state=42)
    model.fit(input_train, output_train)
    output_test = model.predict(input_test)
    print(model.score(input_vector, output_vector))
    plt.subplot(3, 1, 1)
    plt.plot(input_train, output_train, 'o')
    plt.title('')
    plt.ylabel('Actual Values(y)')
    plt.subplot(3, 1, 2)
    plt.plot(input_test, output_test, 'o')
    plt.xlabel('x values')
    plt.ylabel('Predicted Values(y)')
    plt.subplot(3, 1, 3)
    plt.plot(input_train, output_train, 'o')
    plt.plot(input_test, output_test, 'o')
    plt.xlabel('x values')
    plt.ylabel('Predicted Values(y)')
    plt.savefig('plot.pdf')
    plt.show()
