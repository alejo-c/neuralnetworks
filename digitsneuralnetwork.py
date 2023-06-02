import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt


def main():
    digit_to_array = lambda digit: np.eye(10)[digit]
    array_to_digit = lambda arr: np.argmax(arr)
    inputs_shape = 784, 1
    learning_rate = 0.003
    epochs = 10
    activation = "sigmoid"

    print('Importing datasets...')
    mnist = keras.datasets.mnist
    (training_inputs, training_results), (test_inputs, test_results) = mnist.load_data()

    print("Fix data to be modeled...")
    training_inputs, test_inputs = training_inputs / 255.0, test_inputs / 255.0

    training_inputs = [np.reshape(digit, inputs_shape) for digit in training_inputs]
    training_results = [digit_to_array(digit) for digit in training_results]
    test_inputs = [np.reshape(digit, inputs_shape) for digit in test_inputs]

    training_inputs = np.squeeze(training_inputs)
    training_results = np.squeeze(training_results)
    test_inputs = np.squeeze(test_inputs)
    test_results = np.squeeze(test_results)

    print("Designing Model...")
    model = keras.Sequential(
        [
            keras.layers.Dense(units=16, activation=activation, input_shape=(784,)),
            keras.layers.Dense(units=16, activation=activation),
            keras.layers.Dense(units=10, activation=activation),
        ]
    )

    print("Compiling Model...")
    # model.compile(optimizer="adam", loss="mean_squared_error")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mean_squared_error",
    )

    print("Training Model...")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir="./logs")
    training = model.fit(training_inputs, training_results, epochs=epochs)
    # training = model.fit(
        # training_inputs,
        # training_results,
        # epochs=epochs,
        # verbose=True,
        # callbacks=tensorboard_callback,
    # )

    print('Evaluate Model...')
    model.evaluate(test_inputs, test_results, verbose=2)

    # print("Predicting with test inputs...")
    # predictions = model.predict(test_inputs)
    # predictions = np.apply_along_axis(array_to_digit, axis=1, arr=predictions)
    # matching_percentage = np.sum(predictions == test_results) / len(predictions) * 100
    # print(f"Matching percentage: {matching_percentage}%")

    # plt.title("Digits Neural Network")
    # plt.grid(True)
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.plot(training.history["loss"])
    # plt.show()


if __name__ == "__main__":
    main()
