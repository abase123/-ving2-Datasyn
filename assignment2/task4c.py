import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import one_hot_encode, pre_process_images, SoftmaxModel, gradient_approximation_test
from task2 import SoftmaxTrainer

def main():
    # Simple test on one-hot encoding
    num_epochs = 50
    learning_rate = .1
    batch_size = 32
    neurons_per_layer = [64, 64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert Y[0, 3] == 1 and Y.sum() == 1, \
        f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    Y_train = one_hot_encode(Y_train, 10)
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    # Modify your network here
    neurons_per_layer = [64, 10]
    use_improved_sigmoid = False
    use_improved_weight_init = True
    use_momentum = True
    use_relu = True
    shuffle_data = True

    #model = SoftmaxModel(
    #    neurons_per_layer, use_improved_sigmoid, use_improved_weight_init, use_relu)

    # Gradient approximation check for 100 images
    #X_train = X_train[:100]
    #Y_train = Y_train[:100]
    #for layer_idx, w in enumerate(model.ws):
     #   model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    #gradient_approximation_test(model, X_train, Y_train)

    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    learning_rate = .02
    # Train a new model with new parameters
    model_mom = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer_mom = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_mom, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_mom, val_history_mom = trainer_mom.train(
        num_epochs)
    neurons_per_layer = [64, 64, 10]
    model_3lay = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer_3lay = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_3lay, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_3lay, val_history_3lay = trainer_3lay.train(
        num_epochs)
    neurons_per_layer = [64, 64, 64, 64, 64, 64, 64, 64, 64, 10]
    model_10lay = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer_10lay = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_10lay, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_10lay, val_history_10lay = trainer_10lay.train(
        num_epochs)

    plt.subplot(1, 2, 1)
    utils.plot_loss(
        train_history_mom["loss"], "Task 4 Model - 2 Layers", npoints_to_average=10)
    utils.plot_loss(
        train_history_3lay["loss"], "Task 4 Model - 3 Layers", npoints_to_average=10)
    utils.plot_loss(
        train_history_10lay["loss"], "Task 4 Model - 10 Layers", npoints_to_average=10)
    plt.ylim([0, .6])
    plt.subplot(1, 2, 2)
    plt.ylim([0.65, .99])
    utils.plot_loss(
        val_history_mom["accuracy"], "Task 4 Model - 2 Layer")
    utils.plot_loss(
        val_history_3lay["accuracy"], "Task 4 Model - 3 Layer")
    utils.plot_loss(
        val_history_10lay["accuracy"], "Task 4 Model - 10 Layer")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.show()
if __name__ == "__main__":
    main()
