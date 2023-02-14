import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer


def main():
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .1
    batch_size = 32
    neurons_per_layer = [64, 64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = False
    use_improved_weight_init = False
    use_momentum = False
    use_relu = False

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    # Example created for comparing with and without shuffling.
    # For comparison, show all loss/accuracy curves in the same plot
    # YOU CAN DELETE EVERYTHING BELOW!

    use_improved_weight_init = True

    # Train a new model with new parameters
    model_improved_weigth = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer_weight = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_improved_weigth, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_improved_weight, val_history_improved_weigth = trainer_weight.train(
        num_epochs)

    use_improved_weight_init = True
    use_improved_sigmoid = True


    # Train a new model with new parameters
    model_improved_sigmoid = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer_soft = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_improved_sigmoid, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_improved_sigmoid, val_history_improved_sigmoid = trainer_soft.train(
        num_epochs)
    use_momentum = True
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


    plt.subplot(1, 2, 1)
    utils.plot_loss(train_history["loss"],
                    "Task 2 Model", npoints_to_average=10)
    utils.plot_loss(
        train_history_improved_weight["loss"], "Task 2 Model - Improved Weight", npoints_to_average=10)
    utils.plot_loss(
        train_history_improved_sigmoid["loss"], "Task 2 Model - Improved Sigmoid", npoints_to_average=10)
    utils.plot_loss(
        train_history_mom["loss"], "Task 2 Model - With momentum", npoints_to_average=10)
    plt.ylim([0, .4])
    plt.subplot(1, 2, 2)
    plt.ylim([0.85, .99])
    utils.plot_loss(val_history["accuracy"], "Task 2 Model")
    utils.plot_loss(
        val_history_improved_weigth["accuracy"], "Task 2 Model - Improved Weigth")
    utils.plot_loss(
        val_history_improved_sigmoid["accuracy"], "Task 2 Model - Improved Sigmoid")
    utils.plot_loss(
        val_history_mom["accuracy"], "Task 2 Model - With momentum")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
