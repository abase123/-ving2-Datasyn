import numpy as np
import utils
import typing
np.random.seed(1)


def pre_process_images(X: np.ndarray):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] normalized as described in task2a
    """
    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"
    # TODO implement this function (Task 2a)

    mean = np.mean(X)
    std = np.std(X)
    X_norm = (X-mean)/std
    #X_train, _, _, _ = utils.load_full_mnist()
    bias = np.ones((X_norm.shape[0], 1))
    X_norm = np.hstack((X_norm, bias))
    return X_norm


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
    # TODO: Implement this function (copy from last assignment)

    neg_lik = np.sum(targets * np.log(outputs))
    loss = -np.sum(neg_lik)/targets.shape[0]

    return loss



class SoftmaxModel:

    def __init__(self,
                 # Number of neurons per layer
                 neurons_per_layer: typing.List[int],
                 use_improved_sigmoid: bool,  # Task 3b hyperparameter
                 use_improved_weight_init: bool,  # Task 3a hyperparameter
                 use_relu: bool  # Task 4 hyperparameter
                 ):
        # Always reset random seed before weight init to get comparable results.
        np.random.seed(1)
        # Define number of input nodes
        self.I = 785
        self.use_improved_sigmoid = use_improved_sigmoid
        self.use_relu = use_relu
        self.use_improved_weight_init = use_improved_weight_init
        self.hidden_layer_output = []

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer
        self.z = [None for i in range(len(neurons_per_layer) - 1)]
        self.a = [None for i in range(len(neurons_per_layer) - 1)]

        # Initialize the weights
        self.ws = []
        #if(self.use_improved_weight_init):
         #   self.ws = [np.random.normal(loc= 0, scale=1/np.sqrt(785), size=(785,64)),
         #              np.random.normal(loc= 0, scale=1/np.sqrt(64), size=(64,10))]
        prev = self.I
        for size in self.neurons_per_layer:
            w_shape = (prev, size)
            print("Initializing weight to shape:", w_shape)
            w = np.zeros(w_shape)
            self.ws.append(w)
            prev = size
        self.grads = [None for i in range(len(self.ws))]
        self.grads_prev = [None for i in range(len(self.ws))]
        if use_improved_weight_init:
            self.ws[0] = np.random.normal(0, 1 / np.sqrt(self.I), self.ws[0].shape)
            for i in range(len(self.neurons_per_layer) - 1):
                self.ws[i+1] = np.random.normal(0, 1/np.sqrt(self.neurons_per_layer[i]), self.ws[i+1].shape)
        else:
            self.ws[0] = np.random.uniform(-1, 1, (self.I, self.neurons_per_layer[0]))
            for i in range(len(self.neurons_per_layer)-1):
                self.ws[i+1] = np.random.uniform(-1, 1, (self.neurons_per_layer[i], self.neurons_per_layer[i+1]))

    def forward(self, X: np.ndarray, layer = 0) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """
        # TODO implement this function (Task 2b)
        # HINT: For performing the backward pass, you can save intermediate activations in variables in the forward pass.
        # such as self.hidden_layer_output = ...

        self.z[layer] = np.dot(X, self.ws[layer])

        if self.use_relu:
            self.a[layer] = np.maximum(0, self.z[layer])
        elif self.use_improved_sigmoid:
            self.a[layer] = 1.7159*np.tanh(2.0/3.0 * self.z[layer])
        else:
            self.a[layer] = 1 / (1 + np.exp(-self.z[layer]))


        if (layer+1) < (len(self.neurons_per_layer)-1):
            layer = layer + 1
            return self.forward(self.a[layer-1], layer)

        z2 = np.dot(self.a[-1], self.ws[-1])
        #self.z.append(z2)
        z2 = z2 - np.max(z2, axis=1, keepdims=True)  # subtract max for stability
        output = np.exp(z2) / np.sum(np.exp(z2), axis=1, keepdims=True)

        return output


    def backward(self, X: np.ndarray, outputs: np.ndarray,
                 targets: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad

        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        # TODO implement this function (Task 2b)


        batch_size = X.shape[0]
        delta = -(targets-outputs)
        self.grads[-1] = np.dot(delta.T, self.a[-1]).T/(batch_size)

        #sigmoid = 1/(1+np.exp(-self.z[0]))
        for i in range(len(self.a), 0, -1):
            #if self.use_relu:
             #   for i in range(len(np.dot(delta, self.ws[i].T))):
             #       if np.dot(delta, self.ws[i].T[i]):
              #          delta[i] = 1
               #     else:
               #         delta[i] = 0
            elif self.use_improved_sigmoid:
                der_sig = 2/3 * 1.7159*(1-self.a[i-1]**2/1.7159**2)
                delta = der_sig * np.dot(delta, self.ws[i].T)
            else:
                der_sig = self.a[i-1]*(1-self.a[i-1])
                delta = der_sig*np.dot(delta, self.ws[i].T)
            if i != 1:
                self.grads[i-1] = np.dot(delta.T, self.a[i-2]).T/outputs.shape[0]
            else:
                self.grads[0] = np.dot(delta.T, X).T /(batch_size)







        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"
        # A list of gradients.
        # For example, self.grads[0] will be the gradient for the first hidden layer

        #for grad, w in zip(self.grads, self.ws):
         #   assert grad.shape == w.shape,\
          #      f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."

    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    # TODO: Implement this function (copy from last assignment)
    one_hot = np.eye(num_classes)[Y.flatten()]
    return one_hot



def gradient_approximation_test(
        model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited. 
        Details about this test is given in the appendix in the assignment.
    """
    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**1,\
                    f"Calculated gradient is incorrect. " \
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n" \
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n" \
                    f"If this test fails there could be errors in your cross entropy loss function, " \
                    f"forward function or backward function"


def main():
    # Simple test on one-hot encoding
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

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_relu = False
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init, use_relu)

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)


if __name__ == "__main__":
    main()
