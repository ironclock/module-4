from minitorch.nn import dropout, logsoftmax, maxpool2d
from minitorch.tensor import Tensor
from minitorch.tensor_functions import ReLU
from mnist import MNIST

import minitorch

mndata = MNIST("project/data/")
images, labels = mndata.load_training()

BACKEND = minitorch.TensorBackend(minitorch.FastOps)
BATCH = 16

# Number of classes (10 digits)
C = 10

# Size of images (height and width)
H, W = 28, 28


def RParam(*shape):
    r = 0.1 * (minitorch.rand(shape, backend=BACKEND) - 0.5)
    return minitorch.Parameter(r)


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size

    def forward(self, x):
        batch, in_size = x.shape
        return (
            x.view(batch, in_size) @ self.weights.value.view(in_size, self.out_size)
        ).view(batch, self.out_size) + self.bias.value


class Conv2d(minitorch.Module):
    def __init__(self, in_channels, out_channels, kh, kw):
        super().__init__()
        self.weights = RParam(out_channels, in_channels, kh, kw)
        self.bias = RParam(out_channels, 1, 1)

    def forward(self, input):
        # DONE: Implement for Task 4.5.
        weights = self.weights()
        bias = self.bias()
        _, in_channels, _, _ = self.weights().shape  # Get in_channels from weights

        # Determine output dimensions
        batch_size, _, height, width = input.shape
        out_channels, _, kh, kw = weights.shape
        output_height = height - kh + 1
        output_width = width - kw + 1

        # Create output tensor
        output = Tensor.zeros((batch_size, out_channels, output_height, output_width))

        # Perform convolution
        for i in range(batch_size):
            for j in range(out_channels):
                for h in range(output_height):
                    for w in range(output_width):
                        for c in range(in_channels):
                            output[i, j, h, w] += \
                                (input[i, c, h:h+kh, w:w+kw] * weights[j, c]).sum()

        # Add bias to each output channel
        for j in range(out_channels):
            output[:, j, :, :] += bias[j]

        return output


class Network(minitorch.Module):
    """
    Implement a CNN for MNist classification based on LeNet.

    This model should implement the following procedure:

    1. Apply a convolution with 4 output channels and a 3x3 kernel followed by a ReLU (save to self.mid)
    2. Apply a convolution with 8 output channels and a 3x3 kernel followed by a ReLU (save to self.out)
    3. Apply 2D pooling (either Avg or Max) with 4x4 kernel.
    4. Flatten channels, height, and width. (Should be size BATCHx392)
    5. Apply a Linear to size 64 followed by a ReLU and Dropout with rate 25%
    6. Apply a Linear to size C (number of classes).
    7. Apply a logsoftmax over the class dimension.
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()

        # For vis
        self.mid = None
        self.out = None

        # DONE: Implement for Task 4.5.
        self.conv1 = Conv2d(in_channels, 4, 3, 3)  # 4 output channels, 3x3 kernel
        self.conv2 = Conv2d(4, 8, 3, 3)           # 8 output channels, 3x3 kernel

        # ReLU activations
        self.relu = ReLU()

        # Pooling layer
        self.pool = maxpool2d(kernel=(4, 4))  # Replace with AvgPool2d if needed

        # Linear layers
        self.fc1 = Linear(392, 64)  # Flatten to size 392, then linear to size 64
        self.fc2 = Linear(64, num_classes)  # Linear to number of classes

        # Dropout
        self.dropout = dropout(rate=0.25)

    def forward(self, x):
        # DONE: Implement for Task 4.5.
        # Step 1: First Convolution and ReLU
        self.mid = self.relu(self.conv1(x))

        # Step 2: Second Convolution and ReLU
        self.out = self.relu(self.conv2(self.mid))

        # Step 3: Pooling
        x = self.pool(self.out)

        # Step 4: Flatten
        x = x.view(x.shape[0], -1)  # Flatten the tensor while keeping the batch dimension

        # Step 5: Linear, ReLU, and Dropout
        x = self.dropout(self.relu(self.fc1(x)))

        # Step 6: Second Linear layer
        x = self.fc2(x)

        # Step 7: Logsoftmax
        x = logsoftmax(x, dim=1)

        return x


def make_mnist(start, stop):
    ys = []
    X = []
    for i in range(start, stop):
        y = labels[i]
        vals = [0.0] * 10
        vals[y] = 1.0
        ys.append(vals)
        X.append([[images[i][h * W + w] for w in range(W)] for h in range(H)])
    return X, ys


def default_log_fn(epoch, total_loss, correct, total, losses, model):
    print(f"Epoch {epoch} loss {total_loss} valid acc {correct}/{total}")


class ImageTrain:
    def __init__(self):
        self.model = Network()

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x], backend=BACKEND))

    def train(
        self, data_train, data_val, learning_rate, max_epochs=500, log_fn=default_log_fn
    ):
        (X_train, y_train) = data_train
        (X_val, y_val) = data_val
        self.model = Network()
        model = self.model
        n_training_samples = len(X_train)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)
        losses = []
        for epoch in range(1, max_epochs + 1):
            total_loss = 0.0

            model.train()
            for batch_num, example_num in enumerate(
                range(0, n_training_samples, BATCH)
            ):

                if n_training_samples - example_num <= BATCH:
                    continue
                y = minitorch.tensor(
                    y_train[example_num : example_num + BATCH], backend=BACKEND
                )
                x = minitorch.tensor(
                    X_train[example_num : example_num + BATCH], backend=BACKEND
                )
                x.requires_grad_(True)
                y.requires_grad_(True)
                # Forward
                out = model.forward(x.view(BATCH, 1, H, W)).view(BATCH, C)
                prob = (out * y).sum(1)
                loss = -(prob / y.shape[0]).sum()

                assert loss.backend == BACKEND
                loss.view(1).backward()

                total_loss += loss[0]
                losses.append(total_loss)

                # Update
                optim.step()

                if batch_num % 5 == 0:
                    model.eval()
                    # Evaluate on 5 held-out batches

                    correct = 0
                    for val_example_num in range(0, 1 * BATCH, BATCH):
                        y = minitorch.tensor(
                            y_val[val_example_num : val_example_num + BATCH],
                            backend=BACKEND,
                        )
                        x = minitorch.tensor(
                            X_val[val_example_num : val_example_num + BATCH],
                            backend=BACKEND,
                        )
                        out = model.forward(x.view(BATCH, 1, H, W)).view(BATCH, C)
                        for i in range(BATCH):
                            m = -1000
                            ind = -1
                            for j in range(C):
                                if out[i, j] > m:
                                    ind = j
                                    m = out[i, j]
                            if y[i, ind] == 1.0:
                                correct += 1
                    log_fn(epoch, total_loss, correct, BATCH, losses, model)

                    total_loss = 0.0
                    model.train()


if __name__ == "__main__":
    data_train, data_val = (make_mnist(0, 5000), make_mnist(10000, 10500))
    ImageTrain().train(data_train, data_val, learning_rate=0.01)
