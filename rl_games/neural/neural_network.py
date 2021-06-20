from typing import NewType, Tuple, Sequence
from dataclasses import dataclass
import numpy as np

from rl_games.dqn.types import ActionVector, StateVector

Weights01Array = NewType('Weights01Array', np.ndarray)  # (I, H)
Weights12Array = NewType('Weights12Array', np.ndarray)  # (H, O)
Layer1Vector = NewType('Layer1Vector', np.ndarray)  # (1, H)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x)) # type: ignore


def dsigmoid_dx(x: np.ndarray) -> np.ndarray:
    return sigmoid(x) * (1 - sigmoid(x)) # type: ignore


# This Neural Network class has an input layer, one hidden layer, and an output layer.
@dataclass
class NeuralNetwork:
    input_size: int
    hidden_size: int
    output_size: int
    learning_rate: float = 0.1
    initial_scale: float = 0.01

    def __post_init__(self) -> None:
        # Between input layer (layer 0) and hidden layer (layer 1)
        self.weights_01 = Weights01Array(np.random.normal(0, self.initial_scale, (self.input_size, self.hidden_size)))
        # Between hidden (layer 1) and output (layer 2)
        self.weights_12 = Weights12Array(np.random.normal(0, self.initial_scale, (self.hidden_size, self.output_size)))

    def predict(self, input_vector: StateVector) -> ActionVector:
        """
        Feed the input vector forward through the neural network to get the output.
        >>> np.random.seed(1)
        >>> nn = NeuralNetwork(5, 3, 2, initial_scale=1)
        >>> v = np.array([[0.1 * i for i in range(1, nn.input_size + 1)]])
        >>> nn.predict(v)
        array([[-0.91008182, -0.44266949]])
        """
        assert input_vector.shape == (1, self.input_size)  # (1, I)
        linear_layer_1 = np.dot(input_vector, self.weights_01)  # (1, I) . (I, H) -> (1, H)
        layer_1 = Layer1Vector(sigmoid(linear_layer_1))
        linear_layer_2 = np.dot(layer_1, self.weights_12)  # (1, H) . (H, O) -> (1, O)

        output = ActionVector(np.array(linear_layer_2)) # (1, O), extra call as np.dot can return a non-array.
        return output

    def _compute_gradients(self, input_vector: StateVector, target: ActionVector) -> Tuple[Weights01Array, Weights12Array]:
        # pylint: disable=too-many-locals
        assert input_vector.shape == (1, self.input_size)  # (1, I)
        assert target.shape == (1, self.output_size)  # (1, O)

        # Feed forward (same as predict, but we'll need some intermediate variables)
        linear_layer_1 = np.dot(input_vector, self.weights_01)
        layer_1 = sigmoid(linear_layer_1)  # (1, H)
        linear_layer_2 = np.dot(layer_1, self.weights_12)
        output = np.array(linear_layer_2)  # (1, O)

        output_error = output - target  # (1, O)
        dcost_doutput = output_error  # Since the cost is output_error ^ 2

        # Backpropagate
        dlinear2_dweights12 = layer_1  # (1, H)
        doutput_dlinear2 = dsigmoid_dx(linear_layer_2)  # (1, O)
        dcost_dweights12 = np.dot(dlinear2_dweights12.T, dcost_doutput * doutput_dlinear2)  # (H, 1) . (1, O) -> (H, O)

        dcost_dlinear2 = dcost_doutput * doutput_dlinear2  # (1, O)
        dlinear2_dlayer1 = self.weights_12  # (H, O)
        dcost_dlayer1 = np.dot(dcost_dlinear2, dlinear2_dlayer1.T)  # (1, O) . (O, H) -> (1, H)

        dlayer1_dlinear1 = dsigmoid_dx(linear_layer_1)  # (1, H)
        dlinear1_dweights01 = input_vector  # (1, I)
        dcost_dweights01 = np.dot(dlinear1_dweights01.T, dlayer1_dlinear1 * dcost_dlayer1)  # (I, 1) . (1, H) -> (I, H)

        return Weights01Array(dcost_dweights01), Weights12Array(dcost_dweights12)

    def _update_weights(self, dcost_dweights01: Weights01Array, dcost_dweights12: Weights12Array) -> None:
        self.weights_01 -= dcost_dweights01 * self.learning_rate
        self.weights_12 -= dcost_dweights12 * self.learning_rate

    def calculate_total_cost(self, input_vectors: Sequence[StateVector], targets: Sequence[ActionVector]) -> float:
        """
        >>> np.random.seed(1)
        >>> nn = NeuralNetwork(5, 3, 2, initial_scale=1)
        >>> input_vectors = [np.random.standard_normal((1, nn.input_size)) for _ in range(200)]
        >>> targets = [np.array([[np.sum(v), np.sum(np.square(v))]]) for v in input_vectors]
        >>> nn.calculate_total_cost(input_vectors, targets)
        9558.742406617279
        """
        return sum(np.sum(np.square(self.predict(input_vector) - target))
                   for input_vector, target in zip(input_vectors, targets))


    def train(
        self,
        input_vectors: Sequence[StateVector],
        targets: Sequence[ActionVector],
        num_iterations: int,
        total_cost_step: int = 100,
    ) -> Sequence[float]:
        """
        >>> np.random.seed(1)
        >>> nn = NeuralNetwork(6, 4, 2, initial_scale=1)
        >>> input_vectors = [np.random.standard_normal((1, nn.input_size)) for _ in range(200)]
        >>> targets = [np.array([[np.sum(v), np.sum(np.square(v))]]) for v in input_vectors]
        >>> nn.train(input_vectors, targets, 20001, total_cost_step=20000)
        [11569.105614656739, 2278.1650967165383]
        >>> v = [1, -1, 0.5, 2, -2, 1]
        >>> sum(v), sum(x * x for x in v)
        (1.5, 11.25)
        >>> nn.predict(np.array([v]))
        array([[1.47841033, 7.34374856]])
        """
        total_costs = []
        for current_iteration in range(num_iterations):
            # Pick a data instance at random
            random_data_index = np.random.randint(len(input_vectors))
            input_vector = input_vectors[random_data_index]
            target = targets[random_data_index]
            # Compute the gradients and update the weights
            gradients = self._compute_gradients(input_vector, target)
            self._update_weights(*gradients)

            # Periodically calculate the total cost over the training period
            if current_iteration % total_cost_step == 0:
                total_cost = self.calculate_total_cost(input_vectors, targets)
                total_costs.append(total_cost)

        return total_costs
