import typing
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import numpy as np

from docarray.computation import AbstractComputationalBackend
from docarray.computation.abstract_numpy_based_backend import AbstractNumpyBasedBackend
from docarray.typing import TensorFlowTensor
from docarray.utils._internal.misc import import_library

if TYPE_CHECKING:
    import tensorflow as tf  # type: ignore
    import tensorflow._api.v2.experimental.numpy as tnp  # type: ignore
else:
    tf = import_library('tensorflow', raise_error=True)
    tnp = tf._api.v2.experimental.numpy


def _unsqueeze_if_single_axis(*matrices: tf.Tensor) -> List[tf.Tensor]:
    """
    Unsqueezes tensors that only have one axis, at dim 0.
    This ensures that all outputs can be treated as matrices, not vectors.

    :param matrices: Matrices to be unsqueezed
    :return: List of the input matrices,
        where single axis matrices are unsqueezed at dim 0.
    """
    pass


def _unsqueeze_if_scalar(t: tf.Tensor) -> tf.Tensor:
    """
    Unsqueezes tensor of a scalar, from shape () to shape (1,).

    :param t: tensor to unsqueeze.
    :return: unsqueezed tf.Tensor
    """
    pass






class TensorFlowCompBackend(AbstractNumpyBasedBackend[TensorFlowTensor]):
    """
    Computational backend for TensorFlow.
    """

    _module = tnp
    _cast_output: Callable = norm_left
    _get_tensor: Callable = norm_right

    @classmethod
    def to_numpy(cls, array: 'TensorFlowTensor') -> 'np.ndarray':
        return cls._get_tensor(array).numpy()

    @classmethod
    def none_value(cls) -> typing.Any:
        """Provide a compatible value that represents None in numpy."""
        pass

    @classmethod
    def to_device(cls, tensor: 'TensorFlowTensor', device: str) -> 'TensorFlowTensor':
        """Move the tensor to the specified device."""
        if cls.device(tensor) == device:
            return tensor
        else:
            with tf.device(device):
                return cls._cast_output(tf.identity(cls._get_tensor(tensor)))

    @classmethod
    def device(cls, tensor: 'TensorFlowTensor') -> Optional[str]:
        """Return device on which the tensor is allocated."""
        return cls._get_tensor(tensor).device

    @classmethod
    def detach(cls, tensor: 'TensorFlowTensor') -> 'TensorFlowTensor':
        """
        Returns the tensor detached from its current graph.

        :param tensor: tensor to be detached
        :return: a detached tensor with the same data.
        """
        return cls._cast_output(tf.stop_gradient(cls._get_tensor(tensor)))

    @classmethod
    def dtype(cls, tensor: 'TensorFlowTensor') -> tf.dtypes:
        """Get the data type of the tensor."""
        pass


    @classmethod
    def equal(cls, tensor1: 'TensorFlowTensor', tensor2: 'TensorFlowTensor') -> bool:
        """
        Check if two tensors are equal.

        :param tensor1: the first tensor
        :param tensor2: the second tensor
        :return: True if two tensors are equal, False otherwise.
            If one or more of the inputs is not a TensorFlowTensor, return False.
        """
        pass

    class Retrieval(AbstractComputationalBackend.Retrieval[TensorFlowTensor]):
        """
        Abstract class for retrieval and ranking functionalities
        """

        @staticmethod
        def top_k(
            values: 'TensorFlowTensor',
            k: int,
            descending: bool = False,
            device: Optional[str] = None,
        ) -> Tuple['TensorFlowTensor', 'TensorFlowTensor']:
            """
            Retrieves the top k smallest values in `values`,
            and returns them alongside their indices in the input `values`.
            Can also be used to retrieve the top k largest values,
            by setting the `descending` flag.

            :param values: TensorFlowTensor of values to rank.
                Should be of shape (n_queries, n_values_per_query).
                Inputs of shape (n_values_per_query,) will be expanded
                to (1, n_values_per_query).
            :param k: number of values to retrieve
            :param descending: retrieve largest values instead of smallest values
            :param device: the computational device to use.
            :return: Tuple of TensorFlowTensors containing the retrieved values, and
                their indices. Both are of shape (n_queries, k)
            """
            comp_be = TensorFlowCompBackend
            if device is not None:
                values = comp_be.to_device(values, device)

            tf_values: tf.Tensor = comp_be._get_tensor(values)
            if len(tf_values.shape) <= 1:
                tf_values = tf.expand_dims(tf_values, axis=0)

            len_tf_values = (
                tf_values.shape[-1] if len(tf_values.shape) > 1 else len(tf_values)
            )
            k = min(k, len_tf_values)

            if not descending:
                tf_values = -tf_values

            result = tf.math.top_k(input=tf_values, k=k, sorted=True)
            res_values = result.values
            res_indices = result.indices

            if not descending:
                res_values = -result.values

            return comp_be._cast_output(res_values), comp_be._cast_output(res_indices)

    class Metrics(AbstractComputationalBackend.Metrics[TensorFlowTensor]):
        """
        Abstract base class for metrics (distances and similarities).
        """

        @staticmethod
        def cosine_sim(
            x_mat: 'TensorFlowTensor',
            y_mat: 'TensorFlowTensor',
            eps: float = 1e-7,
            device: Optional[str] = None,
        ) -> 'TensorFlowTensor':
            """Pairwise cosine similarities between all vectors in x_mat and y_mat.

            :param x_mat: tensor of shape (n_vectors, n_dim), where n_vectors is the
                number of vectors and n_dim is the number of dimensions of each example.
            :param y_mat: tensor of shape (n_vectors, n_dim), where n_vectors is the
                number of vectors and n_dim is the number of dimensions of each example.
            :param eps: a small jitter to avoid divde by zero
            :param device: the device to use for computations.
                If not provided, the devices of x_mat and y_mat are used.
            :return: Tensor  of shape (n_vectors, n_vectors) containing all pairwise
                cosine distances.
                The index [i_x, i_y] contains the cosine distance between
                x_mat[i_x] and y_mat[i_y].
            """
            pass

        @staticmethod
        def euclidean_dist(
            x_mat: 'TensorFlowTensor',
            y_mat: 'TensorFlowTensor',
            device: Optional[str] = None,
        ) -> 'TensorFlowTensor':
            """Pairwise Euclidian distances between all vectors in x_mat and y_mat.

            :param x_mat: tensor of shape (n_vectors, n_dim), where n_vectors is the
                number of vectors and n_dim is the number of dimensions of each example.
            :param y_mat: tensor of shape (n_vectors, n_dim), where n_vectors is the
                number of vectors and n_dim is the number of dimensions of each example.
            :param device: the device to use for pytorch computations.
                If not provided, the devices of x_mat and y_mat are used.
            :return: Tensor of shape (n_vectors, n_vectors) containing all pairwise
                euclidian distances.
                The index [i_x, i_y] contains the euclidian distance between
                x_mat[i_x] and y_mat[i_y].
            """
            pass

        @staticmethod
        def sqeuclidean_dist(
            x_mat: 'TensorFlowTensor',
            y_mat: 'TensorFlowTensor',
            device: Optional[str] = None,
        ) -> 'TensorFlowTensor':
            """Pairwise Squared Euclidian distances between all vectors
                in x_mat and y_mat.

            :param x_mat: tensor of shape (n_vectors, n_dim), where n_vectors is the
                number of vectors and n_dim is the number of dimensions of each
                example.
            :param y_mat: tensor of shape (n_vectors, n_dim), where n_vectors is the
                number of vectors and n_dim is the number of dimensions of each
                example.
            :param device: the device to use for pytorch computations.
                If not provided, the devices of x_mat and y_mat are used.
            :return: Tensor of shape (n_vectors, n_vectors) containing all pairwise
                euclidian distances.
                The index [i_x, i_y] contains the euclidian distance between
                x_mat[i_x] and y_mat[i_y].
            """
            pass
