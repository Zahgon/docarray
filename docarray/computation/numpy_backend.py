import warnings
from typing import Any, List, Optional, Tuple

import numpy as np

from docarray.computation import AbstractComputationalBackend
from docarray.computation.abstract_numpy_based_backend import AbstractNumpyBasedBackend


def _expand_if_single_axis(*matrices: np.ndarray) -> List[np.ndarray]:
    """Expands arrays that only have one axis, at dim 0.
    This ensures that all outputs can be treated as matrices, not vectors.

    :param matrices: Matrices to be expanded
    :return: List of the input matrices,
        where single axis matrices are expanded at dim 0.
    """
    pass




def identity(array: np.ndarray) -> np.ndarray:
    return array


class NumpyCompBackend(AbstractNumpyBasedBackend):
    """
    Computational backend for Numpy.
    """

    _module = np
    _cast_output = identity
    _get_tensor = identity

    @classmethod
    def to_device(cls, tensor: 'np.ndarray', device: str) -> 'np.ndarray':
        """Move the tensor to the specified device."""
        raise NotImplementedError('Numpy does not support devices (GPU).')

    @classmethod
    def device(cls, tensor: 'np.ndarray') -> Optional[str]:
        """Return device on which the tensor is allocated."""
        return None

    @classmethod
    def to_numpy(cls, array: 'np.ndarray') -> 'np.ndarray':
        return array

    @classmethod
    def none_value(cls) -> Any:
        """Provide a compatible value that represents None in numpy."""
        pass

    @classmethod
    def detach(cls, tensor: 'np.ndarray') -> 'np.ndarray':
        """
        Returns the tensor detached from its current graph.

        :param tensor: tensor to be detached
        :return: a detached tensor with the same data.
        """
        return tensor

    @classmethod
    def dtype(cls, tensor: 'np.ndarray') -> np.dtype:
        """Get the data type of the tensor."""
        pass

    @classmethod
    def minmax_normalize(
        cls,
        tensor: 'np.ndarray',
        t_range: Tuple = (0, 1),
        x_range: Optional[Tuple] = None,
        eps: float = 1e-7,
    ) -> 'np.ndarray':
        """
        Normalize values in `tensor` into `t_range`.

        `tensor` can be a 1D array or a 2D array. When `tensor` is a 2D array, then
        normalization is row-based.

        !!! note

            - with `t_range=(0, 1)` will normalize the min-value of data to 0, max to 1;
            - with `t_range=(1, 0)` will normalize the min-value of data to 1, max value
              of the data to 0.

        :param tensor: the data to be normalized
        :param t_range: a tuple represents the target range.
        :param x_range: a tuple represents tensors range.
        :param eps: a small jitter to avoid divide by zero
        :return: normalized data in `t_range`
        """
        pass

    @classmethod
    def equal(cls, tensor1: 'np.ndarray', tensor2: 'np.ndarray') -> bool:
        """
        Check if two tensors are equal.

        :param tensor1: the first array
        :param tensor2: the second array
        :return: True if two arrays are equal, False otherwise.
            If one or more of the inputs is not an ndarray, return False.
        """
        pass

    class Retrieval(AbstractComputationalBackend.Retrieval[np.ndarray]):
        """
        Abstract class for retrieval and ranking functionalities
        """

        @staticmethod
        def top_k(
            values: 'np.ndarray',
            k: int,
            descending: bool = False,
            device: Optional[str] = None,
        ) -> Tuple['np.ndarray', 'np.ndarray']:
            """
            Retrieves the top k smallest values in `values`,
            and returns them alongside their indices in the input `values`.
            Can also be used to retrieve the top k largest values,
            by setting the `descending` flag.

            :param values: Torch tensor of values to rank.
                Should be of shape (n_queries, n_values_per_query).
                Inputs of shape (n_values_per_query,) will be expanded
                to (1, n_values_per_query).
            :param k: number of values to retrieve
            :param descending: retrieve largest values instead of smallest values
            :param device: Not supported for this backend
            :return: Tuple containing the retrieved values, and their indices.
                Both ar of shape (n_queries, k)
            """
            if device is not None:
                warnings.warn('`device` is not supported for numpy operations')

            if len(values.shape) == 1:
                values = np.expand_dims(values, axis=0)

            if descending:
                values = -values

            if k >= values.shape[1]:
                idx = values.argsort(axis=1)[:, :k]
                values = np.take_along_axis(values, idx, axis=1)
            else:
                idx_ps = values.argpartition(kth=k, axis=1)[:, :k]
                values = np.take_along_axis(values, idx_ps, axis=1)
                idx_fs = values.argsort(axis=1)
                idx = np.take_along_axis(idx_ps, idx_fs, axis=1)
                values = np.take_along_axis(values, idx_fs, axis=1)

            if descending:
                values = -values

            return values, idx

    class Metrics(AbstractComputationalBackend.Metrics[np.ndarray]):
        """
        Abstract base class for metrics (distances and similarities).
        """

        @staticmethod
        def cosine_sim(
            x_mat: np.ndarray,
            y_mat: np.ndarray,
            eps: float = 1e-7,
            device: Optional[str] = None,
        ) -> np.ndarray:
            """Pairwise cosine similarities between all vectors in x_mat and y_mat.

            :param x_mat: np.ndarray of shape (n_vectors, n_dim), where n_vectors is
                the number of vectors and n_dim is the number of dimensions of each
                example.
            :param y_mat: np.ndarray of shape (n_vectors, n_dim), where n_vectors is
                the number of vectors and n_dim is the number of dimensions of each
                example.
            :param eps: a small jitter to avoid divde by zero
            :param device: Not supported for this backend
            :return: np.ndarray  of shape (n_vectors, n_vectors) containing all
                pairwise cosine distances.
                The index [i_x, i_y] contains the cosine distance between
                x_mat[i_x] and y_mat[i_y].
            """
            pass

        @classmethod
        def euclidean_dist(
            cls, x_mat: np.ndarray, y_mat: np.ndarray, device: Optional[str] = None
        ) -> np.ndarray:
            """Pairwise Euclidian distances between all vectors in x_mat and y_mat.

            :param x_mat: np.ndarray of shape (n_vectors, n_dim), where n_vectors is
                the number of vectors and n_dim is the number of dimensions of each
                example.
            :param y_mat: np.ndarray of shape (n_vectors, n_dim), where n_vectors is
                the number of vectors and n_dim is the number of dimensions of each
                example.
            :param eps: a small jitter to avoid divde by zero
            :param device: Not supported for this backend
            :return: np.ndarray  of shape (n_vectors, n_vectors) containing all
                pairwise euclidian distances.
                The index [i_x, i_y] contains the euclidian distance between
                x_mat[i_x] and y_mat[i_y].
            """
            pass

        @staticmethod
        def sqeuclidean_dist(
            x_mat: np.ndarray,
            y_mat: np.ndarray,
            device: Optional[str] = None,
        ) -> np.ndarray:
            """Pairwise Squared Euclidian distances between all vectors in
            x_mat and y_mat.

            :param x_mat: np.ndarray of shape (n_vectors, n_dim), where n_vectors is
                the number of vectors and n_dim is the number of dimensions of each
                example.
            :param y_mat: np.ndarray of shape (n_vectors, n_dim), where n_vectors is
                the number of vectors and n_dim is the number of dimensions of each
                example.
            :param device: Not supported for this backend
            :return: np.ndarray  of shape (n_vectors, n_vectors) containing all
                pairwise Squared Euclidian distances.
                The index [i_x, i_y] contains the cosine Squared Euclidian between
                x_mat[i_x] and y_mat[i_y].
            """
            pass
