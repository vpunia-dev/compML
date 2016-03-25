import numpy as np
from numpy import isscalar
import scipy.sparse as sp
from numpy import linalg,array
import numpy
import random
import warnings
def _parse_version(version_string):
    version = []
    for x in version_string.split('.'):
        try:
            version.append(int(x))
        except ValueError:
            # x may be of the form dev-1ea1592
            version.append(x)
    return tuple(version)


np_version = _parse_version(np.__version__)
def _fast_dot(A, B):
    if B.shape[0] != A.shape[A.ndim - 1]:  # check adopted from '_dotblas.c'
        raise ValueErrorcat sk.py extratrees.py modelextratrees.py > final.py

    if A.dtype != B.dtype or any(x.dtype not in (np.float32, np.float64)
                                 for x in [A, B]):
        warnings.warn('Falling back to np.dot. '
                      'Data must be of same type of either '
                      '32 or 64 bit float for the BLAS function, gemm, to be '
                      'used for an efficient dot operation. ',
                      NonBLASDotWarning)
        raise ValueError

    if min(A.shape) == 1 or min(B.shape) == 1 or A.ndim != 2 or B.ndim != 2:
        raise ValueError

    # scipy 0.9 compliant API
    dot = linalg.get_blas_funcs(['gemm'], (A, B))[0]
    A, trans_a = _impose_f_order(A)
    B, trans_b = _impose_f_order(B)
    return dot(alpha=1.0, a=A, b=B, trans_a=trans_a, trans_b=trans_b)

def _have_blas_gemm():
    try:
        linalg.get_blas_funcs(['gemm'])
        return True
    except (AttributeError, ValueError):
        warnings.warn('Could not import BLAS, falling back to np.dot')
        return False
# Only use fast_dot for older NumPy; newer ones have tackled the speed issue.
if np_version < (1, 7, 2) and _have_blas_gemm():
    def fast_dot(A, B):
        """Compute fast dot products directly calling BLAS.

        This function calls BLAS directly while warranting Fortran contiguity.
        This helps avoiding extra copies `np.dot` would have created.
        For details see section `Linear Algebra on large Arrays`:
        http://wiki.scipy.org/PerformanceTips

        Parameters
        ----------
        A, B: instance of np.ndarray
            Input arrays. Arrays are supposed to be of the same dtype and to
            have exactly 2 dimensions. Currently only floats are supported.
            In case these requirements aren't met np.dot(A, B) is returned
            instead. To activate the related warning issued in this case
            execute the following lines of code:

            >> import warnings
            >> from sklearn.exceptions import NonBLASDotWarning
            >> warnings.simplefilter('always', NonBLASDotWarning)
        """
        try:
            return _fast_dot(A, B)
        except ValueError:
            # Maltyped or malformed data.
            return np.dot(A, B)
else:
    fast_dot = np.dot
def check_is_fitted(estimator, attributes, msg=None, all_or_any=all):
    """Perform is_fitted validation for estimator.

    Checks if the estimator is fitted by verifying the presence of
    "all_or_any" of the passed attributes and raises a NotFittedError with the
    given message.

    Parameters
    ----------
    estimator : estimator instance.
        estimator instance for which the check is performed.

    attributes : attribute name(s) given as string or a list/tuple of strings
        Eg. : ["coef_", "estimator_", ...], "coef_"

    msg : string
        The default error message is, "This %(name)s instance is not fitted
        yet. Call 'fit' with appropriate arguments before using this method."

        For custom messages if "%(name)s" is present in the message string,
        it is substituted for the estimator name.

        Eg. : "Estimator, %(name)s, must be fitted before sparsifying".

    all_or_any : callable, {all, any}, default all
        Specify whether all or any of the given attributes must exist.
    """
    if msg is None:
        msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
               "appropriate arguments before using this method.")

    if not hasattr(estimator, 'fit'):
        raise TypeError("%s is not an estimator instance." % (estimator))

    if not isinstance(attributes, (list, tuple)):
        attributes = [attributes]

    if not all_or_any([hasattr(estimator, attr) for attr in attributes]):
        # FIXME NotFittedError_ --> NotFittedError in 0.19
        raise _NotFittedError(msg % {'name': type(estimator).__name__})
def as_float_array(X, copy=True, force_all_finite=True):
    """Converts an array-like to an array of floats

    The new dtype will be np.float32 or np.float64, depending on the original
    type. The function can create a copy or modify the argument depending
    on the argument copy.

    Parameters
    ----------
    X : {array-like, sparse matrix}

    copy : bool, optional
        If True, a copy of X will be created. If False, a copy may still be
        returned if X's dtype is not a floating point type.

    force_all_finite : boolean (default=True)
        Whether to raise an error on np.inf and np.nan in X.

    Returns
    -------
    XT : {array, sparse matrix}
        An array of type np.float
    """
    if isinstance(X, np.matrix) or (not isinstance(X, np.ndarray)
                                    and not sp.issparse(X)):
        return check_array(X, ['csr', 'csc', 'coo'], dtype=np.float64,
                           copy=copy, force_all_finite=force_all_finite,
                           ensure_2d=False)
    elif sp.issparse(X) and X.dtype in [np.float32, np.float64]:
        return X.copy() if copy else X
    elif X.dtype in [np.float32, np.float64]:  # is numpy array
        return X.copy('F' if X.flags['F_CONTIGUOUS'] else 'C') if copy else X
    else:
        return X.astype(np.float32 if X.dtype == np.int32 else np.float64)
def _num_samples(x):
    """Return number of samples in array-like x."""
    if hasattr(x, 'fit'):
        # Don't get num_samples from an ensembles length!
        raise TypeError('Expected sequence or array-like, got '
                        'estimator %s' % x)
    if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
        if hasattr(x, '__array__'):
            x = np.asarray(x)
        else:
            raise TypeError("Expected sequence or array-like, got %s" %
                            type(x))
    if hasattr(x, 'shape'):
        if len(x.shape) == 0:
            raise TypeError("Singleton array %r cannot be considered"
                            " a valid collection." % x)
        return x.shape[0]
    else:
        return len(x)
def _shape_repr(shape):
    """Return a platform independent representation of an array shape

    Under Python 2, the `long` type introduces an 'L' suffix when using the
    default %r format for tuples of integers (typically used to store the shape
    of an array).

    Under Windows 64 bit (and Python 2), the `long` type is used by default
    in numpy shapes even when the integer dimensions are well below 32 bit.
    The platform specific type causes string messages or doctests to change
    from one platform to another which is not desirable.

    Under Python 3, there is no more `long` type so the `L` suffix is never
    introduced in string representation.

    >>> _shape_repr((1, 2))
    '(1, 2)'
    >>> one = 2 ** 64 / 2 ** 64  # force an upcast to `long` under Python 2
    >>> _shape_repr((one, 2 * one))
    '(1, 2)'
    >>> _shape_repr((1,))
    '(1,)'
    >>> _shape_repr(())
    '()'
    """
    if len(shape) == 0:
        return "()"
    joined = ", ".join("%d" % e for e in shape)
    if len(shape) == 1:
        # special notation for singleton tuples
        joined += ','
    return "(%s)" % joined
def _assert_all_finite(X):
    """Like assert_all_finite, but only for ndarray."""
    X = np.asanyarray(X)
    # First try an O(n) time, O(1) space solution for the common case that
    # everything is finite; fall back to O(n) space np.isfinite to prevent
    # false positives from overflow in sum method.
    if (X.dtype.char in np.typecodes['AllFloat'] and not np.isfinite(X.sum())
            and not np.isfinite(X).all()):
        raise ValueError("Input contains NaN, infinity"
                         " or a value too large for %r." % X.dtype)
string_types = basestring
from scipy import sparse
def _incremental_mean_and_var(X, last_mean=.0, last_variance=None,
                              last_sample_count=0):
    """Calculate mean update and a Youngs and Cramer variance update.

    last_mean and last_variance are statistics computed at the last step by the
    function. Both must be initialized to 0.0. In case no scaling is required
    last_variance can be None. The mean is always required and returned because
    necessary for the calculation of the variance. last_n_samples_seen is the
    number of samples encountered until now.

    From the paper "Algorithms for computing the sample variance: analysis and
    recommendations", by Chan, Golub, and LeVeque.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data to use for variance update

    last_mean : array-like, shape: (n_features,)

    last_variance : array-like, shape: (n_features,)

    last_sample_count : int

    Returns
    -------
    updated_mean : array, shape (n_features,)

    updated_variance : array, shape (n_features,)
        If None, only mean is computed

    updated_sample_count : int

    References
    ----------
    T. Chan, G. Golub, R. LeVeque. Algorithms for computing the sample
        variance: recommendations, The American Statistician, Vol. 37, No. 3,
        pp. 242-247

    Also, see the sparse implementation of this in
    `utils.sparsefuncs.incr_mean_variance_axis` and
    `utils.sparsefuncs_fast.incr_mean_variance_axis0`
    """
    # old = stats until now
    # new = the current increment
    # updated = the aggregated stats
    last_sum = last_mean * last_sample_count
    new_sum = X.sum(axis=0)

    new_sample_count = X.shape[0]
    updated_sample_count = last_sample_count + new_sample_count

    updated_mean = (last_sum + new_sum) / updated_sample_count

    if last_variance is None:
        updated_variance = None
    else:
        new_unnormalized_variance = X.var(axis=0) * new_sample_count
        if last_sample_count == 0:  # Avoid division by 0
            updated_unnormalized_variance = new_unnormalized_variance
        else:
            last_over_new_count = last_sample_count / new_sample_count
            last_unnormalized_variance = last_variance * last_sample_count
            updated_unnormalized_variance = (
                last_unnormalized_variance +
                new_unnormalized_variance +
                last_over_new_count / updated_sample_count *
                (last_sum / last_over_new_count - new_sum) ** 2)
        updated_variance = updated_unnormalized_variance / updated_sample_count

    return updated_mean, updated_variance, updated_sample_count
def check_array(array, accept_sparse=None, dtype="numeric", order=None,
                copy=False, force_all_finite=True, ensure_2d=True,
                allow_nd=False, ensure_min_samples=1, ensure_min_features=1,
                warn_on_dtype=False, estimator=None):
    """Input validation on an array, list, sparse matrix or similar.

    By default, the input is converted to an at least 2nd numpy array.
    If the dtype of the array is object, attempt converting to float,
    raising on failure.

    Parameters
    ----------
    array : object
        Input object to check / convert.

    accept_sparse : string, list of string or None (default=None)
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc.  None means that sparse matrix input will raise an error.
        If the input is sparse but not in the allowed format, it will be
        converted to the first listed format.

    dtype : string, type, list of types or None (default="numeric")
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
        If dtype is a list of types, conversion on the first type is only
        performed if the dtype of the input is not in the list.

    order : 'F', 'C' or None (default=None)
        Whether an array will be forced to be fortran or c-style.

    copy : boolean (default=False)
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    force_all_finite : boolean (default=True)
        Whether to raise an error on np.inf and np.nan in X.

    ensure_2d : boolean (default=True)
        Whether to make X at least 2d.

    allow_nd : boolean (default=False)
        Whether to allow X.ndim > 2.

    ensure_min_samples : int (default=1)
        Make sure that the array has a minimum number of samples in its first
        axis (rows for a 2D array). Setting to 0 disables this check.

    ensure_min_features : int (default=1)
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when the input data has effectively 2
        dimensions or is originally 1D and ``ensure_2d`` is True. Setting to 0
        disables this check.

    warn_on_dtype : boolean (default=False)
        Raise DataConversionWarning if the dtype of the input data structure
        does not match the requested dtype, causing a memory copy.

    estimator : str or estimator instance (default=None)
        If passed, include the name of the estimator in warning messages.

    Returns
    -------
    X_converted : object
        The converted and validated X.
    """
    if isinstance(accept_sparse, str):
        accept_sparse = [accept_sparse]

    # store whether originally we wanted numeric dtype
    dtype_numeric = dtype == "numeric"

    dtype_orig = getattr(array, "dtype", None)
    if not hasattr(dtype_orig, 'kind'):
        # not a data type (e.g. a column named dtype in a pandas DataFrame)
        dtype_orig = None

    if dtype_numeric:
        if dtype_orig is not None and dtype_orig.kind == "O":
            # if input is object, convert to float.
            dtype = np.float64
        else:
            dtype = None

    if isinstance(dtype, (list, tuple)):
        if dtype_orig is not None and dtype_orig in dtype:
            # no dtype conversion required
            dtype = None
        else:
            # dtype conversion required. Let's select the first element of the
            # list of accepted types.
            dtype = dtype[0]

    if estimator is not None:
        if isinstance(estimator, string_types):
            estimator_name = estimator
        else:
            estimator_name = estimator.__class__.__name__
    else:
        estimator_name = "Estimator"
    context = " by %s" % estimator_name if estimator is not None else ""

    if sp.issparse(array):
        array = _ensure_sparse_format(array, accept_sparse, dtype, copy,
                                      force_all_finite)
    else:
        array = np.array(array, dtype=dtype, order=order, copy=copy)

        if ensure_2d:
            if array.ndim == 1:
                if ensure_min_samples >= 2:
                    raise ValueError("%s expects at least 2 samples provided "
                                     "in a 2 dimensional array-like input"
                                     % estimator_name)
                warnings.warn(
                    "Passing 1d arrays as data is deprecated in 0.17 and will "
                    "raise ValueError in 0.19. Reshape your data either using "
                    "X.reshape(-1, 1) if your data has a single feature or "
                    "X.reshape(1, -1) if it contains a single sample.",
                    DeprecationWarning)
            array = np.atleast_2d(array)
            # To ensure that array flags are maintained
            array = np.array(array, dtype=dtype, order=order, copy=copy)

        # make sure we actually converted to numeric:
        if dtype_numeric and array.dtype.kind == "O":
            array = array.astype(np.float64)
        if not allow_nd and array.ndim >= 3:
            raise ValueError("Found array with dim %d. %s expected <= 2."
                             % (array.ndim, estimator_name))
        if force_all_finite:
            _assert_all_finite(array)

    shape_repr = _shape_repr(array.shape)
    if ensure_min_samples > 0:
        n_samples = _num_samples(array)
        if n_samples < ensure_min_samples:
            raise ValueError("Found array with %d sample(s) (shape=%s) while a"
                             " minimum of %d is required%s."
                             % (n_samples, shape_repr, ensure_min_samples,
                                context))

    if ensure_min_features > 0 and array.ndim == 2:
        n_features = array.shape[1]
        if n_features < ensure_min_features:
            raise ValueError("Found array with %d feature(s) (shape=%s) while"
                             " a minimum of %d is required%s."
                             % (n_features, shape_repr, ensure_min_features,
                                context))

    if warn_on_dtype and dtype_orig is not None and array.dtype != dtype_orig:
        msg = ("Data with input dtype %s was converted to %s%s."
               % (dtype_orig, array.dtype, context))
        warnings.warn(msg, _DataConversionWarning)
    return array

class TransformerMixin(object):
    """Mixin class for all transformers in scikit-learn."""

    def fit_transform(self, X, y=None, **fit_params):
        """Fit to data, then transform it.

        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.

        Parameters
        ----------
        X : numpy array of shape [n_samples, n_features]
            Training set.

        y : numpy array of shape [n_samples]
            Target values.

        Returns
        -------
        X_new : numpy array of shape [n_samples, n_features_new]
            Transformed array.

        """
        # non-optimized default implementation; override when a better
        # method is possible for a given clustering algorithm
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.fit(X, **fit_params).transform(X)
        else:
            # fit method of arity 2 (supervised transformation)
            return self.fit(X, y, **fit_params).transform(X)
class BaseEstimator(object):
    """Base class for all estimators in scikit-learn

    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("scikit-learn estimators should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep: boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            # We need deprecation warnings to always be on in order to
            # catch deprecated param values.
            # This is set in utils/__init__.py but it gets overwritten
            # when running under python3 somehow.
            warnings.simplefilter("always", DeprecationWarning)
            try:
                with warnings.catch_warnings(record=True) as w:
                    value = getattr(self, key, None)
                if len(w) and w[0].category == DeprecationWarning:
                    # if the parameter is deprecated, don't show it
                    continue
            finally:
                warnings.filters.pop(0)

            # XXX: should we rather test if instance of estimator?
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The former have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Returns
        -------
        self
        """
        if not params:
            # Simple optimisation to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)
        for key, value in six.iteritems(params):
            split = key.split('__', 1)
            if len(split) > 1:
                # nested objects case
                name, sub_name = split
                if name not in valid_params:
                    raise ValueError('Invalid parameter %s for estimator %s. '
                                     'Check the list of available parameters '
                                     'with `estimator.get_params().keys()`.' %
                                     (name, self))
                sub_object = valid_params[name]
                sub_object.set_params(**{sub_name: value})
            else:
                # simple objects case
                if key not in valid_params:
                    raise ValueError('Invalid parameter %s for estimator %s. '
                                     'Check the list of available parameters '
                                     'with `estimator.get_params().keys()`.' %
                                     (key, self.__class__.__name__))
                setattr(self, key, value)
        return self

    def __repr__(self):
        class_name = self.__class__.__name__
        return '%s(%s)' % (class_name, _pprint(self.get_params(deep=False),
                                               offset=len(class_name),),)

def _handle_zeros_in_scale(scale, copy=True):
    ''' Makes sure that whenever scale is zero, we handle it correctly.

    This happens in most scalers when we have constant features.'''

    # if we are fitting on 1D arrays, scale might be a scalar
    if np.isscalar(scale):
        if scale == .0:
            scale = 1.
        return scale
    elif isinstance(scale, np.ndarray):
        if copy:
            # New array to avoid side-effects
            scale = scale.copy()
        scale[scale == 0.0] = 1.0
        return scale
class StandardScaler(BaseEstimator, TransformerMixin):
    """Standardize features by removing the mean and scaling to unit variance

    Centering and scaling happen independently on each feature by computing
    the relevant statistics on the samples in the training set. Mean and
    standard deviation are then stored to be used on later data using the
    `transform` method.

    Standardization of a dataset is a common requirement for many
    machine learning estimators: they might behave badly if the
    individual feature do not more or less look like standard normally
    distributed data (e.g. Gaussian with 0 mean and unit variance).

    For instance many elements used in the objective function of
    a learning algorithm (such as the RBF kernel of Support Vector
    Machines or the L1 and L2 regularizers of linear models) assume that
    all features are centered around 0 and have variance in the same
    order. If a feature has a variance that is orders of magnitude larger
    that others, it might dominate the objective function and make the
    estimator unable to learn from other features correctly as expected.

    This scaler can also be applied to sparse CSR or CSC matrices by passing
    `with_mean=False` to avoid breaking the sparsity structure of the data.

    Read more in the :ref:`User Guide <preprocessing_scaler>`.

    Parameters
    ----------
    with_mean : boolean, True by default
        If True, center the data before scaling.
        This does not work (and will raise an exception) when attempted on
        sparse matrices, because centering them entails building a dense
        matrix which in common use cases is likely to be too large to fit in
        memory.

    with_std : boolean, True by default
        If True, scale the data to unit variance (or equivalently,
        unit standard deviation).

    copy : boolean, optional, default True
        If False, try to avoid a copy and do inplace scaling instead.
        This is not guaranteed to always work inplace; e.g. if the data is
        not a NumPy array or scipy.sparse CSR matrix, a copy may still be
        returned.

    Attributes
    ----------
    scale_ : ndarray, shape (n_features,)
        Per feature relative scaling of the data.

        .. versionadded:: 0.17
           *scale_* is recommended instead of deprecated *std_*.

    mean_ : array of floats with shape [n_features]
        The mean value for each feature in the training set.

    var_ : array of floats with shape [n_features]
        The variance for each feature in the training set. Used to compute
        `scale_`

    n_samples_seen_ : int
        The number of samples processed by the estimator. Will be reset on
        new calls to fit, but increments across ``partial_fit`` calls.

    See also
    --------
    :func:`sklearn.preprocessing.scale` to perform centering and
    scaling without using the ``Transformer`` object oriented API

    :class:`sklearn.decomposition.RandomizedPCA` with `whiten=True`
    to further remove the linear correlation across features.
    """

    def __init__(self, copy=True, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy = copy

    @property
    def std_(self):
        return self.scale_

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.

        __init__ parameters are not touched.
        """

        # Checking one attribute is enough, becase they are all set together
        # in partial_fit
        if hasattr(self, 'scale_'):
            del self.scale_
            del self.n_samples_seen_
            del self.mean_
            del self.var_

    def fit(self, X, y=None):
        """Compute the mean and std to be used for later scaling.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.

        y: Passthrough for ``Pipeline`` compatibility.
        """

        # Reset internal state before fitting
        self._reset()
        return self.partial_fit(X, y)

    def partial_fit(self, X, y=None):
        """Online computation of mean and std on X for later scaling.
        All of X is processed as a single batch. This is intended for cases
        when `fit` is not feasible due to very large number of `n_samples`
        or because X is read from a continuous stream.

        The algorithm for incremental mean and std is given in Equation 1.5a,b
        in Chan, Tony F., Gene H. Golub, and Randall J. LeVeque. "Algorithms
        for computing the sample variance: Analysis and recommendations."
        The American Statistician 37.3 (1983): 242-247:

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.

        y: Passthrough for ``Pipeline`` compatibility.
        """
        X = check_array(X, accept_sparse=('csr', 'csc'), copy=self.copy,
                        ensure_2d=False, warn_on_dtype=True,
                        estimator=self, dtype=FLOAT_DTYPES)

        if X.ndim == 1:
            warnings.warn(DEPRECATION_MSG_1D, DeprecationWarning)

        # Even in the case of `with_mean=False`, we update the mean anyway
        # This is needed for the incremental computation of the var
        # See incr_mean_variance_axis and _incremental_mean_variance_axis

        if sparse.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot center sparse matrices: pass `with_mean=False` "
                    "instead. See docstring for motivation and alternatives.")
            if self.with_std:
                # First pass
                if not hasattr(self, 'n_samples_seen_'):
                    self.mean_, self.var_ = mean_variance_axis(X, axis=0)
                    self.n_samples_seen_ = X.shape[0]
                # Next passes
                else:
                    self.mean_, self.var_, self.n_samples_seen_ = \
                        incr_mean_variance_axis(X, axis=0,
                                                last_mean=self.mean_,
                                                last_var=self.var_,
                                                last_n=self.n_samples_seen_)
            else:
                self.mean_ = None
                self.var_ = None
        else:
            # First pass
            if not hasattr(self, 'n_samples_seen_'):
                self.mean_ = .0
                self.n_samples_seen_ = 0
                if self.with_std:
                    self.var_ = .0
                else:
                    self.var_ = None

            self.mean_, self.var_, self.n_samples_seen_ = \
                _incremental_mean_and_var(X, self.mean_, self.var_,
                                          self.n_samples_seen_)

        if self.with_std:
            self.scale_ = _handle_zeros_in_scale(np.sqrt(self.var_))
        else:
            self.scale_ = None

        return self

    def transform(self, X, y=None, copy=None):
        """Perform standardization by centering and scaling

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to scale along the features axis.
        """
        check_is_fitted(self, 'scale_')

        copy = copy if copy is not None else self.copy
        X = check_array(X, accept_sparse='csr', copy=copy,
                        ensure_2d=False, warn_on_dtype=True,
                        estimator=self, dtype=FLOAT_DTYPES)

        if X.ndim == 1:
            warnings.warn(DEPRECATION_MSG_1D, DeprecationWarning)

        if sparse.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot center sparse matrices: pass `with_mean=False` "
                    "instead. See docstring for motivation and alternatives.")
            if self.scale_ is not None:
                inplace_column_scale(X, 1 / self.scale_)
        else:
            if self.with_mean:
                X -= self.mean_
            if self.with_std:
                X /= self.scale_
        return X

    def inverse_transform(self, X, copy=None):
        """Scale back the data to the original representation

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to scale along the features axis.
        """
        check_is_fitted(self, 'scale_')

        copy = copy if copy is not None else self.copy
        if sparse.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot uncenter sparse matrices: pass `with_mean=False` "
                    "instead See docstring for motivation and alternatives.")
            if not sparse.isspmatrix_csr(X):
                X = X.tocsr()
                copy = False
            if copy:
                X = X.copy()
            if self.scale_ is not None:
                inplace_column_scale(X, self.scale_)
        else:
            X = np.asarray(X)
            if copy:
                X = X.copy()
            if self.with_std:
                X *= self.scale_
            if self.with_mean:
                X += self.mean_
        return X

FLOAT_DTYPES = (np.float64, np.float32, np.float16)
class PCA(BaseEstimator, TransformerMixin):
    """Principal component analysis (PCA)

    Linear dimensionality reduction using Singular Value Decomposition of the
    data and keeping only the most significant singular vectors to project the
    data to a lower dimensional space.

    This implementation uses the scipy.linalg implementation of the singular
    value decomposition. It only works for dense arrays and is not scalable to
    large dimensional data.

    The time complexity of this implementation is ``O(n ** 3)`` assuming
    n ~ n_samples ~ n_features.

    Read more in the :ref:`User Guide <PCA>`.

    Parameters
    ----------
    n_components : int, None or string
        Number of components to keep.
        if n_components is not set all components are kept::

            n_components == min(n_samples, n_features)

        if n_components == 'mle', Minka\'s MLE is used to guess the dimension
        if ``0 < n_components < 1``, select the number of components such that
        the amount of variance that needs to be explained is greater than the
        percentage specified by n_components

    copy : bool
        If False, data passed to fit are overwritten and running
        fit(X).transform(X) will not yield the expected results,
        use fit_transform(X) instead.

    whiten : bool, optional
        When True (False by default) the `components_` vectors are divided
        by n_samples times singular values to ensure uncorrelated outputs
        with unit component-wise variances.

        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometime
        improve the predictive accuracy of the downstream estimators by
        making there data respect some hard-wired assumptions.

    Attributes
    ----------
    components_ : array, [n_components, n_features]
        Principal axes in feature space, representing the directions of
        maximum variance in the data. The components are sorted by
        explained_variance_.

    explained_variance_ : array, [n_components]
        The amount of variance explained by each of the selected components.

    explained_variance_ratio_ : array, [n_components]
        Percentage of variance explained by each of the selected components.

        If ``n_components`` is not set then all components are stored and the
        sum of explained variances is equal to 1.0.

    mean_ : array, [n_features]
        Per-feature empirical mean, estimated from the training set.

        Equal to `X.mean(axis=1)`.

    n_components_ : int
        The estimated number of components. Relevant when `n_components` is set
        to 'mle' or a number between 0 and 1 to select using explained
        variance.

    noise_variance_ : float
        The estimated noise covariance following the Probabilistic PCA model
        from Tipping and Bishop 1999. See "Pattern Recognition and
        Machine Learning" by C. Bishop, 12.2.1 p. 574 or
        http://www.miketipping.com/papers/met-mppca.pdf. It is required to
        computed the estimated data covariance and score samples.

    Notes
    -----
    For n_components='mle', this class uses the method of `Thomas P. Minka:
    Automatic Choice of Dimensionality for PCA. NIPS 2000: 598-604`

    Implements the probabilistic PCA model from:
    M. Tipping and C. Bishop, Probabilistic Principal Component Analysis,
    Journal of the Royal Statistical Society, Series B, 61, Part 3, pp. 611-622
    via the score and score_samples methods.
    See http://www.miketipping.com/papers/met-mppca.pdf

    Due to implementation subtleties of the Singular Value Decomposition (SVD),
    which is used in this implementation, running fit twice on the same matrix
    can lead to principal components with signs flipped (change in direction).
    For this reason, it is important to always use the same estimator object to
    transform data in a consistent fashion.

    Examples
    --------

    >>> import numpy as np
    >>> from sklearn.decomposition import PCA
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> pca = PCA(n_components=2)
    >>> pca.fit(X)
    PCA(copy=True, n_components=2, whiten=False)
    >>> print(pca.explained_variance_) # doctest: +ELLIPSIS
    [ 6.6162...  0.05038...]
    >>> print(pca.explained_variance_ratio_) # doctest: +ELLIPSIS
    [ 0.99244...  0.00755...]

    See also
    --------
    RandomizedPCA
    KernelPCA
    SparsePCA
    TruncatedSVD
    """
    def __init__(self, n_components=None, copy=True, whiten=False):
        self.n_components = n_components
        self.copy = copy
        self.whiten = whiten

    def fit(self, X, y=None):
        """Fit the model with X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._fit(X)
        return self

    def fit_transform(self, X, y=None):
        """Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)

        """
        U, S, V = self._fit(X)
        U = U[:, :self.n_components_]

        if self.whiten:
            # X_new = X * V / S * sqrt(n_samples) = U * sqrt(n_samples)
            U *= sqrt(X.shape[0])
        else:
            # X_new = X * V = U * S * V^T * V = U * S
            U *= S[:self.n_components_]

        return U

    def _fit(self, X):
        """Fit the model on X

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        Returns
        -------
        U, s, V : ndarrays
            The SVD of the input data, copied and centered when
            requested.
        """
        X = check_array(X)
        n_samples, n_features = X.shape
        X = as_float_array(X, copy=self.copy)
        # Center data
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_
        U, S, V = linalg.svd(X, full_matrices=False)
        explained_variance_ = (S ** 2) / n_samples
        explained_variance_ratio_ = (explained_variance_ /
                                     explained_variance_.sum())

        components_ = V

        n_components = self.n_components
        if n_components is None:
            n_components = n_features
        elif n_components == 'mle':
            if n_samples < n_features:
                raise ValueError("n_components='mle' is only supported "
                                 "if n_samples >= n_features")

            n_components = _infer_dimension_(explained_variance_,
                                             n_samples, n_features)
        elif not 0 <= n_components <= n_features:
            raise ValueError("n_components=%r invalid for n_features=%d"
                             % (n_components, n_features))

        if 0 < n_components < 1.0:
            # number of components for which the cumulated explained variance
            # percentage is superior to the desired threshold
            ratio_cumsum = explained_variance_ratio_.cumsum()
            n_components = np.sum(ratio_cumsum < n_components) + 1

        # Compute noise covariance using Probabilistic PCA model
        # The sigma2 maximum likelihood (cf. eq. 12.46)
        if n_components < min(n_features, n_samples):
            self.noise_variance_ = explained_variance_[n_components:].mean()
        else:
            self.noise_variance_ = 0.

        # store n_samples to revert whitening when getting covariance
        self.n_samples_ = n_samples

        self.components_ = components_[:n_components]
        self.explained_variance_ = explained_variance_[:n_components]
        explained_variance_ratio_ = explained_variance_ratio_[:n_components]
        self.explained_variance_ratio_ = explained_variance_ratio_
        self.n_components_ = n_components

        return (U, S, V)

    def get_covariance(self):
        """Compute data covariance with the generative model.

        ``cov = components_.T * S**2 * components_ + sigma2 * eye(n_features)``
        where  S**2 contains the explained variances.

        Returns
        -------
        cov : array, shape=(n_features, n_features)
            Estimated covariance of data.
        """
        components_ = self.components_
        exp_var = self.explained_variance_
        if self.whiten:
            components_ = components_ * np.sqrt(exp_var[:, np.newaxis])
        exp_var_diff = np.maximum(exp_var - self.noise_variance_, 0.)
        cov = np.dot(components_.T * exp_var_diff, components_)
        cov.flat[::len(cov) + 1] += self.noise_variance_  # modify diag inplace
        return cov

    def get_precision(self):
        """Compute data precision matrix with the generative model.

        Equals the inverse of the covariance but computed with
        the matrix inversion lemma for efficiency.

        Returns
        -------
        precision : array, shape=(n_features, n_features)
            Estimated precision of data.
        """
        n_features = self.components_.shape[1]

        # handle corner cases first
        if self.n_components_ == 0:
            return np.eye(n_features) / self.noise_variance_
        if self.n_components_ == n_features:
            return linalg.inv(self.get_covariance())

        # Get precision using matrix inversion lemma
        components_ = self.components_
        exp_var = self.explained_variance_
        exp_var_diff = np.maximum(exp_var - self.noise_variance_, 0.)
        precision = np.dot(components_, components_.T) / self.noise_variance_
        precision.flat[::len(precision) + 1] += 1. / exp_var_diff
        precision = np.dot(components_.T,
                           np.dot(linalg.inv(precision), components_))
        precision /= -(self.noise_variance_ ** 2)
        precision.flat[::len(precision) + 1] += 1. / self.noise_variance_
        return precision

    def transform(self, X):
        """Apply the dimensionality reduction on X.

        X is projected on the first principal components previous extracted
        from a training set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)

        """
        check_is_fitted(self, 'mean_')

        X = check_array(X)
        if self.mean_ is not None:
            X = X - self.mean_
        X_transformed = fast_dot(X, self.components_.T)
        if self.whiten:
            X_transformed /= np.sqrt(self.explained_variance_)
        return X_transformed

    def inverse_transform(self, X):
        """Transform data back to its original space using `n_components_`.

        Returns an input X_original whose transform would be X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)
            New data, where n_samples is the number of samples
            and n_components is the number of components. X represents
            data from the projection on to the principal components.

        Returns
        -------
        X_original array-like, shape (n_samples, n_features)
        """
        check_is_fitted(self, 'mean_')

        if self.whiten:
            return fast_dot(
                X,
                np.sqrt(self.explained_variance_[:, np.newaxis]) *
                self.components_) + self.mean_
        else:
            return fast_dot(X, self.components_) + self.mean_

    def score_samples(self, X):
        """Return the log-likelihood of each sample.

        See. "Pattern Recognition and Machine Learning"
        by C. Bishop, 12.2.1 p. 574
        or http://www.miketipping.com/papers/met-mppca.pdf

        Parameters
        ----------
        X: array, shape(n_samples, n_features)
            The data.

        Returns
        -------
        ll: array, shape (n_samples,)
            Log-likelihood of each sample under the current model
        """
        check_is_fitted(self, 'mean_')

        X = check_array(X)
        Xr = X - self.mean_
        n_features = X.shape[1]
        log_like = np.zeros(X.shape[0])
        precision = self.get_precision()
        log_like = -.5 * (Xr * (np.dot(Xr, precision))).sum(axis=1)
        log_like -= .5 * (n_features * log(2. * np.pi)
                          - fast_logdet(precision))
        return log_like

    def score(self, X, y=None):
        """Return the average log-likelihood of all samples.

        See. "Pattern Recognition and Machine Learning"
        by C. Bishop, 12.2.1 p. 574
        or http://www.miketipping.com/papers/met-mppca.pdf

        Parameters
        ----------
        X: array, shape(n_samples, n_features)
            The data.

        Returns
        -------
        ll: float
            Average log-likelihood of the samples under the current model
        """
        return np.mean(self.score_samples(X))
from collections import namedtuple, Counter, defaultdict
from math import log
from itertools import groupby
import random
import numpy as np

class RandomDecisionTree(object):

		def __init__(self,targets,max_features=-1,max_depth=-1,min_samples=-1):
				self.root_node = None
				self.max_features = max_features
				self.max_depth = max_depth
				self.min_samples = min_samples
				self.rfeature = random.Random(1)
				self.rfeature2 = random.Random(3)
				self.rvalue = random.Random(2)
				self.targets = targets

		def fit(self, samples, target):
				training_samples = [TrainingSample(s, t)
														for s, t in zip(samples, target)]
				predicting_features = list(range(len(samples[0])))
				if self.max_features == -1:
					self.max_features = len(predicting_features)
				self.root_node = self.create_decision_tree(training_samples,
																									 predicting_features)

		def predict(self, X, allow_unclassified=False):
				default_klass = 1
				predicted_klasses = []

				for sample in X:
						klass = None
						current_node = self.root_node
						while klass is None:
								if current_node.is_leaf():
										klass = current_node.klass
								else:
										key_value = sample[current_node.feature]
										split_val = current_node.split_val
										if key_value < split_val:
												current_node = current_node[split_val][0]
										else:
												current_node = current_node[split_val][1]
						predicted_klasses.append(klass)
				return predicted_klasses

		def create_decision_tree(self, training_samples, predicting_features,count=0):
				if not predicting_features:
						# No more predicting features
						default_klass = self.get_most_common_class(training_samples)
						root_node = DecisionTreeLeaf(default_klass)
				else:
						klasses = [sample.klass for sample in training_samples]
						all_same = True 
						for f in predicting_features:
							for s1,s2 in zip(training_samples,training_samples[1:]):
								if s1.sample[f] != s2.sample[f]:
									all_same = False
									break	
						if len(set(klasses)) == 1:
							target_klass = training_samples[0].klass
							root_node = DecisionTreeLeaf(target_klass)
						elif len(training_samples) <= self.min_samples:
							default_klass = self.get_most_common_class(training_samples)
							root_node = DecisionTreeLeaf(default_klass)
						elif self.max_depth!=-1 and count==self.max_depth:
							default_klass = self.get_most_common_class(training_samples)
							root_node = DecisionTreeLeaf(default_klass)
						else:	
							subspace = random.sample(predicting_features,self.max_features)
							best_feature,feature_value = self.select_best_feature(training_samples,subspace,klasses)
							#best_feature = self.rfeature2.choice(subspace)
							best_feature_values = {s.sample[best_feature] for s in training_samples}
							if len(best_feature_values)==1:
								default_klass = self.get_most_common_class(training_samples)
								root_node = DecisionTreeLeaf(default_klass)
							else:
								#feature_value = self.rvalue.uniform(min(best_feature_values),max(best_feature_values))
								root_node = DecisionTreeNode(best_feature,feature_value)
								lsamples = [s for s in training_samples if s.sample[best_feature] < feature_value]
								rsamples = [s for s in training_samples if s.sample[best_feature] >= feature_value]
								lchild = self.create_decision_tree(lsamples,predicting_features,count+1)
								rchild = self.create_decision_tree(rsamples,predicting_features,count+1)
								root_node[feature_value] = (lchild,rchild)
				return root_node

		@staticmethod
		def get_most_common_class(trainning_samples):
				klasses = [s.klass for s in trainning_samples]
				counter = Counter(klasses)
				k, = counter.most_common(n=1)
				return k[0]

		def select_best_feature(self, samples, features, klasses):
				gain_factors = [(self.gini_impurity(samples, feat, klasses)+tuple([feat]))
												for feat in features]
				gain_factors.sort()
				best_feature_value = gain_factors[0][1]
				best_feature = gain_factors[0][2]
				return best_feature,best_feature_value

		def friedman(self, samples, feature, klasses):
				values = {s.sample[feature] for s in samples}
				p = self.rvalue.uniform(min(values),max(values))
				reg = 200
				leaf = 10

				lclasses = [-s.klass for s in samples if s.sample[feature] < p]
				rclasses = [-s.klass for s in samples if s.sample[feature] >= p]
				gl = sum(lclasses)**2
				gr = sum(rclasses)**2
				hl = 2*len(lclasses)
				hr = 2*len(rclasses)
				leftscore = float(1.0*(gl**2)/(hl+reg))
				rightscore = float(1.0*(gr**2)/(hr+reg))
				parentscore = float(1.0* (gl+gr)**2/(hl+hr+reg))
				
				gain = leftscore + rightscore - parentscore - leaf
				return (gain,p)

		def sklearn_gini(self, samples, feature, klasses):
				values = {s.sample[feature] for s in samples}
				partition = self.rvalue.uniform(min(values),max(values))
				lclasses = [s.klass for s in samples if s.sample[feature] < partition]
				rclasses = [s.klass for s in samples if s.sample[feature] >= partition]
				lscore=0;rscore=0;pscore=0
				if len(lclasses)!=0:
					lscore = self.skgini(lclasses)
				if len(rclasses)!=0:
					rscore = self.skgini(rclasses)
				pscore = self.skgini(klasses)

				gini = lscore + rscore - pscore
				return (gini,partition)
		def skgini(self,data):
				score = 0.0
				for t in self.targets:
					pk = float(1.0*data.count(t)/len(data)	)
					score += pk*(1-pk)	
				return score
		def gini_impurity(self, samples, feature, klasses):
				N = len(samples)
				values = {s.sample[feature] for s in samples}
				partition = random.uniform(min(values),max(values))
				lclasses = [s.klass for s in samples if s.sample[feature] < partition]
				rclasses = [s.klass for s in samples if s.sample[feature] >= partition]
				lgini = self.gini(lclasses)
				rgini = self.gini(rclasses)
				parentgini = self.gini(klasses)
				pl = float(1.0*len(lclasses)/len(klasses))
				pr = float(1.0*len(rclasses)/len(klasses))

				return (parentgini - (pl*lgini) - (pr*rgini),partition)

		def information_gain(self, samples, feature, klasses):
				N = len(samples)
				values = [s.sample[feature] for s in samples]
				partition = self.rvalue.uniform(min(values),max(values))
				lclasses = [s.klass for s in samples if s.sample[feature] < partition]
				rclasses = [s.klass for s in samples if s.sample[feature] >= partition]
				lentropy = self.entropy(lclasses)
				rentropy = self.entropy(rclasses)
				parentropy = self.entropy(klasses)
				pl = float(1.0*len(lclasses)/len(klasses))
				pr = float(1.0*len(rclasses)/len(klasses))

				return (parentropy - (pl*lentropy) - (pr*rentropy),partition)

		@staticmethod
		def gini(dataset):
				N = len(dataset)
				counter = Counter(dataset)
				return 1.0 - sum(-1.0 * ((counter[k]/N)**2) for k in counter)

		@staticmethod
		def entropy(dataset):
				N = len(dataset)
				counter = Counter(dataset)
				return sum(-1.0*(counter[k] / N)*log( float(1.0*counter[k] / N),2) for k in counter)

TrainingSample = namedtuple('TrainingSample', ('sample', 'klass'))


class DecisionTreeNode(dict):
    def __init__(self, feature, split_val = -1,*args, **kwargs):
        self.feature = feature
        self.split_val = split_val 
        super(DecisionTreeNode, self).__init__(*args, **kwargs)

    def is_leaf(self):
        return False

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.feature)


class DecisionTreeLeaf(dict):
    def __init__(self, klass, *args, **kwargs):
        self.klass = klass
        super(DecisionTreeLeaf, self).__init__(*args, **kwargs)

    def is_leaf(self):
        return True

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.klass)

class ExtraTrees():
	def __init__(self,n_estimators=10,max_features=3,max_depth=-1,min_samples_split=5,subsample=10,targets=[-1,0,1]):
		self.n_estimators = n_estimators
		self.max_features = max_features
		self.max_depth = max_depth
		self.min_samples_split = min_samples_split
		self.subsample = subsample
		self.estimators = []
		self.r = random.Random(4)
		self.targets = targets

	def fit(self,X,y):
		trainingsubseti = range(len(X))
		for n in range(self.n_estimators):
			'''
			trainingsubseti = range(len(X))
			indices = self.r.sample(trainingsubseti,self.subsample*len(trainingsubseti)/10)
			trainxn = [X[i] for i in indices]
			trainyn = [y[i] for i in indices]
			'''
			clf = RandomDecisionTree(max_features=self.max_features,max_depth=self.max_depth,min_samples=self.min_samples_split,targets=self.targets)
			clf.fit(X,y)
			self.estimators.append(clf)

	def predict(self,X):
		xpred = []
		for x in X:
			predictions = []	
			for clf in self.estimators:	
				pred = clf.predict([x])
				predictions.append(pred[0])
			counts = [len(list(group)) for k,group in groupby(predictions)]
			p = [list(group) for k,group in groupby(predictions)]
			maxindex = counts.index(max(counts))
			
			xpred.append(p[maxindex][0])	

		return xpred
from datetime import datetime
import random
import numpy as np
from collections import Counter,defaultdict
thresholdyes = 0.05
thresholdno = -0.05
trainx = []
trainy = []

def l2reg(x):
	return np.dot(x,x)

def l1reg(x):
	return sum(x)

def errorfunc(w):
	totalerror = 0

	for i in range(len(trainx)):
		x = trainx[i]
		pred = np.dot(w,x[1:])

		if pred < -0.1:
			pred = -1
		elif pred > 0.1:
			pred = 1
		else:	
			pred = 0

		if trainy[i]==0 and pred==-1:
			totalerror += 0.2	

		if trainy[i]==1 and pred==-1:
			totalerror += 0.7

		if trainy[i]==-1 and pred==0:
			totalerror += 0.5

		if trainy[i]==1 and pred==0:
			totalerror += 0.01

		if trainy[i]==-1 and pred==1:
			totalerror += 1.0

		if trainy[i]==0 and pred==1:
			totalerror += 0.01

	return totalerror+l1reg(w)

def removeDuplicates(X,y):
	dictX = {}
	culprits = []
	for x,Y in zip(X,y):
	    if tuple(x) in dictX:
		if dictX[tuple(x)]!=Y:
			culprits.append(x)
	    else:
		dictX[tuple(x)] = Y

	print(len(X),len(culprits))
def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

pTimes = {}
pPrices = {}
pCosts = {}
pFreq = {}
pTimeranges = {}

def productStats(data):
	global pTimes,pPrices,pCosts,pFreq,pTimeranges
	categorical = {}
	categorical[8] = ["A","B"]
	categorical[10] = ["1","2","3"]
	categorical[11] = ["A","B","C"]
	categorical[12] = ["N","L"]
	categorical[13] = ["ST","DM"]
	categorical[19] = ["IN_HOUSE","NOT_IN_HOUSE"]
	categorical[21] = ["Y","N"]
	categorical[25] = ["B","EA","LB"]
	categorical[26] = ["A","B"]
	categorical[27] = ["1","2","3","4","5"]

	datefields = [2,14]

	#dropfields = [0,1,2,5,6,7,8,9,10,11,12,13,14,21,25,26,27]
	dropfields = [0,1,2,5,6,7,8,9,10,11,12,13,14,21,25,26,27]
	customerfields = [1,7,8,9,10,11,12,13,14]
	productfields = [5,6,19,21,25,26]
	constantfields = [12,10,11,13]
	#dropfields = []
	#dropfields = [2,14]
	productA = defaultdict(list)
	productB = defaultdict(list)
	for d in data:
		row = []
		for i in range(len(d)):
			if i in datefields:
				date = d[i].split(" ")[0]				
				year = int(date.split("-")[0])
				month = int(date.split("-")[1])
				day = int(date.split("-")[2])

				seconds = (datetime(year,month,day) - datetime(1970,1,1)).total_seconds()

				row.append(seconds)
			elif i not in categorical:
				if i!=28:
					row.append(float(d[i]))	
				else:
					if d[i]=="Yes":
						row.append(1)
					if d[i]=="No":
						row.append(-1)
					if d[i]=="Maybe":
						row.append(0)
			else:
				bitv = [0]*len(categorical[i])
				index=0
				for j in range(len(categorical[i])):
					if d[i]==categorical[i][j]:
						row.append(index)
						#bitv[j]=1
						break
					index+=1
				#row = row + bitv
		if d[8]=="A":
			productA[int(d[0])].append(row)
		else:
			productB[int(d[0])].append(row)

	featureA = []
	featureB = []
	for pid,feats in productA.items():
		f = np.array(feats)
		numtransactions = f.shape[0]
		times = f[:,2]
		times.sort()
		period = times[-1]-times[0]
		freq = 0.0
		for t1,t2 in zip(times,times[1:]):
		    freq += abs(t2-t1)
		freq = freq/(len(times)-1.0)

		prices = f[:,3]
		prices.sort()
		pricerange = prices[-1]-prices[0]
		avgprice = sum(prices)/len(prices)

		sales = f[:,4]
		totalsales = sum(sales)

		class1 = f[:,15][0]
		class2 = f[:,16][0]
		class3 = f[:,17][0]
		class4 = f[:,18][0]
		
		brand = f[:,19][0]
		pattr = f[:,20][0]
		totalweight = sum(f[:,22])
		totalboxes = sum(f[:,23])

		costs = f[:,24]
		costs.sort()
		costrange = costs[-1]-costs[0]
		avgcost = sum(costs)/len(costs)

		if f.shape[1]==29:
			labels = f[:,28]
			featureA.append([numtransactions,period,freq,pricerange,avgprice,totalsales,class1,class2,class3,class4,brand,pattr,totalweight,totalboxes,costrange,avgcost,labels[0]])
		else:
			featureA.append([numtransactions,period,freq,pricerange,avgprice,totalsales,class1,class2,class3,class4,brand,pattr,totalweight,totalboxes,costrange,avgcost])

	for pid,feats in productB.items():
		f = np.array(feats)
		numtransactions = f.shape[0]
		times = f[:,2]
		times.sort()
		period = times[-1]-times[0]
		freq = 0.0
		for t1,t2 in zip(times,times[1:]):
		    freq += abs(t2-t1)
		freq = freq/(len(times)-1.0)

		prices = f[:,3]
		prices.sort()
		pricerange = prices[-1]-prices[0]
		avgprice = sum(prices)/len(prices)

		sales = f[:,4]
		totalsales = sum(sales)

		class1 = f[:,15][0]
		class2 = f[:,16][0]
		class3 = f[:,17][0]
		class4 = f[:,18][0]
		
		brand = f[:,19][0]
		pattr = f[:,20][0]
		totalweight = sum(f[:,22])
		totalboxes = sum(f[:,23])

		costs = f[:,24]
		costs.sort()
		costrange = costs[-1]-costs[0]
		avgcost = sum(costs)/len(costs)

		if f.shape[1]==29:
			labels = f[:,28]
			featureB.append([numtransactions,period,freq,pricerange,avgprice,totalsales,class1,class2,class3,class4,brand,pattr,totalweight,totalboxes,costrange,avgcost,labels[0]])
		else:
			featureB.append([numtransactions,period,freq,pricerange,avgprice,totalsales,class1,class2,class3,class4,brand,pattr,totalweight,totalboxes,costrange,avgcost])

	return np.array(featureA),np.array(featureB)

def convertToFeatures(data):
	global pTimes,pPrices,pCosts,pFreq,pTimeranges
	categorical = {}
	categorical[8] = ["A","B"]
	categorical[10] = ["1","2","3"]
	categorical[11] = ["A","B","C"]
	categorical[12] = ["N","L"]
	categorical[13] = ["ST","DM"]
	categorical[19] = ["IN_HOUSE","NOT_IN_HOUSE"]
	categorical[21] = ["Y","N"]
	categorical[25] = ["B","EA","LB"]
	categorical[26] = ["A","B"]
	categorical[27] = ["1","2","3","4","5"]

	datefields = [2,14]

	#dropfields = [0,1,2,5,6,7,8,9,10,11,12,13,14,21,25,26,27]
	#dropfields = [0,1,2,5,6,7,8,9,10,11,12,13,14,21,25,26,27]
	dropfields = [0,1,2,14]
	keepfields = [16,8,22,17,18,15,20,24,3,4]
	customerfields = [1,7,8,9,10,11,12,13,14]
	productfields = [5,6,19,21,25,26]
	constantfields = [12,10,11,13]
	#dropfields = []
	#dropfields = [2,14]
	featureA = []
	featureB = []
	for d in data:
		row = []
		for i in range(len(d)):
			if i not in keepfields:
				continue
			if i in datefields:
				date = d[i].split(" ")[0]				
				year = int(date.split("-")[0])
				month = int(date.split("-")[1])
				day = int(date.split("-")[2])

				seconds = (datetime(year,month,day) - datetime(1970,1,1)).total_seconds()

				#row.append(seconds)
				if int(d[0]) not in pTimes:
					pTimes[int(d[0])] = [seconds]
				else:
					pTimes[int(d[0])].append(seconds)
			elif i not in categorical:
				if i!=0:
					row.append(float(d[i]))	
				else:
					row.append(int(d[i]))
				if i==3:
					if int(d[0]) not in pPrices:
						pPrices[int(d[0])] = [float(d[i])]	
					else:
						pPrices[int(d[0])].append(float(d[i]))
				if i==24:
					if int(d[0]) not in pCosts:
						pCosts[int(d[0])] = [float(d[i])]	
					else:
						pCosts[int(d[0])].append(float(d[i]))
			else:
				bitv = [0]*len(categorical[i])
				index=0
				for j in range(len(categorical[i])):
					if d[i]==categorical[i][j]:
						row.append(index)
						#bitv[j]=1
						break
					index+=1
				#row = row + bitv

		if d[8]=="A":
			featureA.append(row) 
		else:
			featureB.append(row) 

	return featureA,featureB

def getLabels(data):
	labelA = []
	labelB = []
	for d in data:
		label = -2
		if d[-1]=="Yes":
			label = 1	
		if d[-1]=="No":
			label = -1
		if d[-1]=="Maybe":
			label = 0

		if d[8]=="A":
			labelA.append(label)
		else:
			labelB.append(label)

	return labelA,labelB

def getID(data):
	idsa = []
	idsb = []
	for d in data:
		if d[8]=="A":
			idsa.append(int(d[0]))
		else:
			idsb.append(int(d[0]))

	return idsa,idsb

def rulesA(x):
	price = 0
	sales = 1
	cs1 = 2
	pc1 = 3
	pc2 = 4
	pc3 = 5
	pc4 = 6
	pattr = 7
	swt = 8
	pcost = 9

	if (x[pc2]==24 or x[pc2]==25) and (x[pc4]>=2387 and x[pc4]<=2514) and (x[pc3]>=195 and x[pc3]<=209):
		return 0
	if (x[pc2]==24 or x[pc2]==25) and (x[pc4]>=2387 and x[pc4]<=2514) and (x[pc3]>332):
		return 1
	if (x[pc2]==24 or x[pc2]==25) and (x[pc4]>6230):
		return 1

	return -2

def rulesB(x):
	price = 0
	sales = 1
	cs1 = 2
	pc1 = 3
	pc2 = 4
	pc3 = 5
	pc4 = 6
	pattr = 7
	swt = 8
	pcost = 9

	if (x[pc2]==24 or x[pc2]==25) and (x[swt]>66.01):
		return 0
	if (x[pc2]==24 or x[pc2]==25) and (x[pc4]>=2387 and x[pc4]<=2514) and (x[pc3]>332) and (x[swt]>25):
		return 1

	return -2

def rules(x):
	price = 0
	sales = 1
	cs1 = 2
	pc1 = 3
	pc2 = 4
	pc3 = 5
	pc4 = 6
	pattr = 7
	swt = 8
	pcost = 9

	if (x[pc2]==24 or x[pc2]==25) and (x[pc4]>=2387 and x[pc4]<=2514) and (x[pc3]>=210 and x[pc3]<=249):
		return 0
	if (x[pc2]==24 or x[pc2]==25) and (x[pc4]>=2514 and x[pc4]<=6230) and (x[pc3]>209):
		return 1
	if (x[pc2]==24 or x[pc2]==25) and (x[pc4]>=1725 and x[pc4]<=2387):
		return 0
	if (x[pc2]<=22) and (x[pc1]>=2 and x[pc1]<=10) and (x[pattr]>=14 and x[pattr]<=120) and (x[pc3]>332):
		return 1
	if (x[pc2]<=22) and (x[pc1]>=2 and x[pc1]<=10) and (x[pattr]>270):
		return 0
	if (x[pc2]>=22 and x[pc2]<=24):
		return -1
	if (x[pc2]<=22) and (x[pc1]<=2) and (x[swt]>=25 and x[swt]<=39.75) and (x[pcost]<=20.16):
		return -1
	if (x[pc2]<=22) and (x[pc1]<=2) and (x[swt]>39.75) and (x[price]>35.85):
		return -1
	if (x[pc2]<=22) and (x[pc1]<=2) and (x[swt]>=10 and x[swt]<=25) and (x[pattr]>=14 and x[pattr]<=270):
		return -1
	if (x[pc2]<=22) and (x[pc1]<=2) and (x[swt]<=10) and (x[sales]>=18.77 and x[pattr]<=59.47):
		return -1
	return -2


class ElectronicPartsClassification:
	def classifyParts(self,trndata,testdata):
		global trainx,trainy

		delimtrn = [x.split(',')[:-1] for x in trndata]
		delimtest = [x.split(',') for x in testdata]
		idsA,idsB = getID(delimtest)

		scalerA = StandardScaler()
		scalerB = StandardScaler()

		trainxA,trainxB = convertToFeatures(delimtrn)
		trainxA = scalerA.fit_transform(trainxA)
		trainxB = scalerB.fit_transform(trainxB)
		print(len(trainxA[0]))
		testA,testB = convertToFeatures(delimtest)
		testA = scalerA.transform(testA)
		testB = scalerB.transform(testB)

		delimtrn = [x.split(',') for x in trndata]
		trainyA,trainyB = getLabels(delimtrn)
		clf = ExtraTrees(n_estimators=15,subsample=8,min_samples_split=16)
		#clf = BoostedTree([-1,0,1],n_estimators=10,max_depth=-1,max_features=3,min_samples_split=4,learning_rate=0.1,subsample=8)
		#clf = BoostedTreeSimple(n_estimators=2,max_depth=-1,max_features=3,min_samples_split=4,subsample=6)
		#clf = BinaryDecisionTree()
		trainx = trainxA
		trainy = trainyA
		clf.fit(trainx,trainy)
		predA = clf.predict(testA)		
		clf = ExtraTrees(n_estimators=15,subsample=8,min_samples_split=16)
		#clf = ExtraTreesClassifier(n_estimators=20,max_depth=20,min_samples_split=10,bootstrap=True)
		#clf = BoostedTreeSimple(n_estimators=2,max_depth=-1,max_features=3,min_samples_split=4,subsample=6)
		#clf = BoostedTree([-1,0,1],n_estimators=10,max_depth=-1,max_features=3,min_samples_split=4,learning_rate=0.1,subsample=8)
		trainx = trainxB
		trainy = trainyB
		clf.fit(trainx,trainy)
		predB = clf.predict(testB)		

		productidpreds = {}
		for i in range(len(testA)):
			pred = predA[i]

			if idsA[i] not in productidpreds:
				productidpreds[idsA[i]] = {}

			if "A" not in productidpreds[idsA[i]]:
				productidpreds[idsA[i]]["A"] = [pred]
			else:
				productidpreds[idsA[i]]["A"].append(pred)

		for i in range(len(testB)):
			pred = predB[i]

			if idsB[i] not in productidpreds:
				productidpreds[idsB[i]] = {}

			if "B" not in productidpreds[idsB[i]]:
				productidpreds[idsB[i]]["B"] = [pred]
			else:
				productidpreds[idsB[i]]["B"].append(pred)

		predictions = []
		for k,v in productidpreds.iteritems():
			strpred = str(k)+","
			if "A" in v:
				cnt = Counter(v["A"])
				pred = cnt.most_common(n=1)[0][0]
				if pred==-1:
					strpred+="No,"
				elif pred==1:
					strpred+="Yes,"
				else:
					strpred+="Maybe,"
			else:
					strpred+="NA,"
			if "B" in v:
				cnt = Counter(v["B"])
				pred = cnt.most_common(n=1)[0][0]
				if pred==-1:
					strpred+="No,"
				elif pred==1:
					strpred+="Yes,"
				else:
					strpred+="Maybe,"
			else:
					strpred+="NA"

			predictions.append(strpred)

		print(predictions)
		return predictions

def mymain():
	trainingData = []
	testingData = []
	testingTruth = []
	testType = 0
	index=1 
  # read data
	uids = []
	truepredictions = {}
	with open('example_data.csv', 'r') as f:
		header = True
		for line in f:
			# skip header
			if header:
				header = False
				continue
			# remove carriage return
			line = line.rstrip('\n').rstrip('\r')
			pid = line.split(",")[0]
			if pid not in uids:
				uids.append(pid)
	
	trainids = random.sample(uids,2*len(uids)/3)
	'''
	trainids = []
	for i in range(len(uids)):
	    if i%3!=0:
		trainids.append(uids[i])
	'''
	print(len(trainids),len(uids))
	print(trainids)
	with open('example_data.csv', 'r') as f:
		header = True
		for line in f:
			# skip header
			if header:
				header = False
				continue
			# remove carriage return
			line = line.rstrip('\n').rstrip('\r')
			# affect data to training or testing randomly
		
			pid = line.split(",")[0]	
			#if numpy.random.randint(0, 3) == 0 :
			if pid not in trainids:
				# remove the last column
				pos = line.rfind(',')
				testingData.append(line[:pos])
				label = line.split(",")[-1]
				segment = line.split(",")[8]

				if pid not in truepredictions:
					truepredictions[pid] = {}
				truepredictions[pid][segment] = label
			else :
				trainingData.append(line)
			index+=1

	# DemographicMembership instance and predict function call      
	epc = ElectronicPartsClassification()
	testingPred = epc.classifyParts(trainingData, testingData)

	validelements = 0
	totalerror = 0
	for p in testingPred:
		pid = p.split(",")[0]
		alabel = p.split(",")[1]
		blabel = p.split(",")[2]

		if "A" in truepredictions[pid]:
			validelements+=1
			if truepredictions[pid]["A"]=="Maybe" and alabel=="No":
				totalerror+=0.2
			if truepredictions[pid]["A"]=="Yes" and alabel=="No":
				totalerror+=0.7
			if truepredictions[pid]["A"]=="No" and alabel=="Maybe":
				totalerror+=0.5
			if truepredictions[pid]["A"]=="Yes" and alabel=="Maybe":
				totalerror+=0.01
			if truepredictions[pid]["A"]=="No" and alabel=="Yes":
				totalerror+=1.0
			if truepredictions[pid]["A"]=="Maybe" and alabel=="Yes":
				totalerror+=0.01
		if "B" in truepredictions[pid]:
			validelements+=1
			if truepredictions[pid]["B"]=="Maybe" and blabel=="No":
				totalerror+=0.2
			if truepredictions[pid]["B"]=="Yes" and blabel=="No":
				totalerror+=0.7
			if truepredictions[pid]["B"]=="No" and blabel=="Maybe":
				totalerror+=0.5
			if truepredictions[pid]["B"]=="Yes" and blabel=="Maybe":
				totalerror+=0.01
			if truepredictions[pid]["B"]=="No" and blabel=="Yes":
				totalerror+=1.0
			if truepredictions[pid]["B"]=="Maybe" and blabel=="Yes":
				totalerror+=0.01
		if alabel=="NA" and "A" in truepredictions[pid]:
			print("predicted NA for valid A")
		if blabel=="NA" and "B" in truepredictions[pid]:
			print("predicted NA for valid B")

	print(totalerror,validelements,totalerror/validelements)
	score = float(1000000.0 * (1.0 - (totalerror/validelements)))
	print("Score:",score)
	print(len(testingPred),len(truepredictions))
	return score
if __name__ == '__main__':
	avgscore = 0.0	
	for i in range(30):
		print(i)
		avgscore+=mymain()
	avgscore/=30
	print("Avg:",avgscore)
