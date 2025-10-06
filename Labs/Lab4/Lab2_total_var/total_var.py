import numpy as np

def processCodeDoc(x : np.ndarray) -> np.ndarray :
    """Return the discrete difference of a 2D matrix along the columns and append a vector of zeros at the end"""
    res=np.diff(x,axis=1)
    res = np.concatenate((res, np.zeros((x.shape[0], 1))), axis=1)
    return res

def gradient2DDoc(x : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the discrete gradient of a 2D matrix as a tuple of two 2D arrays(horizontal and vertical gradients).

    Parameters
    -----------
    x : ndarray
        Input 2D array.

    Returns
    -----------
    (ndarray, ndarray)
        A tuple containing two 2D arrays: the first array is the horizontal gradient, and the second array is the vertical gradient.

    Raises
    -----------
    AssertionError
        If the input array is not 2D.

    Notes
    -----------
    Uses the processCodeDoc function to compute the discrete differences.
    """
    assert x.ndim == 2, "Input must be a 2D array"
    return (processCodeDoc(x), processCodeDoc(x.T).T)

def tv(x : np.ndarray) -> float:
    """Compute the total variation of a 2D array.

    Parameters
    -----------
    x : ndarray
        Input 2D array.

    Returns
    -----------
    float
        The total variation of the input array.

    Raises
    -----------
    AssertionError
        If the input array is not 2D.

    Notes
    -----------
    The total variation is computed as the sum of the square root values of thesum of squares of the horizontal and vertical gradients.
    """
    assert x.ndim == 2, "Input must be a 2D array"
    XDh, DvX = gradient2DDoc(x)
    return np.sum(np.sqrt(XDh**2 + DvX**2))
