import numpy as np             # For numerical arrays and math


# --- Flexible field getter for MATLAB structs ---
def get_field(obj, name):
    """
    Retrieve a field `name` from a MATLAB struct or nested object.

    Works for:
      - scipy.io.loadmat object trees
      - MATLAB structs stored as object arrays or numpy.void types
      - Attributes (.RawData) or dict-like access
    """

    if hasattr(obj, name):                                 #if stored as attribute
        #print("obj is attribute")
        return getattr(obj, name)

    if isinstance(obj, dict) and name in obj:              #if stored as dict
        print("obj is dict")
        return obj[name]

    if isinstance(obj, np.ndarray) and obj.dtype.names:    #if stored as structured NumPy arrays
        val = obj[name]
        if isinstance(val, np.ndarray) and val.size == 1:  #if stored inside array
            print("obj is NumpyArray")
            return val.item()
        return val
    if isinstance(obj, np.ndarray):                        #if stored in single python object wrapped in another array
        print("obj is wrapped in NumpyArray")
        for el in obj.flatten():
            try:
                v = get_field(el, name)
                if v is not None:
                    return v
            except Exception:
                continue
    try:
        it = obj.item()  #extract it
        return get_field(it, name)  # recursion
    except Exception:
        pass
    raise KeyError(f"Field '{name}' not found in object of type {type(obj)}")


# --- Identify all valid trials in a MATLAB file ---
def find_trials(mat):
    # Searches for keys like 'data' or selects the largest
    # non-metadata ndarray. Then checks for RawData/FileHeader fields.

    if "trials" in mat:
        cand = mat["trials"]   #candidate key
    elif "data" in mat:
        cand = mat["data"]
    else:
        cand = None
        for k, v in mat.items():
            if k.startswith("__"):
                continue
            if isinstance(v, np.ndarray):
                if cand is None or v.size > getattr(cand, "size", 0):
                    cand = v
        if cand is None:
            raise ValueError("No candidate trial array found in .mat")

    arr = np.array(cand, copy=False).flatten()
    trials = []

    # loop through each element of the array.
    # skip any element that’s just a scalar, number, or string
    for el in arr:
        if isinstance(el, (int, float, np.integer, np.floating, str)):
            continue
        for key in ("RawData", "FileHeader"):
            try:
                _ = get_field(el, key)
                trials.append(el)
                break
            except Exception:
                continue
    return trials

# Orient array correctly (samples (t) × channels) (necessary for mne)
def orient_eeg(eeg_arr, n_ch):
    if n_ch is not None:
        if eeg_arr.shape[0] == n_ch:  # nr of rows equals nr of channels
            eeg = eeg_arr.T  # transpose
        elif eeg_arr.shape[1] == n_ch:  # nr of columns equals nr of channels
            eeg = eeg_arr
        else:
            # if no not right, the biggest dim is probably time
            eeg = eeg_arr.T if eeg_arr.shape[0] > eeg_arr.shape[1] else eeg_arr
    else:
        # n_ch=None
        # <256 is a plausible nr of electrodes and there are more time samples
        eeg = eeg_arr.T if eeg_arr.shape[0] <= 256 and eeg_arr.shape[1] > eeg_arr.shape[0] else eeg_arr
    return eeg

