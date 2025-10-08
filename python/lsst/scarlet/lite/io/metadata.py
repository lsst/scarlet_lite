from typing import Any

import numpy as np


def numpy_to_json(arr: np.ndarray) -> dict[str, Any]:
    """
    Encode a numpy array as JSON-serializable dictionary.

    Parameters
    ----------
    arr :
        The numpy array to encode

    Returns
    -------
    result :
        A JSON formatted dictionary containing the dtype, shape,
        and data of the array.
    """
    # Convert to native Python types for JSON serialization
    flattened = arr.flatten()

    # Convert numpy scalars to native Python types
    if np.issubdtype(arr.dtype, np.integer):
        data: list = [int(x) for x in flattened]
    elif np.issubdtype(arr.dtype, np.floating):
        data = [float(x) for x in flattened]
    elif np.issubdtype(arr.dtype, np.complexfloating):
        data = [complex(x) for x in flattened]
    elif np.issubdtype(arr.dtype, np.bool_):
        data = [bool(x) for x in flattened]
    else:
        # For other types (strings, objects, etc.), convert to string
        data = [str(x) for x in flattened]

    return {"dtype": str(arr.dtype), "shape": tuple(arr.shape), "data": data}


def json_to_numpy(encoded_dict: dict[str, Any]) -> np.ndarray:
    """
    Decode a JSON dictionary back to a numpy array.

    Parameters
    ----------
    encoded_dict :
        Dictionary with 'dtype', 'shape', and 'data' keys.

    Returns
    -------
    result :
        The reconstructed numpy array.
    """
    if "dtype" not in encoded_dict or "shape" not in encoded_dict or "data" not in encoded_dict:
        raise ValueError("Encoded dictionary must contain 'dtype', 'shape', and 'data' keys.")
    return np.array(encoded_dict["data"], dtype=encoded_dict["dtype"]).reshape(encoded_dict["shape"])


def encode_metadata(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    """Pack metadata into a JSON compatible format.

    Parameters
    ----------
    metadata :
        The metadata to be packed.

    Returns
    -------
    result :
        The packed metadata.
    """
    if metadata is None:
        return None
    encoded = {}
    array_keys = []
    for key, value in metadata.items():
        if isinstance(value, np.ndarray):
            _encoded = numpy_to_json(value)
            encoded[key] = _encoded["data"]
            encoded[f"{key}_shape"] = _encoded["shape"]
            encoded[f"{key}_dtype"] = _encoded["dtype"]
            array_keys.append(key)
        else:
            encoded[key] = value
    if len(array_keys) > 0:
        encoded["array_keys"] = array_keys
    return encoded


def decode_metadata(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    """Unpack metadata from a JSON compatible format.

    Parameters
    ----------
    metadata :
        The metadata to be unpacked.

    Returns
    -------
    result :
        The unpacked metadata.
    """
    if metadata is None:
        return None
    if "array_keys" in metadata:
        for key in metadata["array_keys"]:
            # Default dtype is float32 to support legacy models
            dtype = metadata.pop(f"{key}_dtype", "float32")
            shape = metadata.pop(f"{key}_shape", None)
            if shape is None and f"{key}Shape" in metadata:
                # Support legacy models that use `keyShape`
                shape = metadata[f"{key}Shape"]
            decoded = json_to_numpy({"dtype": dtype, "shape": shape, "data": metadata[key]})
            metadata[key] = decoded
        # Remove the array keys after decoding
        del metadata["array_keys"]
    return metadata


def extract_from_metadata(
    data: Any,
    metadata: dict[str, Any] | None,
    key: str,
) -> Any:
    """Extract relevant information from the metadata.

    Parameters
    ----------
    data :
        The data to extract information from.
    metadata :
        The metadata to extract information from.
    key :
        The key to extract from the metadata.

    Returns
    -------
    result :
        A tuple containing the extracted data and metadata.
    """
    if data is not None:
        return data
    if metadata is None:
        raise ValueError("Both data and metadata cannot be None")
    if key not in metadata:
        raise ValueError(f"'{key}' not found in metadata")
    return metadata[key]
