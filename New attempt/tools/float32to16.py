
def Float32_To_Float16(float32):
    """
    Convert a float32 number to float16.
    """
    # Convert float32 to float16 using numpy
    import numpy as np
    return np.float16(float32).item()