def apply_car(data):
    return data - np.mean(data, axis=1, keepdims=True