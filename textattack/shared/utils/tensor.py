import numpy as np
import torch
from IPython import embed


def batch_model_predict(model_predict, inputs, batch_size=32):
    """Runs prediction on iterable ``inputs`` using batch size ``batch_size``.

    Aggregates all predictions into an ``np.ndarray``.
    """
    results = []
    i = 0
    while i < len(inputs):
        input_ids = inputs[0][i : i + batch_size]
        attention_mask = inputs[1][i : i + batch_size]

        batch_preds = model_predict.predict(input_ids, attention_mask)

        # print(str(type(model_predict)))

        # Some seq-to-seq models will return a single string as a prediction
        # for a single-string list. Wrap these in a list.
        if isinstance(batch_preds, str):
            batch_preds = [batch_preds]

        # Get PyTorch tensors off of other devices.
        if isinstance(batch_preds, torch.Tensor):
            batch_preds = batch_preds.cpu()

        # Cast all predictions iterables to ``np.ndarray`` types.
        if not isinstance(batch_preds, np.ndarray):
            batch_preds = np.array(batch_preds)
        results.append(batch_preds)
        i += batch_size

    return np.concatenate(results, axis=0)
