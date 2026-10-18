try:
    import onnxruntime as ort
except ImportError:
    ort = None
import torch

from . import config
from .model import ScamMLP


def export(model, n_features, path=config.ONNX_PATH):
    """Export ``model`` to ONNX: float32 input (batch, n_features) -> raw logit (batch, 1).

    The batch dimension is dynamic; apply sigmoid to the output for a probability.
    """
    model.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, n_features, dtype=torch.float32)
    kwargs = dict(
        input_names=["features"],
        output_names=["logit"],
        dynamic_axes={"features": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=17,
    )
    try:
        torch.onnx.export(model, (dummy,), str(path), dynamo=False, **kwargs)
    except TypeError:          # torch < 2.5 has no ``dynamo`` argument
        torch.onnx.export(model, (dummy,), str(path), **kwargs)
    return path


def verify(model, sample, path=config.ONNX_PATH, atol=1e-5):
    """Run one sample through PyTorch and onnxruntime and compare the logits.

    Returns:
        (torch_logit, onnx_logit, max_abs_diff).
    """
    model.eval()
    sample = np.asarray(sample, dtype=np.float32).reshape(1, -1)
    with torch.no_grad():
        torch_out = model(torch.from_numpy(sample)).numpy()
    if ort is None:
        print("onnxruntime not available, skipping ONNX validation")
        return
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(["logit"], {"features": sample})[0]
    diff = float(np.abs(torch_out - onnx_out).max())
    if diff > atol:
        raise RuntimeError(f"ONNX output differs from PyTorch by {diff:.2e} (> {atol})")
    return float(torch_out[0, 0]), float(onnx_out[0, 0]), diff


def run(model=None, n_features=None, sample=None):
    """Export + verify. Loads artifacts/model_federated.pt if no model is passed."""
    if n_features is None:
        n_features = len(config.FEATURE_COLUMNS)
    if model is None:
        model = ScamMLP(n_features)
        model.load_state_dict(torch.load(config.MODEL_FED_PATH))
    if sample is None:
        sample = np.zeros(n_features, dtype=np.float32)
    path = export(model, n_features)
    if ort is None:
        print("onnxruntime not available, skipping ONNX validation")
        return {"exported": str(path)}
    if ort is None:
        print("onnxruntime not available — skipping ONNX validation")
        return {"exported": str(path)}
    t, o, diff = verify(model, sample, path)
    prob = 1 / (1 + np.exp(-o))
    print(f"  ONNX saved to {path}")
    print(f"  Verify: torch logit={t:.6f}  onnx logit={o:.6f}  |diff|={diff:.2e}  p(fraud)={prob:.4f}")
    return {"path": str(path), "torch_logit": t, "onnx_logit": o, "max_abs_diff": diff}


if __name__ == "__main__":
    run()
