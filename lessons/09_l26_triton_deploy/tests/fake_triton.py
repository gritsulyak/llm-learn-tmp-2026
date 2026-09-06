"""Minimal stand-in for triton_python_backend_utils to unit-test BLS model.py.

This mimics the subset of the Triton Python backend API used by the
ensemble model's model.py, so it can be exercised without a live server.
"""

import numpy as np


class Tensor:
    def __init__(self, name, numpy_array):
        self.name = name
        self._data = np.asarray(numpy_array)

    def as_numpy(self):
        return self._data

    def __repr__(self):
        return f"Tensor({self.name!r}, shape={self._data.shape})"


class FakeResponse:
    def __init__(self, tensors, error=None):
        self._tensors = {t.name: t for t in tensors}
        self._error = error

    def has_error(self):
        return self._error is not None

    def error(self):
        return self._error


class _ModelServer:
    """Simulates other models: handlers produce output dicts from input dicts."""

    def __init__(self):
        self.handlers = {}
        self.call_log = []

    def register(self, model_name, handler):
        self.handlers[model_name] = handler

    def unregister(self, model_name):
        self.handlers.pop(model_name, None)

    def run(self, model_name, input_dict):
        self.call_log.append(model_name)
        if model_name not in self.handlers:
            raise RuntimeError(f"No handler registered for model {model_name!r}")
        return dict(self.handlers[model_name](input_dict))


_simulator = None


def set_simulator(simulator):
    global _simulator
    _simulator = simulator


class InferenceRequest:
    def __init__(
        self,
        model_name=None,
        request_id=None,
        correlation_id=0,
        model_version=None,
        inputs=None,
        requested_output_names=None,
        flags=0,
    ):
        self.model_name = model_name
        items = inputs or []
        if isinstance(items, dict):
            items = items.values()
        self._inputs = {t.name: np.asarray(t.as_numpy()) for t in items}

    def exec(self, timeout=None):
        if _simulator is None:
            raise RuntimeError("No simulator configured")
        try:
            raw_outputs = _simulator.run(self.model_name, self._inputs)
        except RuntimeError as exc:
            return FakeResponse(
                [],
                error=f"{self.model_name}: {exc}",
            )
        tensors = [Tensor(n, a) for n, a in raw_outputs.items()]
        return FakeResponse(tensors)


def get_input_tensor_by_name(request, name):
    if name not in request._inputs:
        raise KeyError(f"Input {name!r} not found")
    return Tensor(name, request._inputs[name])


def get_output_tensor_by_name(response, name):
    if name not in response._tensors:
        raise KeyError(f"Output {name!r} not found")
    return response._tensors[name]


def InferenceResponse(output_tensors=None, error=None):
    return FakeResponse(output_tensors or [], error)