"""ONNX Runtime profiling helpers for Qualcomm QNN and CPU execution."""

import os
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence


QNN_PROVIDER = "QNNExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"


TARGET_DEVICE = "Snapdragon X Elite CRD"


class QualcommProfiler:
    """Measure ONNX Runtime inference on QNN when it is available."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        qnn_backend_path: Optional[str] = None,
        iterations: int = 10,
        session_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        if iterations < 1:
            raise ValueError("iterations must be at least 1")
        self.model_path = model_path
        self.qnn_backend_path = qnn_backend_path or os.getenv("QNN_BACKEND_PATH")
        self.iterations = iterations
        self._session_factory = session_factory
        self._latencies_ms = []
        self._throughputs = []
        self._runtime = None
        self.available_providers = self._detect_available_providers()
        self.active_provider = self._select_provider()

    def _detect_available_providers(self) -> Sequence[str]:
        """Return providers exposed by the installed ONNX Runtime build."""
        try:
            runtime = self._get_runtime()
            return tuple(runtime.get_available_providers())
        except (ImportError, AttributeError):
            return ()

    def _get_runtime(self) -> Any:
        if self._runtime is None:
            if self._session_factory is not None:
                return None
            try:
                import onnxruntime as runtime
            except ImportError:
                raise ImportError(
                    "ONNX Runtime is required for inference; install onnxruntime "
                    "or provide session_factory for testing."
                ) from None
            self._runtime = runtime
        return self._runtime

    def _select_provider(self) -> str:
        if QNN_PROVIDER in self.available_providers:
            return QNN_PROVIDER
        return CPU_PROVIDER

    @property
    def backend_path(self) -> Optional[str]:
        """Configured QNN backend library path, if one was supplied."""
        return self.qnn_backend_path

    def _providers_for(self, provider: str) -> Sequence[Any]:
        if provider == QNN_PROVIDER and self.qnn_backend_path:
            return [(QNN_PROVIDER, {"backend_path": self.qnn_backend_path})]
        return [provider]

    def _create_session(self, model_path: str, provider: str) -> Any:
        if provider == QNN_PROVIDER and QNN_PROVIDER not in self.available_providers:
            raise RuntimeError("QNNExecutionProvider is not available")
        factory = self._session_factory
        if factory is None:
            factory = self._get_runtime().InferenceSession
        try:
            return factory(model_path, providers=self._providers_for(provider))
        except Exception as exc:
            if provider == QNN_PROVIDER:
                raise RuntimeError(f"Could not create a QNN session: {exc}") from exc
            raise

    @staticmethod
    def _input_feed(session: Any, sample_input: Any) -> Mapping[str, Any]:
        if isinstance(sample_input, Mapping):
            return sample_input
        inputs = session.get_inputs()
        if not inputs:
            raise ValueError("The ONNX model has no inputs")
        return {inputs[0].name: sample_input}

    @staticmethod
    def _default_work_units(sample_input: Any, outputs: Sequence[Any]) -> int:
        if outputs and hasattr(outputs[0], "shape") and outputs[0].shape:
            return max(1, int(outputs[0].shape[0]))
        if hasattr(sample_input, "shape") and sample_input.shape:
            return max(1, int(sample_input.shape[0]))
        return 1

    def measure_inference(
        self,
        model_path: Optional[str],
        sample_input: Any,
        provider: Optional[str] = None,
        work_units: Optional[int] = None,
    ) -> Dict[str, float]:
        """Run inference repeatedly and record latency and throughput."""
        path = model_path or self.model_path
        if not path:
            raise ValueError("model_path must be provided")
        selected_provider = provider or self.active_provider
        session = self._create_session(path, selected_provider)
        feed = self._input_feed(session, sample_input)
        latencies = []
        outputs = ()
        for _ in range(self.iterations):
            started = time.perf_counter()
            outputs = session.run(None, feed)
            latencies.append((time.perf_counter() - started) * 1000.0)
        average_latency = sum(latencies) / len(latencies)
        units = work_units or self._default_work_units(sample_input, outputs)
        throughput = units / (average_latency / 1000.0)
        self._latencies_ms.append(average_latency)
        self._throughputs.append(throughput)
        return {
            "provider": selected_provider,
            "latency_ms": average_latency,
            "throughput_per_second": throughput,
            "tokens_per_second": throughput,
            "embeddings_per_second": throughput,
            "work_units": units,
        }

    def record_latency(self, latency_ms: float, work_units: int = 1) -> Dict[str, float]:
        """Record latency from an inference session owned by another component."""
        if latency_ms < 0 or work_units < 1:
            raise ValueError("latency_ms must be non-negative and work_units must be positive")
        throughput = work_units / (latency_ms / 1000.0) if latency_ms else float("inf")
        self._latencies_ms.append(latency_ms)
        self._throughputs.append(throughput)
        return {"latency_ms": latency_ms, "throughput_per_second": throughput}

    def benchmark_provider_comparison(
        self, model_path: str, sample_input: Any
    ) -> Dict[str, Any]:
        """Compare QNN (Hexagon HTP) and CPU latency for one model/input."""
        result: Dict[str, Any] = {"qnn": None, "cpu": None}
        try:
            result["qnn"] = self.measure_inference(model_path, sample_input, QNN_PROVIDER)
        except (RuntimeError, ImportError) as exc:
            result["qnn_error"] = str(exc)
        result["cpu"] = self.measure_inference(model_path, sample_input, CPU_PROVIDER)
        if result["qnn"]:
            qnn_latency = result["qnn"]["latency_ms"]
            cpu_latency = result["cpu"]["latency_ms"]
            result["speedup_factor"] = cpu_latency / qnn_latency
            result["latency_delta_ms"] = cpu_latency - qnn_latency
        else:
            result["speedup_factor"] = None
            result["latency_delta_ms"] = None
        return result

    def format_telemetry_report(self, as_dict: bool = False) -> Any:
        """Return telemetry formatted for a CLI string or UI-friendly dict."""
        average_latency = (
            sum(self._latencies_ms) / len(self._latencies_ms)
            if self._latencies_ms
            else None
        )
        report = {
            "active_provider": self.active_provider,
            "backend_path": self.backend_path,
            "average_latency_ms": average_latency,
            "throughput_per_second": (
                sum(self._throughputs) / len(self._throughputs)
                if self._throughputs
                else None
            ),
            "tokens_per_second": (
                sum(self._throughputs) / len(self._throughputs)
                if self._throughputs
                else None
            ),
            "embeddings_per_second": (
                sum(self._throughputs) / len(self._throughputs)
                if self._throughputs
                else None
            ),
            "acceleration_status": (
                "QNN / Hexagon HTP active"
                if self.active_provider == QNN_PROVIDER
                else "CPU fallback"
            ),
        }
        if as_dict:
            return report
        return "\n".join(
            f"{key.replace('_', ' ').title()}: {value}"
            for key, value in report.items()
        )


def _print_mock_diagnostics():
    print("Qualcomm AI Hub is not configured; showing simulated Snapdragon validation.")
    print(f"Target device: {TARGET_DEVICE}")
    print("Runtime: Qualcomm QNN / Hexagon NPU")
    print("Model                 Latency (ms)   Status")
    print("MobileCLIP             4.2             simulated")
    print("Whisper-Base          12.8             simulated")
    print("Set QAI_HUB_API_TOKEN and rerun to submit a real AI Hub job.")


def profile_snapdragon():
    """Submit a real AI Hub profile when configured, otherwise remain runnable."""
    token = os.getenv("QAI_HUB_API_TOKEN") or os.getenv("QAI_TOKEN")
    try:
        import qai_hub
    except ImportError:
        _print_mock_diagnostics()
        return None

    if not token:
        _print_mock_diagnostics()
        return None

    try:
        if hasattr(qai_hub, "set_token"):
            qai_hub.set_token(token)
        device = qai_hub.Device(TARGET_DEVICE)
        from qai_hub_models.models.mobilenet_v2.model import MobileNetV2

        model = MobileNetV2.from_pretrained()
        job = qai_hub.submit_profile_job(model=model, device=device)
        print(f"Submitted Qualcomm AI Hub profile job: {job}")
        return job
    except Exception as exc:
        print(f"Qualcomm AI Hub profiling could not be submitted: {exc}")
        print("Verify the token, model package, and target device availability.")
        return None