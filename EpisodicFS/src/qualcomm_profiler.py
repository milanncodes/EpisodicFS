"""ONNX Runtime profiling helpers for Qualcomm QNN and CPU execution."""

import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

try:
    import numpy as np
except ImportError:
    np = None


QNN_PROVIDER = "QNNExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"


TARGET_DEVICE = "Snapdragon X Elite CRD"


def _warn(message: str) -> None:
    print(f"Warning: {message}")


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
        self._memory_deltas_mb = []
        self._runtime = None
        self.is_npu_active = False
        self.available_providers = self._detect_available_providers()
        self.active_provider = self._select_provider()

    def _detect_available_providers(self) -> Sequence[str]:
        """Return providers exposed by the installed ONNX Runtime build."""
        try:
            runtime = self._get_runtime()
            return tuple(runtime.get_available_providers())
        except ImportError:
            _warn("ONNX Runtime is not installed; using CPU fallback telemetry.")
            return ()
        except Exception as exc:
            _warn(f"Could not query ONNX Runtime providers; using CPU fallback ({exc}).")
            return ()

    def _get_runtime(self) -> Any:
        if self._runtime is None:
            if self._session_factory is not None:
                return None
            try:
                import onnxruntime as runtime
            except ImportError:
                _warn("ONNX Runtime is unavailable; Qualcomm profiling cannot create sessions.")
                raise ImportError(
                    "ONNX Runtime is required for inference; install onnxruntime "
                    "or provide session_factory for testing."
                ) from None
            self._runtime = runtime
        return self._runtime

    def _select_provider(self) -> str:
        backend_path = self.qnn_backend_path
        if backend_path and not Path(backend_path).is_file():
            backend_path = None
        if QNN_PROVIDER in self.available_providers and (backend_path or self._find_qnn_backend()):
            return QNN_PROVIDER
        if QNN_PROVIDER in self.available_providers:
            _warn("QNNExecutionProvider is registered, but QnnHtp.dll was not found; using CPU fallback.")
        return CPU_PROVIDER

    @staticmethod
    def _find_qnn_backend() -> Optional[str]:
        """Find QnnHtp.dll using SDK variables, PATH, and common Windows roots."""
        requested = os.getenv("QNN_BACKEND_PATH")
        if requested and Path(requested).is_file():
            return str(Path(requested))

        candidates = []
        for variable in ("QNN_SDK_ROOT", "ONNXRUNTIME_QNN_PATH"):
            root = os.getenv(variable)
            if root:
                root_path = Path(root)
                candidates.extend((root_path / "QnnHtp.dll", root_path / "lib" / "QnnHtp.dll"))
                if root_path.is_dir():
                    candidates.extend(root_path.glob("**/QnnHtp.dll"))
        for entry in os.getenv("PATH", "").split(os.pathsep):
            if entry:
                candidates.append(Path(entry) / "QnnHtp.dll")

        if sys.platform == "win32" or platform.system() == "Windows":
            candidates.extend(
                Path(root) / "QnnHtp.dll"
                for root in (
                    os.getenv("ProgramFiles", r"C:\\Program Files"),
                    os.getenv("ProgramW6432", r"C:\\Program Files"),
                    os.getenv("LOCALAPPDATA", r"C:\\Users\\Public\\AppData\\Local"),
                )
            )
        for candidate in candidates:
            try:
                if candidate.is_file():
                    return str(candidate.resolve())
            except OSError as exc:
                _warn(f"Could not inspect possible QNN backend '{candidate}': {exc}")
                continue
        return None

    @property
    def backend_path(self) -> Optional[str]:
        """Configured QNN backend library path, if one was supplied."""
        return self.qnn_backend_path

    def _providers_for(self, provider: str) -> Sequence[Any]:
        if provider == QNN_PROVIDER:
            backend_path = self.qnn_backend_path or self._find_qnn_backend()
            if not backend_path:
                raise RuntimeError("QnnHtp.dll was not found")
            self.qnn_backend_path = backend_path
            return [(QNN_PROVIDER, {"backend_path": backend_path})]
        return [provider]

    def _create_session(self, model_path: str, provider: str) -> Any:
        if provider == QNN_PROVIDER and QNN_PROVIDER not in self.available_providers:
            raise RuntimeError("QNNExecutionProvider is not available")
        factory = self._session_factory
        if factory is None:
            factory = self._get_runtime().InferenceSession
        try:
            session = factory(model_path, providers=self._providers_for(provider))
            session_providers = tuple(session.get_providers())
            if provider == QNN_PROVIDER and QNN_PROVIDER not in session_providers:
                raise RuntimeError(
                    "QNNExecutionProvider was requested but is not active in the session"
                )
            if provider == QNN_PROVIDER:
                self.is_npu_active = True
                self.active_provider = QNN_PROVIDER
            else:
                self.is_npu_active = False
                self.active_provider = CPU_PROVIDER
            return session
        except Exception as exc:
            if provider == QNN_PROVIDER:
                self.is_npu_active = False
                self.active_provider = CPU_PROVIDER
                _warn(f"Could not activate QNN session; using CPU fallback ({exc}).")
                raise RuntimeError(f"Could not activate QNN session: {exc}") from exc
            _warn(f"Could not create CPU inference session: {exc}")
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
        try:
            session = self._create_session(path, selected_provider)
            feed = self._input_feed(session, sample_input)
            latencies = []
            outputs = ()
            memory_before = self._memory_usage_mb()
            for _ in range(self.iterations):
                started = time.perf_counter()
                outputs = session.run(None, feed)
                latencies.append((time.perf_counter() - started) * 1000.0)
        except Exception as exc:
            _warn(f"{selected_provider} inference failed: {exc}")
            raise
        average_latency = sum(latencies) / len(latencies)
        units = work_units or self._default_work_units(sample_input, outputs)
        throughput = units / (average_latency / 1000.0)
        self._latencies_ms.extend(latencies)
        self._throughputs.extend([units / (latency / 1000.0) if latency else float("inf") for latency in latencies])
        self._memory_deltas_mb.append(self._memory_usage_mb() - memory_before)
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

    @staticmethod
    def _memory_usage_mb() -> float:
        try:
            import psutil

            return psutil.Process().memory_info().rss / (1024 * 1024)
        except ImportError:
            try:
                import resource

                value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                return value / (1024 * 1024) if sys.platform != "darwin" else value / (1024 * 1024)
            except (ImportError, AttributeError):
                _warn("Process memory metrics are unavailable; reporting a zero memory delta.")
                return 0.0

    @staticmethod
    def _percentile(values: Sequence[float], percentile: float) -> Optional[float]:
        if not values:
            return None
        if np is not None:
            return float(np.percentile(values, percentile))
        ordered = sorted(values)
        index = (len(ordered) - 1) * percentile / 100.0
        lower = int(index)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)

    def benchmark_provider_comparison(
        self, model_path: str, sample_input: Any
    ) -> Dict[str, Any]:
        """Compare QNN (Hexagon HTP) and CPU latency for one model/input."""
        result: Dict[str, Any] = {"qnn": None, "cpu": None}
        try:
            result["qnn"] = self.measure_inference(model_path, sample_input, QNN_PROVIDER)
        except (RuntimeError, ImportError) as exc:
            _warn(f"QNN benchmark unavailable; measuring CPU fallback ({exc}).")
            result["qnn_error"] = str(exc)
        try:
            result["cpu"] = self.measure_inference(model_path, sample_input, CPU_PROVIDER)
        except Exception as exc:
            _warn(f"CPU benchmark failed: {exc}")
            result["cpu_error"] = str(exc)
            result["speedup_factor"] = None
            result["latency_delta_ms"] = None
            return result
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
        average_latency = sum(self._latencies_ms) / len(self._latencies_ms) if self._latencies_ms else None
        average_throughput = sum(self._throughputs) / len(self._throughputs) if self._throughputs else None
        report = {
            "active_provider": self.active_provider,
            "backend_path": self.backend_path,
            "is_npu_active": self.is_npu_active,
            "latency_p50_ms": self._percentile(self._latencies_ms, 50),
            "latency_p95_ms": self._percentile(self._latencies_ms, 95),
            "average_latency_ms": average_latency,
            "throughput_per_second": average_throughput,
            "tokens_per_second": average_throughput,
            "embeddings_per_second": average_throughput,
            "estimated_npu_tops": None,
            "memory_delta_mb": sum(self._memory_deltas_mb) if self._memory_deltas_mb else 0.0,
            "acceleration_status": (
                "QNN / Hexagon HTP active"
                if self.is_npu_active
                else "CPU fallback"
            ),
        }
        if as_dict:
            return report
        try:
            from rich.console import Console
            from rich.table import Table

            table = Table(title="EpisodicFS Qualcomm Telemetry")
            table.add_column("Metric")
            table.add_column("Value")
            for key, value in report.items():
                table.add_row(key.replace("_", " ").title(), str(value))
            console = Console(record=True, width=100)
            console.print(table)
            return console.export_text().rstrip()
        except ImportError:
            return "\n".join(f"{key.replace('_', ' ').title()}: {value}" for key, value in report.items())


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