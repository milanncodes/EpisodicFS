"""Optional Qualcomm AI Hub profiling with a credential-free demo fallback."""

import os


TARGET_DEVICE = "Snapdragon X Elite CRD"


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