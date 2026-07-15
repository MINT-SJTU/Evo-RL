from types import SimpleNamespace

import lerobot.scripts.lerobot_setup_can as setup_can_module
import lerobot.utils.piper_sdk as piper_sdk_utils


def test_resolve_piper_can_interface(monkeypatch, tmp_path):
    interface_path = tmp_path / "can2"
    interface_path.mkdir()
    (interface_path / "type").write_text("280\n")
    monkeypatch.setattr(piper_sdk_utils, "_SYS_CLASS_NET", tmp_path)
    monkeypatch.setattr(
        piper_sdk_utils.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="ID_SERIAL_SHORT=SERIAL_A\n"),
    )

    assert piper_sdk_utils.resolve_piper_can_interface("SERIAL_A") == "can2"


def test_setup_can_uses_classic_can_for_usb_serials(monkeypatch):
    monkeypatch.setattr(setup_can_module, "resolve_piper_can_interface", lambda serial: "can2")
    attempts = []

    def fake_setup(interface, bitrate, data_bitrate, use_fd):
        attempts.append((interface, bitrate, use_fd))
        return False, "failed"

    monkeypatch.setattr(setup_can_module, "setup_interface", fake_setup)

    cfg = setup_can_module.CANSetupConfig(mode="setup", usb_can_serials="SERIAL_A")
    setup_can_module.setup_interface_with_fallback(cfg.get_interfaces()[0], cfg)

    assert attempts == [("can2", 1000000, False)]
