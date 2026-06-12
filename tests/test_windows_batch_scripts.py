from pathlib import Path


def test_windows_entrypoint_scripts_exist_and_document_core_commands():
    install = Path("install.bat").read_text()
    startup = Path("startup.bat").read_text()

    assert "py -3.11 -m venv .venv" in install
    assert "pip install torch torchvision" in install
    assert "pip install -r requirements.txt" in install
    assert "pip install -e ." in install
    assert "scripts\\download_models.py" in install
    assert "scripts\\web.py" in startup
    assert "PYTHONPATH=%ROOT%src" in startup
    assert "http://127.0.0.1:8765" in startup
