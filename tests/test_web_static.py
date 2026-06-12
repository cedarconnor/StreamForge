from pathlib import Path


def test_operator_console_assets_exist_and_reference_api():
    root = Path("src/streamforge/web/static")
    html = (root / "index.html").read_text()
    js = (root / "app.js").read_text()
    css = (root / "styles.css").read_text()
    assert "StreamForge" in html
    assert "/api/validate" in js
    assert "/api/run/start" in js
    assert "/api/status" in js
    assert "setPreviewAspect" in js
    assert "setCropOverlay" in js
    assert "aspect-ratio" in css
    assert "--preview-ratio" in css
