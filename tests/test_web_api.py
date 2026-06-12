from fastapi.testclient import TestClient

from streamforge.web.app import create_app


class FakeRunner:
    def __init__(self):
        self.started = False

    def validate(self, config):
        return {"ok": True, "source": {"width": 640, "height": 480}, "aspect": {"crop_direction": "none"}}

    def start(self, config):
        self.started = True

    def stop(self):
        self.started = False

    def status(self):
        return {"running": self.started, "emitted": 0, "repeats": 0, "filled": 0}


def test_validate_endpoint():
    app = create_app(FakeRunner())
    client = TestClient(app)
    res = client.post("/api/validate", json={"source_type": "webcam", "source_name": "0"})
    assert res.status_code == 200
    assert res.json()["ok"] is True


def test_start_status_stop():
    runner = FakeRunner()
    client = TestClient(create_app(runner))
    assert client.post("/api/run/start", json={"source_type": "webcam", "source_name": "0"}).status_code == 200
    assert client.get("/api/status").json()["running"] is True
    assert client.post("/api/run/stop").status_code == 200
    assert client.get("/api/status").json()["running"] is False
