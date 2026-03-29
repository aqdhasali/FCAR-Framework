"""
Unit tests for api.py — FastAPI REST endpoints.

Uses httpx + FastAPI TestClient so no actual server process is needed.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api import app  # noqa: E402

client = TestClient(app)


# ═══════════════════════════════════════════════════════════════
# Health check
# ═══════════════════════════════════════════════════════════════


class TestHealth:

    def test_health_ok(self):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "version" in body

    def test_health_returns_service_name(self):
        r = client.get("/health")
        assert r.json()["service"] == "FCAR API"


# ═══════════════════════════════════════════════════════════════
# Datasets listing
# ═══════════════════════════════════════════════════════════════


class TestDatasets:

    def test_lists_datasets(self):
        r = client.get("/datasets")
        assert r.status_code == 200
        datasets = r.json()
        assert isinstance(datasets, list)
        names = [d["dataset"] for d in datasets]
        assert "german" in names

    def test_dataset_has_required_fields(self):
        r = client.get("/datasets")
        ds = r.json()[0]
        for key in ["dataset", "description", "mutable_numeric",
                     "mutable_categorical", "immutable", "sensitive_attributes"]:
            assert key in ds, f"Missing field: {key}"


# ═══════════════════════════════════════════════════════════════
# Recourse endpoint
# ═══════════════════════════════════════════════════════════════


class TestRecourse:

    def test_invalid_dataset(self):
        r = client.post("/recourse", json={
            "dataset": "nonexistent",
            "test_index": 0,
        })
        assert r.status_code == 400

    def test_negative_index(self):
        r = client.post("/recourse", json={
            "dataset": "german",
            "test_index": -1,
        })
        assert r.status_code == 400

    def test_index_out_of_range(self):
        r = client.post("/recourse", json={
            "dataset": "german",
            "test_index": 999999,
        })
        assert r.status_code == 400

    def test_valid_recourse_german(self):
        """Find a rejected German applicant and generate recourse."""
        # First, need to find a rejected test index.
        # Use a known low index; if it's not rejected, the API returns 400
        # and we'll skip.
        r = client.post("/recourse", json={
            "dataset": "german",
            "test_index": 0,
            "use_fcar": False,
        })
        if r.status_code == 400 and "NOT rejected" in r.json().get("detail", ""):
            pytest.skip("Applicant 0 is not rejected; need different index")
        if r.status_code == 200:
            body = r.json()
            assert body["dataset"] == "german"
            assert isinstance(body["score_before"], float)
            assert isinstance(body["score_after"], float)
            assert isinstance(body["flipped"], bool)
            assert isinstance(body["changes"], list)
            assert isinstance(body["narrative"], str)
            assert body["solve_time_seconds"] >= 0

    def test_recourse_response_schema(self):
        """Test that response contains all expected fields."""
        r = client.post("/recourse", json={
            "dataset": "german",
            "test_index": 0,
            "use_fcar": False,
        })
        if r.status_code == 200:
            body = r.json()
            expected = [
                "dataset", "test_index", "use_fcar", "fcar_group",
                "score_before", "score_after", "flipped", "slack",
                "changes", "narrative", "solve_time_seconds",
            ]
            for key in expected:
                assert key in body, f"Missing response field: {key}"

    def test_fcar_mode_accepted(self):
        """FCAR flag should be accepted without error."""
        r = client.post("/recourse", json={
            "dataset": "german",
            "test_index": 0,
            "use_fcar": True,
            "group_col": "age_bucket",
        })
        # Either 200 (success) or 400 (not rejected) — not 422 (validation)
        assert r.status_code in (200, 400)


# ═══════════════════════════════════════════════════════════════
# Audit endpoint
# ═══════════════════════════════════════════════════════════════


class TestAudit:

    def test_invalid_dataset(self):
        r = client.get("/audit/nonexistent")
        assert r.status_code == 400

    def test_german_audit(self):
        r = client.get("/audit/german")
        if r.status_code == 404:
            pytest.skip("No benchmark results for german — run benchmark_ab.py first")
        assert r.status_code == 200
        audits = r.json()
        assert isinstance(audits, list)
        assert len(audits) > 0
        for a in audits:
            assert "audit_score" in a
            assert "social_burden" in a
            assert "method" in a

    def test_filter_by_method(self):
        r = client.get("/audit/german?method=fcar")
        if r.status_code == 404:
            pytest.skip("No benchmark results")
        if r.status_code == 200:
            for a in r.json():
                assert a["method"] == "fcar"


# ═══════════════════════════════════════════════════════════════
# Benchmark endpoint
# ═══════════════════════════════════════════════════════════════


class TestBenchmark:

    def test_invalid_dataset(self):
        r = client.get("/benchmark/nonexistent")
        assert r.status_code == 400

    def test_german_benchmark(self):
        r = client.get("/benchmark/german")
        if r.status_code == 404:
            pytest.skip("No benchmark results for german")
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body, list)
        assert len(body) > 0
        entry = body[0]
        assert "dataset" in entry
        assert "group_col" in entry
        assert "unconstrained_ar" in entry
        assert "fcar" in entry
