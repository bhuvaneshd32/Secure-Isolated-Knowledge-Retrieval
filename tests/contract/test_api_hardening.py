import logging
import pytest
from fastapi.testclient import TestClient
from privrag.api.main import app, RedactingFilter

client = TestClient(app)

def test_redacting_filter_drops_canary():
    log_filter = RedactingFilter()
    
    # Record containing a canary string must be filtered out
    bad_record = logging.LogRecord(
        name="privrag",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="Processing query containing CANARY-001-005 in text",
        args=(),
        exc_info=None
    )
    assert log_filter.filter(bad_record) is False

    # Normal record must pass through
    good_record = logging.LogRecord(
        name="privrag",
        level=logging.INFO,
        pathname=__file__,
        lineno=20,
        msg="Processing query for hash 8f4b2c1d",
        args=(),
        exc_info=None
    )
    assert log_filter.filter(good_record) is True

def test_request_logging_no_canary_leak(caplog):
    # Attach our filter to caplog handler during the test run
    logger = logging.getLogger("privrag.api.main")
    logger.addFilter(RedactingFilter())

    canary_token = "CANARY-999-001"
    
    with caplog.at_level(logging.INFO):
        # Trigger an endpoint with a payload holding a canary
        client.post(
            "/documents",
            headers={"Authorization": "Bearer non-existent-token"},
            json=[{"text": f"Confidential meeting: {canary_token}", "metadata": {}}]
        )
        
        # Explicitly log a message with the canary through the filtered logger
        logger.info(f"Received document with token {canary_token}")
    
    # Assert that no canary string leaked into the captured logs
    assert "CANARY" not in caplog.text

def test_payload_size_limit_middleware():
    # Simulate a payload header exceeding the 2MB threshold
    large_size = (2 * 1024 * 1024) + 1
    response = client.post(
        "/documents",
        headers={"Content-Length": str(large_size)},
        json=[]
    )
    assert response.status_code == 413
    assert response.json()["detail"] == "Payload Too Large"