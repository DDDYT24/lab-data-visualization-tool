from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
from botocore.session import get_session
from botocore.validate import validate_parameters
from fastapi.testclient import TestClient

from labviz_api.auth import AuthService, SesV2EmailSender, build_auth_service
from labviz_api.config import Settings
from labviz_api.main import create_app
from labviz_api.repository import ProjectRepository
from scripts import probe_ses_delivery


class RecordingSesClient:
    def __init__(self, response: dict[str, Any] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.response = response or {"MessageId": "ses-message-accepted-1"}

    def send_email(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return self.response


class FailingSesClient(RecordingSesClient):
    def send_email(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        raise RuntimeError("simulated provider failure containing no safe operational detail")


def ses_settings(tmp_path: Path) -> Settings:
    return Settings(
        database_path=tmp_path / "ses.db",
        allowed_origins=("http://localhost:3000",),
        public_web_url="http://localhost:3000",
        environment="test",
        auth_mode="ses",
        ses_region="ap-southeast-1",
        ses_from="LabViz <noreply@labviz.example>",
        ses_configuration_set="labviz-test-auth",
    )


def test_ses_v2_sends_one_recipient_with_non_pii_tags_and_records_receipt(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = ses_settings(tmp_path)
    repository = ProjectRepository(settings.database_path, settings.project_ttl_seconds)
    client = RecordingSesClient()
    auth = AuthService(
        SesV2EmailSender(settings, client),
        settings.session_ttl_seconds,
        repository,
    )

    monkeypatch.setattr(logging.getLogger("labviz_api.auth"), "disabled", False)
    with caplog.at_level(logging.INFO, logger="labviz_api.auth"):
        challenge_id, _expires, _resend = auth.request_code("Researcher@Example.com")

    assert len(client.calls) == 1
    request = client.calls[0]
    assert request["FromEmailAddress"] == "LabViz <noreply@labviz.example>"
    assert request["Destination"] == {"ToAddresses": ["researcher@example.com"]}
    assert request["ConfigurationSetName"] == "labviz-test-auth"
    assert request["EmailTags"] == [
        {"Name": "purpose", "Value": "authentication-code"},
        {"Name": "environment", "Value": "test"},
    ]
    assert set(request["Destination"]) == {"ToAddresses"}
    operation = get_session().get_service_model("sesv2").operation_model("SendEmail")
    assert operation.input_shape is not None
    validate_parameters(request, operation.input_shape)

    body = request["Content"]["Simple"]["Body"]["Text"]["Data"]
    code = body.split("code is ", 1)[1].split(".", 1)[0]
    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert "ses-message-accepted-1" in logs
    assert challenge_id in logs
    assert "researcher@example.com" not in logs
    assert code not in logs


def test_ses_delivery_failure_deletes_challenge_and_keeps_stable_api_error(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = ses_settings(tmp_path)
    repository = ProjectRepository(settings.database_path, settings.project_ttl_seconds)
    client = FailingSesClient()
    auth = AuthService(
        SesV2EmailSender(settings, client),
        settings.session_ttl_seconds,
        repository,
    )

    monkeypatch.setattr(logging.getLogger("labviz_api.main"), "disabled", False)
    with (
        caplog.at_level(logging.INFO),
        TestClient(create_app(settings, repository=repository, auth_service=auth)) as api,
    ):
        response = api.post(
            "/api/v1/auth/email-code",
            json={"email": "failure@example.com"},
        )

    assert response.status_code == 503
    assert response.json() == {
        "code": "email-delivery-failed",
        "message": "The verification email could not be sent. Try again later.",
    }
    assert repository.latest_auth_challenge("failure@example.com") is None
    body = client.calls[0]["Content"]["Simple"]["Body"]["Text"]["Data"]
    code = body.split("code is ", 1)[1].split(".", 1)[0]
    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert "authentication-code-delivery-failed" in logs
    assert "failure@example.com" not in logs
    assert code not in logs
    assert "simulated provider failure" not in logs


def test_ses_requires_a_returned_message_identifier(tmp_path: Path) -> None:
    settings = ses_settings(tmp_path)
    sender = SesV2EmailSender(settings, RecordingSesClient({"MessageId": ""}))

    with pytest.raises(RuntimeError, match="message identifier"):
        sender.send_code("recipient@example.com", "123456")


def test_configured_ses_client_uses_region_timeouts_and_standard_credential_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = ses_settings(tmp_path)
    repository = ProjectRepository(settings.database_path, settings.project_ttl_seconds)
    client = RecordingSesClient()
    captured: dict[str, Any] = {}

    def fake_client(service_name: str, **kwargs: Any) -> RecordingSesClient:
        captured["service_name"] = service_name
        captured.update(kwargs)
        return client

    monkeypatch.setattr("labviz_api.auth.boto3.client", fake_client)

    service = build_auth_service(settings, repository)

    assert service.delivery_mode == "email"
    assert captured["service_name"] == "sesv2"
    assert captured["region_name"] == "ap-southeast-1"
    assert captured["config"].connect_timeout == settings.ses_connect_timeout_seconds
    assert captured["config"].read_timeout == settings.ses_read_timeout_seconds
    assert not {
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "endpoint_url",
    }.intersection(captured)


def test_ses_probe_output_contains_receipt_but_no_recipient_or_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = ses_settings(tmp_path)

    class ProbeSender:
        def __init__(self, _settings: Settings) -> None:
            pass

        def send_code(self, email: str, code: str) -> Any:
            assert email == "probe-recipient@example.com"
            assert len(code) == 6 and code.isdigit()
            return type(
                "Receipt",
                (),
                {"message_id": "probe-message-id", "provider": "ses"},
            )()

    monkeypatch.setattr("scripts.probe_ses_delivery.Settings.from_env", lambda: settings)
    monkeypatch.setattr("scripts.probe_ses_delivery.SesV2EmailSender", ProbeSender)
    monkeypatch.setenv(
        probe_ses_delivery.RECIPIENT_ENV,
        "probe-recipient@example.com",
    )
    monkeypatch.setattr("scripts.probe_ses_delivery.secrets.randbelow", lambda _limit: 123456)

    assert probe_ses_delivery.main() == 0
    output = capsys.readouterr().out
    assert "probe-message-id" in output
    assert "probe-recipient@example.com" not in output
    assert "123456" not in output
