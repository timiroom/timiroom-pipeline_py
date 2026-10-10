"""Kafka client connection options shared by producer and consumer.

The default deliberately remains PLAINTEXT for the existing in-cluster broker.
Set the SASL_SSL variables only for a secured external listener.
"""

from __future__ import annotations

import ssl
from typing import Any


SUPPORTED_SECURITY_PROTOCOLS = {"PLAINTEXT", "SSL", "SASL_PLAINTEXT", "SASL_SSL"}


def build_kafka_client_options(
    *,
    security_protocol: str = "PLAINTEXT",
    sasl_mechanism: str = "",
    sasl_username: str = "",
    sasl_password: str = "",
    ssl_cafile: str = "",
    ssl_check_hostname: bool = True,
) -> dict[str, Any]:
    """Return aiokafka kwargs without changing legacy plaintext behavior."""
    protocol = security_protocol.upper()
    if protocol not in SUPPORTED_SECURITY_PROTOCOLS:
        raise ValueError(f"지원하지 않는 Kafka 보안 프로토콜: {security_protocol}")
    if protocol == "PLAINTEXT":
        return {}

    options: dict[str, Any] = {"security_protocol": protocol}
    if protocol in {"SSL", "SASL_SSL"}:
        context = ssl.create_default_context(cafile=ssl_cafile or None)
        context.check_hostname = ssl_check_hostname
        options["ssl_context"] = context

    if protocol.startswith("SASL_"):
        if not (sasl_mechanism and sasl_username and sasl_password):
            raise ValueError("SASL Kafka 연결에는 mechanism, username, password가 필요합니다")
        options.update(
            sasl_mechanism=sasl_mechanism,
            sasl_plain_username=sasl_username,
            sasl_plain_password=sasl_password,
        )
    return options
