"""Recheck saved live artifacts with current rules; optionally publish a new recovery event."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import settings
from phase2.agents.api_agent import finalize_api_contracts
from phase2.agents.prd_agent import sync_core_features_with_registry
from phase2.agents.qa_agent import QaAgent
from phase2.openapi_spec import build_openapi_document
from phase2.state import PipelineState
from phase3.schema_validator import SchemaValidator
from phase3.validation_service import ValidationService
from phase4.kafka_connection import build_kafka_client_options
from phase4.kafka_producer import KafkaProducerService


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    root = args.source_dir.resolve()
    failure = json.loads((root / "error.json").read_text(encoding="utf-8"))
    artifacts = failure["partialResult"]
    def encode(value):
        return json.dumps(value, ensure_ascii=False)
    state = PipelineState(
        project_name=artifacts["projectName"], platform="WEB",
        feature_list=artifacts["featureNames"], feature_registry=artifacts["featureRegistry"],
        feature_specs=artifacts["featureSpecDocument"]["features"],
        feature_spec_document=encode(artifacts["featureSpecDocument"]),
        prd_document=encode(artifacts["prdDocument"]), db_schema=encode(artifacts["dbSchema"]),
        api_spec=encode(artifacts["apiSpec"]), market_research=artifacts.get("marketResearch", ""),
        project_plan=artifacts.get("projectPlan", {}),
    )
    state = sync_core_features_with_registry(state)
    api = json.loads(state.api_spec)
    api["endpoints"] = finalize_api_contracts(
        api["endpoints"], state.db_schema, state.prd_document,
        state.feature_list, state.feature_registry,
    )
    state = state.copy(api_spec=encode(api), openapi_spec=encode(build_openapi_document(api, state.project_name)))
    state = ValidationService(SchemaValidator()).validate(QaAgent(object())._execute_lightweight_gate(state))
    recovery_id = str(uuid.uuid4())
    output = root / f"recheck-{recovery_id}"
    output.mkdir()
    report = {
        "mode": "saved-artifact-recheck", "source": str(root), "recoveryPipelineId": recovery_id,
        "validated": state.validated, "qaApproved": state.qa_approved,
        "blockers": state.validation_unresolved_blockers,
        "qaBlockers": {"db": state.qa_db_blockers, "api": state.qa_api_blockers, "prd": state.qa_prd_blockers},
        "warnings": {"db": state.qa_db_warnings, "api": state.qa_api_warnings, "prd": state.qa_prd_warnings},
        "published": False, "storedChunksByType": {},
    }
    for name, value in (("apiSpec", state.api_spec), ("dbSchema", state.db_schema),
                        ("prdDocument", state.prd_document), ("openApiSpec", state.openapi_spec)):
        (output / f"{name}.json").write_text(value, encoding="utf-8")
    if args.publish and state.validated and state.qa_approved:
        producer = KafkaProducerService(
            settings.kafka_bootstrap_servers, settings.kafka_topic_pipeline_result, settings.db_url,
            kafka_client_options=build_kafka_client_options(
                security_protocol=settings.kafka_security_protocol,
                sasl_mechanism=settings.kafka_sasl_mechanism,
                sasl_username=settings.kafka_sasl_plain_username,
                sasl_password=settings.kafka_sasl_plain_password,
                ssl_cafile=settings.kafka_ssl_cafile,
                ssl_check_hostname=settings.kafka_ssl_check_hostname,
            ),
        )
        try:
            await producer.publish(state, recovery_id)
            report["published"] = True
        finally:
            await producer.stop()
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            with psycopg2.connect(settings.db_url, connect_timeout=8) as connection, connection.cursor() as cursor:
                cursor.execute(
                    'SELECT metadata->>\'type\', count(*) FROM "' + settings.get_rag_document_table()
                    + '" WHERE metadata->>\'pipeline_id\' = %s GROUP BY 1', (recovery_id,),
                )
                report["storedChunksByType"] = dict(cursor.fetchall())
            if len(report["storedChunksByType"]) >= 8:
                break
            await asyncio.sleep(2)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**report, "outputDir": str(output)}, ensure_ascii=False, indent=2), flush=True)
    return 0 if state.validated and (not args.publish or len(report["storedChunksByType"]) >= 8) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
