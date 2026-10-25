"""Resume real downstream processing from a verified live Feature Spec checkpoint."""

import argparse
import asyncio
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main as application
from phase2.feature_generation import document_issues, source_cores
from phase2.state import PipelineState
from routers.orchestration import _artifact_snapshot


async def run(checkpoint: Path) -> int:
    report = json.loads(checkpoint.read_text(encoding="utf-8"))
    if not report.get("success") or report.get("published") or report.get("mode") != "feature-spec-only-new-generation":
        raise ValueError("A successful, unpublished live Feature Spec checkpoint is required")
    source = Path(report["sourcePrd"])
    original = json.loads((source / "error.json").read_text(encoding="utf-8"))["partialResult"]
    features = report["featureRegistry"]
    prd = original["prdDocument"]
    cores = source_cores(prd)
    issues = document_issues(features, cores, preflight=True)
    if issues:
        raise ValueError("Checkpoint no longer validates: " + "; ".join(issues))
    by_source = {feature["sourceFeatureId"]: feature for feature in features if feature.get("source") == "prd_core"}
    prd = {**prd, "coreFeatures": [{**core, **by_source[core["sourceFeatureId"]]} for core in cores]}
    encode = lambda value: json.dumps(value, ensure_ascii=False)
    state = PipelineState(project_name=original["projectName"], platform="WEB",
        project_plan=original["projectPlan"], market_research=original.get("marketResearch", ""),
        prd_document=encode(prd), feature_registry=features, feature_specs=features,
        feature_list=[feature["name"] for feature in features], feature_spec_document=encode({"features": features}),
        generation_diagnostics=report["diagnostics"], last_completed_stage="FEATURE_SPEC", rag_context_fallback=True)
    graph = application.orchestration_graph
    state = graph._freeze_feature_registry(state)
    identifier = str(uuid.uuid4())
    output = Path("test-runs") / f"checkpoint-phase4-{datetime.now(timezone.utc):%Y%m%d-%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    start, events = time.perf_counter(), []
    def event(stage):
        elapsed = round(time.perf_counter() - start, 3)
        events.append({"stage": stage, "elapsedSeconds": elapsed})
        print(f"{stage} elapsed={elapsed}s", flush=True)
    async with application.lifespan(application.app):
        event("DBA_API")
        dba, api = await asyncio.gather(application.dba_agent.execute(state), application.api_agent.execute(state))
        state = graph._merge_agent_quality(state, dba, api).copy(db_schema=dba.db_schema, api_spec=api.api_spec, openapi_spec=api.openapi_spec)
        state = graph._align_api_to_final_db(state)
        (output / "generated_artifacts.json").write_text(json.dumps(_artifact_snapshot(state), ensure_ascii=False, indent=2), encoding="utf-8")
        event("QA")
        state = await application.qa_agent.execute(state)
        if state.qa_repair_issues:
            state = await graph._repair_once(state, None)
        event("PHASE3")
        state = application.validation_service.validate(state)
        result = {**_artifact_snapshot(state), "pipelineId": identifier, "runMode": "verified-checkpoint-resume",
                  "sourceCheckpoint": str(checkpoint.resolve()), "sourceChatRun": str(source),
                  "qa": {"approved": state.qa_approved, "blockingIssues": {"db": state.qa_db_blockers, "api": state.qa_api_blockers, "prd": state.qa_prd_blockers}}}
        filename = "error.json"
        if state.validated and state.qa_approved and not state.generation_blockers:
            event("PHASE4")
            await application.kafka_producer_service.publish(state, identifier)
            result["storage"] = await application.kafka_producer_service.confirm_storage(state, identifier)
            if result["storage"].get("status") == "stored":
                filename = "final_result.json"
        else:
            result["validationErrors"] = state.validation_unresolved_blockers or ([state.last_validation_error] if state.last_validation_error else [])
            result["generationBlockers"] = state.generation_blockers
        event("COMPLETE" if filename == "final_result.json" else "FAILED")
        (output / filename).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "events.json").write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(output.resolve()), "pipelineId": identifier, "mode": result["runMode"],
                          "stored": filename == "final_result.json", "storage": result.get("storage"),
                          "validationErrors": result.get("validationErrors")}, ensure_ascii=False), flush=True)
    return 0 if filename == "final_result.json" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    raise SystemExit(asyncio.run(run(parser.parse_args().checkpoint)))
