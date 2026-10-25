"""Probe new Feature Spec generation against a saved PRD; never publish or claim E2E."""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import settings
from phase2.agents.feature_spec_agent import FeatureSpecAgent
from phase2.state import PipelineState


async def run(source: Path) -> int:
    failure = json.loads((source / "error.json").read_text(encoding="utf-8"))
    artifacts = failure["partialResult"]
    output = Path("test-runs") / f"feature-probe-{datetime.now(timezone.utc):%Y%m%d-%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    settings.feature_spec_diagnostics_dir = str(output / "provider-responses")
    state = PipelineState(
        project_name=artifacts["projectName"], platform="WEB",
        project_plan=artifacts["projectPlan"],
        prd_document=json.dumps(artifacts["prdDocument"], ensure_ascii=False),
        feature_registry=artifacts["featureRegistry"], feature_list=artifacts["featureNames"],
    )
    start = time.perf_counter()
    async with AsyncOpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url,
                           timeout=settings.openai_request_timeout_seconds, max_retries=0,
                           http_client=httpx.AsyncClient(trust_env=False)) as client:
        result = await FeatureSpecAgent(client, settings.feature_spec_model or settings.openai_chat_model).execute(state)
    report = {"mode": "feature-spec-only-new-generation", "published": False,
              "sourcePrd": str(source.resolve()), "elapsedSeconds": round(time.perf_counter() - start, 3),
              "success": not result.generation_blockers, "blockers": result.generation_blockers,
              "diagnostics": result.generation_diagnostics, "featureRegistry": result.feature_registry,
              "featureSpecDocument": json.loads(result.feature_spec_document) if result.feature_spec_document else None}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), **{key: report[key] for key in ("mode", "success", "elapsedSeconds", "blockers")}}, ensure_ascii=False), flush=True)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run", type=Path)
    raise SystemExit(asyncio.run(run(parser.parse_args().source_run)))
