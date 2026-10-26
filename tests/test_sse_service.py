import asyncio

from phase2.sse_service import PipelineProgressService


async def _collect(service: PipelineProgressService, pipeline_id: str) -> list[str]:
    return [chunk async for chunk in service.subscribe(pipeline_id)]


def test_late_subscriber_receives_error_details() -> None:
    asyncio.run(_test_late_subscriber_receives_error_details())


async def _test_late_subscriber_receives_error_details() -> None:
    service = PipelineProgressService()
    service.error(
        "pipeline-1",
        "자동 검증 실패",
        {"validationBlockers": ["API endpoint 누락"], "blockerDetails": {"api": ["POST /plots"]}},
    )

    events = await asyncio.wait_for(_collect(service, "pipeline-1"), timeout=1)

    assert len(events) == 1
    assert events[0].startswith("event: error\n")
    assert "validationBlockers" in events[0]
    assert "POST /plots" in events[0]


def test_late_subscriber_receives_completed_result() -> None:
    asyncio.run(_test_late_subscriber_receives_completed_result())


async def _test_late_subscriber_receives_completed_result() -> None:
    service = PipelineProgressService()
    service.complete("pipeline-2", {"status": "SUCCESS", "featureSummary": {"totalCount": 4}})

    events = await asyncio.wait_for(_collect(service, "pipeline-2"), timeout=1)

    assert len(events) == 1
    assert events[0].startswith("event: complete\n")
    assert "featureSummary" in events[0]


def test_terminal_event_retention_is_bounded() -> None:
    service = PipelineProgressService(terminal_ttl_seconds=60, max_terminal_events=1)
    service.error("pipeline-1", "first")
    service.error("pipeline-2", "second")

    assert "pipeline-1" not in service._terminal_events
    assert "pipeline-2" in service._terminal_events


def test_pre_subscription_progress_is_not_replayed_twice() -> None:
    asyncio.run(_test_pre_subscription_progress_is_not_replayed_twice())


async def _test_pre_subscription_progress_is_not_replayed_twice() -> None:
    service = PipelineProgressService()
    service.send("pipeline-3", "PHASE1_START", "시작", 5)

    async def collect():
        return [chunk async for chunk in service.subscribe("pipeline-3")]

    task = asyncio.create_task(collect())
    await asyncio.sleep(0)
    service.send("pipeline-3", "PHASE1_DONE", "완료", 25)
    service.complete("pipeline-3", {"status": "SUCCESS"})
    events = await asyncio.wait_for(task, timeout=1)

    assert sum('"step": "PHASE1_START"' in event for event in events) == 1
    assert sum('"step": "PHASE1_DONE"' in event for event in events) == 1


def test_backend_watcher_and_browser_both_receive_terminal_and_progress():
    async def scenario():
        service = PipelineProgressService()
        service.send("p", "FEATURE_SPEC", "generating", 55)
        first = asyncio.create_task(_collect(service, "p"))
        second = asyncio.create_task(_collect(service, "p"))
        await asyncio.sleep(0)
        service.error("p", "failed", {"failedStage": "FEATURE_SPEC", "skippedStages": ["DBA_API"]})
        events = await asyncio.wait_for(asyncio.gather(first, second), 1)
        assert events[0] == events[1]
        assert len(events[0]) == 2
        assert "p" not in service._queues
    asyncio.run(scenario())


def test_idle_llm_call_emits_heartbeats_without_replaying_progress():
    async def scenario():
        service = PipelineProgressService(heartbeat_seconds=0.01)
        service.send("slow", "PRD", "작성 중", 50)
        stream = service.subscribe("slow")
        assert 'event: progress' in await anext(stream)
        assert 'event: heartbeat' in await asyncio.wait_for(anext(stream), 0.5)
        assert service.snapshot("slow")["lastProgress"]["step"] == "PRD"
        service.complete("slow", {"storage": {"status": "stored"}})
        assert 'event: complete' in await anext(stream)
        await stream.aclose()
        assert "slow" not in service._queues
    asyncio.run(scenario())


def test_unknown_worker_run_is_not_an_infinite_running_stream():
    service = PipelineProgressService()
    events = asyncio.run(_collect(service, "missing"))
    assert len(events) == 1 and events[0].startswith("event: connection")
    assert "PROGRESS_NOT_FOUND" in events[0]
    assert service.snapshot("missing")["status"] == "UNKNOWN"
    assert not service._queues


def test_snapshot_recovers_a_missed_terminal_event_without_generation():
    service = PipelineProgressService()
    service.send("failed", "FEATURE_SPEC", "생성 중", 55)
    service.error("failed", "계약 누락", {"failedStage": "FEATURE_SPEC"})
    assert service.snapshot("failed")["error"]["failedStage"] == "FEATURE_SPEC"
    service.complete("stored", {"storage": {"status": "stored"}})
    assert service.snapshot("stored") == {"status": "COMPLETED", "result": {"storage": {"status": "stored"}}}


def test_state_endpoint_returns_missed_terminal_without_starting_pipeline(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from routers import orchestration

    service = PipelineProgressService()
    pipeline_id = "3d7d00ad-35d8-4578-b8f2-3ff3134c0b43"
    service.complete(pipeline_id, {"storage": {"status": "stored"}})
    monkeypatch.setattr(orchestration, "_services", lambda: (None, None, None, None, None, service, None))
    app = FastAPI()
    app.include_router(orchestration.router)
    response = TestClient(app).get(f"/api/v1/orchestration/state/{pipeline_id}")
    assert response.status_code == 200
    assert response.json()["data"] == {"status": "COMPLETED", "result": {"storage": {"status": "stored"}}}
