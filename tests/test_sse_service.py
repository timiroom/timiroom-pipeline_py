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
