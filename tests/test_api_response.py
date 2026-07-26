from common.api_response import ok


def test_ok_wraps_pipeline_id_for_backend_contract() -> None:
    assert ok({"pipelineId": "pipeline-1"}) == {
        "success": True,
        "code": "SUCCESS",
        "data": {"pipelineId": "pipeline-1"},
    }
