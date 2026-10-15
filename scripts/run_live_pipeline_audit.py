"""Run a live chat-to-Phase4 scenario and persist every terminal artifact for audit."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx


BASE_URL = "http://127.0.0.1:8081"
ANSWERS = [
    "초보 반려식물 사용자가 식물별 물주기와 생육 상태를 기록하고 이상 징후를 조기에 확인하기 어려운 문제를 해결하고 싶어요",
    "웹사이트로 만들고 싶어요",
    "여러 식물의 물주기 시기를 기억하지 못해 과습이나 건조가 발생하고, 잎 변색을 발견해도 원인과 조치를 판단하기 어려운 순간이 가장 불편해요",
    "현재는 휴대폰 알람과 메모장에 식물 이름과 물 준 날짜를 따로 기록하고 있어요",
    "사용자가 식물을 등록하고 물주기와 생육 사진을 기록하며, 식물별 관리 일정을 안내받고 이상 징후와 조치 방법을 확인하면 좋겠어요",
    "처음 식물을 키우는 1인 가구와 여러 반려식물을 관리하는 취미 재배자",
    "식물 등록, 물주기 기록, 생육 사진 기록, 관리 일정 알림, 이상 징후 기록과 조치 안내",
    "그린로그",
]


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    started = time.perf_counter()
    output_dir = Path("test-runs") / f"live-plant-care-{datetime.now():%Y%m%d-%H%M%S}"
    output_dir.mkdir(parents=True, exist_ok=False)
    messages: list[dict[str, str]] = []
    chat_trace: list[dict] = []
    form_data = None

    with httpx.Client(timeout=httpx.Timeout(1200.0, connect=10.0)) as client:
        for index, answer in enumerate(ANSWERS):
            messages.append({"role": "user", "content": answer})
            turn_started = time.perf_counter()
            response = client.post(
                f"{BASE_URL}/api/v1/chat/message", json={"messages": messages}
            )
            response.raise_for_status()
            data = response.json().get("data", {})
            chat_trace.append({
                "turn": index,
                "answer": answer,
                "durationSeconds": round(time.perf_counter() - turn_started, 3),
                "response": data,
            })
            print(
                f"CHAT {index}: {data.get('stage')} complete={data.get('isComplete')}",
                flush=True,
            )
            messages.append({"role": "assistant", "content": json.dumps(data, ensure_ascii=False)})
            if data.get("isComplete") and data.get("formData"):
                form_data = data["formData"]
                break

        _write_json(output_dir / "chat_trace.json", chat_trace)
        if not form_data:
            raise RuntimeError("Chat did not produce formData")
        _write_json(output_dir / "form_data.json", form_data)

        response = client.post(
            f"{BASE_URL}/api/v1/orchestration/generate",
            data={"request": json.dumps(form_data, ensure_ascii=False), "skip_phase1": "false"},
        )
        response.raise_for_status()
        pipeline_id = response.json()["data"]["pipelineId"]
        print(f"PIPELINE_ID={pipeline_id}", flush=True)

        events: list[dict] = []
        with client.stream(
            "GET", f"{BASE_URL}/api/v1/orchestration/progress/{pipeline_id}"
        ) as stream:
            event_name = ""
            for raw_line in stream.iter_lines():
                line = raw_line.strip()
                if line.startswith("event:"):
                    event_name = line[6:].strip()
                    continue
                if not line.startswith("data:"):
                    continue
                payload = json.loads(line[5:].strip())
                events.append({
                    "event": event_name,
                    "elapsedSeconds": round(time.perf_counter() - started, 3),
                    "payload": payload,
                })
                if event_name == "progress":
                    print(
                        f"{payload.get('percent')}% {payload.get('step')}: {payload.get('message')}",
                        flush=True,
                    )
                elif event_name == "complete":
                    _write_json(output_dir / "final_result.json", payload["result"])
                    _write_json(output_dir / "events.json", events)
                    print(f"COMPLETE output={output_dir.resolve()}", flush=True)
                    return 0
                elif event_name == "error":
                    _write_json(output_dir / "error.json", payload)
                    _write_json(output_dir / "events.json", events)
                    print(f"ERROR output={output_dir.resolve()}", flush=True)
                    return 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
