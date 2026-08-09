"""Run a non-refrigerator E2E scenario through chat and orchestration."""
import io
import json
import sys
import time

import requests


if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = "http://localhost:8081"
STARTED = time.time()


def stamp(message: str) -> None:
    print(f"[T+{time.time() - STARTED:7.2f}s] {message}", flush=True)


answers = [
    "반려동물 예방접종 기록을 잊지 않고 관리하고 병원 방문 시 이력을 바로 보여주는 서비스를 만들고 싶어요",
    "모바일 앱(MOBILE)으로 만들고 싶어요",
    "접종 날짜와 다음 접종 시기를 메모에서 찾느라 자주 놓쳐요",
    "종이 수첩과 휴대폰 캘린더에 따로 기록하고 있어요",
    "반려동물별 접종 이력이 정리되고 다음 접종 전에 알림을 받으면 좋겠어요",
    "예방접종 일정을 꾸준히 관리해야 하는 반려동물 보호자",
    "반려동물 등록, 예방접종 기록, 다음 접종 알림",
]

messages: list[dict] = []
form_data = None
for index, answer in enumerate(answers):
    messages.append({"role": "user", "content": answer})
    response = requests.post(
        f"{BASE}/api/v1/chat/message", json={"messages": messages}, timeout=120,
    )
    stamp(f"chat {index}: HTTP {response.status_code}")
    response.raise_for_status()
    data = response.json().get("data", {})
    print(f"  stage={data.get('stage')} complete={data.get('isComplete')} message={data.get('message')}")
    messages.append({"role": "assistant", "content": json.dumps(data, ensure_ascii=False)})
    if data.get("isComplete") and data.get("formData"):
        form_data = data["formData"]
        break

if form_data is None:
    messages.append({"role": "user", "content": "펫케어 캘린더"})
    response = requests.post(
        f"{BASE}/api/v1/chat/message", json={"messages": messages}, timeout=120,
    )
    response.raise_for_status()
    data = response.json().get("data", {})
    stamp(f"chat naming: complete={data.get('isComplete')}")
    form_data = data.get("formData")

if not form_data:
    raise RuntimeError("chat did not produce formData")

print("FORM_DATA=" + json.dumps(form_data, ensure_ascii=False))
response = requests.post(
    f"{BASE}/api/v1/orchestration/generate",
    data={"request": json.dumps(form_data, ensure_ascii=False), "skip_phase1": "false"},
    files=[], timeout=60,
)
response.raise_for_status()
pipeline_id = response.json()["data"]["pipelineId"]
stamp(f"PIPELINE_ID={pipeline_id}")

with requests.get(
    f"{BASE}/api/v1/orchestration/progress/{pipeline_id}", stream=True, timeout=600,
) as stream:
    event = ""
    for raw_line in stream.iter_lines(decode_unicode=True):
        line = (raw_line or "").strip()
        if line.startswith("event:"):
            event = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            payload = line.split(":", 1)[1].strip()
            stamp(f"SSE {event}: {payload[:600]}")
            if event in {"complete", "error"}:
                break

