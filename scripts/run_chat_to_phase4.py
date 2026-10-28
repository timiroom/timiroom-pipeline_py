"""Drive one new topic from intake chat through Phase 4 storage and save the trace.

Usage: python scripts/run_chat_to_phase4.py <slug> [answers.json]
The run is stored under test-runs/live-<slug>-<UTC timestamp>/.
"""
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = "http://localhost:8081"
ROOT = Path(__file__).resolve().parents[1]
STARTED = time.time()

DEFAULT_ANSWERS = [
    "아파트 단지 주민끼리 전동드릴, 사다리 같은 공구를 빌려주고 빌리는데 누가 무엇을 가지고 있는지 몰라서 매번 단체 대화방에 물어봐야 하는 문제를 해결하고 싶어요",
    "웹사이트로 만들고 싶어요",
    "필요한 공구를 가진 이웃을 찾고 반납 날짜를 맞추는 과정이 대화방 메시지에 묻혀서 가장 번거로워요",
    "지금은 단지 단체 대화방에 글을 올리고 개인 메시지로 대여 날짜를 정해요",
    "주민이 보유 공구를 등록하고 다른 주민이 대여를 신청하면 소유자가 승인하고, 대여 기간과 반납 여부를 한 화면에서 확인하면 좋겠어요. 결제, 보증금, 배송, 실시간 채팅은 초기 범위에서 제외해요",
    "같은 아파트 단지에 사는 30~50대 주민",
    "공구 등록, 대여 신청과 승인, 반납 확인",
]
NAME = "이웃공구함"


def stamp(message: str) -> None:
    print(f"[T+{time.time() - STARTED:7.2f}s] {message}", flush=True)


def save(run_dir: Path, name: str, value) -> None:
    (run_dir / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def chat(run_dir: Path, answers: list[str], name: str = NAME) -> dict:
    messages: list[dict] = []
    trace: list[dict] = []
    pending = list(answers)
    for turn in range(len(answers) + 4):
        answer = pending.pop(0) if pending else name
        messages.append({"role": "user", "content": answer})
        began = time.time()
        response = requests.post(f"{BASE}/api/v1/chat/message", json={"messages": messages}, timeout=180)
        response.raise_for_status()
        data = response.json().get("data", {})
        trace.append({"turn": turn, "answer": answer, "durationSeconds": round(time.time() - began, 3), "response": data})
        save(run_dir, "chat_trace.json", trace)
        stamp(f"CHAT {turn}: stage={data.get('stage')} complete={data.get('isComplete')}")
        messages.append({"role": "assistant", "content": data.get("message") or ""})
        if data.get("isComplete") and data.get("formData"):
            return data["formData"]
    raise RuntimeError("chat did not produce formData")


def main() -> int:
    slug = sys.argv[1] if len(sys.argv) > 1 else "tool-lending"
    answers, name = DEFAULT_ANSWERS, NAME
    if len(sys.argv) > 2:
        # Either a plain answer list or {"name": ..., "answers": [...]}.
        topic = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
        answers, name = (topic["answers"], topic.get("name", NAME)) if isinstance(topic, dict) else (topic, NAME)
    run_dir = ROOT / "test-runs" / f"live-{slug}-{datetime.now(timezone.utc):%Y%m%d-%H%M%SZ}"
    run_dir.mkdir(parents=True)
    stamp(f"RUN_DIR={run_dir}")

    form_data = chat(run_dir, answers, name)
    save(run_dir, "form_data.json", form_data)

    response = requests.post(
        f"{BASE}/api/v1/orchestration/generate",
        data={"request": json.dumps(form_data, ensure_ascii=False), "skip_phase1": "false"},
        timeout=60,
    )
    response.raise_for_status()
    pipeline_id = response.json()["data"]["pipelineId"]
    (run_dir / "pipeline_id.txt").write_text(pipeline_id, encoding="utf-8")
    stamp(f"PIPELINE_ID={pipeline_id}")

    events: list[dict] = []
    terminal = None
    with requests.get(f"{BASE}/api/v1/orchestration/progress/{pipeline_id}", stream=True, timeout=(10, 3600)) as stream:
        event = ""
        for raw in stream.iter_lines(decode_unicode=True):
            line = (raw or "").strip()
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                payload = line.split(":", 1)[1].strip()
                try:
                    body = json.loads(payload)
                except json.JSONDecodeError:
                    body = payload
                events.append({"t": round(time.time() - STARTED, 2), "event": event, "data": body})
                save(run_dir, "events.json", events)
                if isinstance(body, dict) and "step" in body:
                    stamp(f"{body.get('progress')}% {body.get('step')}: {body.get('message')}")
                else:
                    stamp(f"SSE {event}: {payload[:400]}")
                if event in {"complete", "error"}:
                    terminal = (event, body)
                    break

    if terminal is None:
        stamp("STREAM ENDED WITHOUT TERMINAL EVENT")
        return 2
    event, body = terminal
    save(run_dir, "result.json" if event == "complete" else "error.json", body)
    storage = requests.get(f"{BASE}/api/v1/orchestration/storage/{pipeline_id}", timeout=30).json()
    save(run_dir, "storage.json", storage)
    stamp(f"STORAGE={json.dumps(storage, ensure_ascii=False)[:400]}")
    stored = event == "complete" and (storage.get("data") or {}).get("status") == "stored"
    stamp("RESULT=STORED" if stored else f"RESULT=NOT_STORED ({event})")
    return 0 if stored else 1


if __name__ == "__main__":
    sys.exit(main())
