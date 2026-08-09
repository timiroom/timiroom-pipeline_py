"""Run the intake chat flow against the configured EXAONE endpoint.

This mirrors the message history that Spring ChatService sends to
POST /api/v1/chat/message, without starting DB, Kafka, or the RAG stack.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from openai import AsyncOpenAI


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings
from routers.chat import ChatMessageDto, ChatRequest, message


ANSWERS = [
    "팀 할 일이 메신저에 흩어져서 중요한 업무를 놓치는 문제를 해결하고 싶어요",
    "웹과 모바일 앱을 모두 만들고 싶어요",
    "담당자와 마감일이 여러 대화방에 흩어져서 해야 할 일을 자주 놓쳐요",
    "지금은 메신저 메시지를 보면서 캘린더와 메모 앱에 직접 기록해요",
    "대화에서 할 일을 자동으로 모으고 담당자와 마감일을 한눈에 확인하고 싶어요",
    "여러 프로젝트를 동시에 진행하며 메신저로 협업하는 20~30대 직장인 팀원",
    "할 일 자동 추출, 담당자와 마감일 지정, 마감 전 알림",
    "팀플로우",
]

INITIAL_GREETING = (
    "안녕하세요! 프로젝트 기획을 시작해 볼게요. "
    "어떤 서비스를 만들고 싶으신가요? 어떤 문제를 해결하는 서비스인지 자유롭게 설명해 주세요."
)


async def run() -> None:
    if not settings.exaone_api_key or not settings.exaone_endpoint_id:
        raise RuntimeError("EXAONE_API_KEY와 EXAONE_ENDPOINT_ID가 필요합니다")

    client = AsyncOpenAI(
        api_key=settings.exaone_api_key,
        base_url="https://api.friendli.ai/dedicated/v1",
    )
    # routers.chat intentionally resolves these lazily from main. Supplying this
    # minimal module keeps the probe isolated from the full application startup.
    sys.modules["main"] = SimpleNamespace(settings=settings, exaone_client=client)

    history = [ChatMessageDto(role="assistant", content=INITIAL_GREETING)]
    started = time.perf_counter()
    for turn, answer in enumerate(ANSWERS, start=1):
        history.append(ChatMessageDto(role="user", content=answer))
        turn_started = time.perf_counter()
        response = await message(ChatRequest(messages=history))
        elapsed = time.perf_counter() - turn_started
        data = response["data"]
        print(json.dumps({
            "turn": turn,
            "user": answer,
            "elapsedSeconds": round(elapsed, 2),
            **data,
        }, ensure_ascii=False, indent=2))
        history.append(ChatMessageDto(role="assistant", content=data["message"]))

    print(json.dumps({"totalSeconds": round(time.perf_counter() - started, 2)}, indent=2))


if __name__ == "__main__":
    asyncio.run(run())
