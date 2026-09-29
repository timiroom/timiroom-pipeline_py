"""EXAONE 엔드포인트 단독 테스트용 스크립트.
FastAPI 앱(DB/Kafka/reranker) 없이 콘솔에서 바로 대화하며 응답 시간을 확인한다.

사용법: python test_exaone_chat.py
"""
import asyncio
import io
import sys
import time

# Windows cp949 터미널에서 한글 입출력 깨짐 방지 (main.py와 동일 처리)
if hasattr(sys.stdin, "buffer"):
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from openai import AsyncOpenAI

from config.settings import settings

client = AsyncOpenAI(api_key=settings.exaone_api_key, base_url="https://api.friendli.ai/dedicated/v1")


async def main():
    print(f"EXAONE endpoint: {settings.exaone_endpoint_id}")
    print("종료하려면 exit 입력\n")

    history: list[dict] = []

    while True:
        user_input = input("나: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})

        t0 = time.time()
        resp = await client.chat.completions.create(
            model=settings.exaone_endpoint_id,
            max_tokens=500,
            temperature=1.0,
            top_p=0.95,
            messages=history,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        elapsed = time.time() - t0

        reply = resp.choices[0].message.content or ""
        history.append({"role": "assistant", "content": reply})

        print(f"EXAONE ({elapsed:.2f}s): {reply}\n")


if __name__ == "__main__":
    asyncio.run(main())
