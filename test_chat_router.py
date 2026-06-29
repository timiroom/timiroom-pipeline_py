"""chat router 로직을 직접 실행해서 예외 추적."""
import asyncio, sys, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

async def main():
    from config.settings import settings
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=settings.exaone_api_key,
        base_url="https://api.friendli.ai/dedicated/v1",
    )

    # _build_messages 로직
    SYSTEM_PROMPT = open("routers/chat.py", encoding="utf-8").read()
    # 실제 SYSTEM_PROMPT 추출
    import importlib.util, types
    spec = importlib.util.spec_from_file_location("chat", "routers/chat.py")
    mod = importlib.util.module_from_spec(spec)
    # main 의존성 mock
    import unittest.mock as mock
    sys.modules['main'] = mock.MagicMock(
        exaone_client=client,
        settings=settings,
    )
    spec.loader.exec_module(mod)

    SYSTEM_PROMPT = mod.SYSTEM_PROMPT
    print("SYSTEM_PROMPT 길이:", len(SYSTEM_PROMPT))

    from pydantic import BaseModel

    class Msg(BaseModel):
        role: str
        content: str

    msgs = [Msg(role="user", content="반려동물 건강 관리 앱을 만들고 싶어")]
    chat_messages = mod._build_messages(msgs)
    print("chat_messages 수:", len(chat_messages))

    try:
        resp = await client.chat.completions.create(
            model=settings.exaone_endpoint_id,
            max_tokens=500,
            messages=chat_messages,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw = resp.choices[0].message.content or ""
        print("RAW (200자):", raw[:200])

        node = mod._parse_response(raw)
        print("PARSED:", json.dumps(node, ensure_ascii=False, indent=2)[:300] if node else "None")
    except Exception as e:
        import traceback
        print("EXCEPTION:")
        traceback.print_exc()

asyncio.run(main())
