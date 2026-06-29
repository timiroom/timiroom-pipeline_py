"""채팅 → 파이프라인 생성 전체 흐름 테스트"""
import asyncio
import json
import httpx

CHAT_URL = "http://localhost:8081/api/v1/chat/message"
PIPELINE_URL = "http://localhost:8080/api/v1/pipeline/start"

CONVERSATION = [
    ("user", "냉장고 재료로 레시피를 추천해주는 앱 만들고 싶어요"),
    ("user", "모바일 앱(APP)으로 만들고 싶어요"),
    ("user", "냉장고에 뭐가 있는지 몰라서 재료를 낭비하고 요리 결정을 못 해요"),
    ("user", "메모장에 직접 적거나 냉장고 문에 붙여둔 메모지로 관리해요"),
    ("user", "앱을 열면 바로 가진 재료로 만들 수 있는 레시피가 자동 추천되면 좋겠어요"),
    ("user", "요리에 관심 있는 20-30대 자취생이나 바쁜 직장인"),
    ("user", "재료 추가/수정, 레시피 자동 추천, 유통기한 알림"),  # 기존에 스킵되던 6번 기능 질문
]

async def main():
    messages = []
    form_data = None

    async with httpx.AsyncClient(timeout=30) as client:
        for i, (role, content) in enumerate(CONVERSATION):
            messages.append({"role": role, "content": content})

            resp = await client.post(CHAT_URL, json={"messages": messages})
            resp.raise_for_status()
            data = resp.json()["data"]

            print(f"\n{'='*60}")
            print(f"[user_msg={i+1}] 사용자: {content[:50]}")
            print(f"BOT: {data['message']}")
            print(f"suggestions: {data.get('suggestions', [])}")
            print(f"isComplete: {data['isComplete']}")
            if data.get("stage"):
                print(f"stage: {data['stage']}")

            if data.get("formData"):
                form_data = data["formData"]
                print(f"\n✅ FormData 수신! projectName={form_data.get('projectName')}")
                break

            # 봇 응답을 다음 요청에 포함
            messages.append({"role": "assistant", "content": data["message"]})

            # naming 단계면 프로젝트 이름 선택
            if data.get("stage") == "naming":
                candidates = data.get("suggestions", [])
                chosen = candidates[0] if candidates else "FridgeChef"
                print(f"\n→ 프로젝트 이름 선택: {chosen}")
                messages.append({"role": "user", "content": chosen})

                resp2 = await client.post(CHAT_URL, json={"messages": messages})
                resp2.raise_for_status()
                data2 = resp2.json()["data"]

                print(f"\n{'='*60}")
                print(f"[naming 응답] isComplete: {data2['isComplete']}")
                if data2.get("formData"):
                    form_data = data2["formData"]
                    print(f"[OK] FormData 수신! projectName={form_data.get('projectName')}")
                break

    print(f"\n{'='*60}")
    if not form_data:
        print("[FAIL] FormData 수신 실패 -- 합성 미완료")
        return

    # FormData 내용 출력
    print("\n[FormData 요약]")
    print(f"  projectName: {form_data.get('projectName')}")
    print(f"  platform: {form_data.get('platform')}")
    pd = form_data.get("problemDefinition", {})
    print(f"  painPoint: {pd.get('currentPainPoint', '')[:80]}")
    print(f"  idealState: {pd.get('idealState', '')[:80]}")
    fd = form_data.get("featureDefinition", {})
    features = fd.get("customFeatures", [])
    print(f"  features ({len(features)}개):")
    for f in features:
        print(f"    - [{f.get('priority')}] {f.get('featureName')}: {f.get('description', '')[:50]}")

    # 파이프라인 트리거
    print(f"\n[Pipeline] POST {PIPELINE_URL}")
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            pr = await client.post(PIPELINE_URL, json=form_data)
            print(f"  status: {pr.status_code}")
            print(f"  response: {pr.text[:500]}")
        except Exception as e:
            print(f"  pipeline trigger error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
