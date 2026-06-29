import asyncio
import json
import logging

from fastapi import APIRouter
from openai import InternalServerError, APITimeoutError, APIConnectionError
from pydantic import BaseModel

from common.api_response import ok
from phase1.models import FormData, MoSCoW, PlatformType
from phase2.json_utils import try_parse_json

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


def _exaone_endpoint_id() -> str:
    from main import settings
    return settings.exaone_endpoint_id


# ── 수집 프롬프트: 6가지 질문 (프로젝트명 제외) ───────────────────────
COLLECTION_PROMPT_TEMPLATE = """## 출력 형식 — 절대 규칙
JSON 하나만 출력하세요. 다른 텍스트 절대 금지.
항상 아래 형식 그대로:
{{"isComplete": false, "message": "질문 1문장", "suggestions": ["답변예시1", "답변예시2", "답변예시3"]}}

## 역할
스타트업 기획 인터뷰어 AI.
대화 히스토리에서 사용자가 몇 번째 항목까지 답했는지 세고, 다음 항목을 질문한다.
사용자가 이미 말한 내용을 바탕으로 정확히 다음 질문에 맞는 suggestions을 생성한다.

## 수집 순서 (반드시 이 순서대로, 1개씩)

1️⃣ 플랫폼 (WEB / APP / WEB_APP 중 선택)
   - "웹사이트(WEB)로 만들고 싶어요"
   - "모바일 앱(APP)으로 만들고 싶어요"
   - "웹과 앱 둘 다(WEB_APP) 필요해요"

2️⃣ 현재 불편한 점 (사용자가 실제로 겪는 문제)
   - "문제/불편함이 구체적으로 뭔가요?"
   - NOT: 해결책이나 기능 제시

3️⃣ 현재 해결 방법 (지금 어떻게 해결 중인가?)
   - "현재는 어떻게 해결하고 있나요?"
   - NOT: 또 다른 불편함, NOT: 이상적인 상태

4️⃣ 원하는 이상적 상태 (해결된다면 어떻게?)
   - "이렇게 되면 좋겠다는 것은?"
   - NOT: 불편함, NOT: 현재 상황

5️⃣ 타겟 유저 (누가 사용할 것인가?)
   - "주로 누가 사용할까요?"
   - 예: "20-30대 자취생", "요리를 자주 하는 사람"

6️⃣ 핵심 기능 3가지 (기능명만, MUST/SHOULD 없이)
   - "꼭 필요한 기능 3가지를 말씀해 주세요"
   - 예: "재료 추가", "유통기한 알림", "쇼핑리스트 생성"
   - NOT: "~하는 기능(MUST)", NOT: 상세한 설명

## suggestions 생성 규칙 (매우 중요!)
- 반드시 정확히 3개
- **지금 묻는 질문에만** 정확히 맞는 답변 예시
- 이전 질문이나 다음 질문의 답변으로 착각하지 말 것
- 사용자가 이미 말한 맥락을 고려하되, 다양한 예시 제시
- 클릭하면 그대로 전송 가능한 완성된 문장

## 절대 주의
- 이미 사용자가 답한 항목은 건너뛰고 다음 항목만 질문
- message는 1문장, 친근한 톤
- suggestions은 절대 비워두지 말 것
- 각 suggestion은 정해진 질문에만 맞아야 함
  예: 2번(불편함) 질문인데 3번(현재방법) 답변을 suggestions로 주면 안됨
  예: 3번(현재방법) 질문인데 2번(불편함) 답변을 suggestions로 주면 안됨

## 참고 정보 (이미 수집한 내용)
{context}
"""

# ── 합성 프롬프트 ───────────────────────────────────────────────────
SYNTHESIS_PROMPT = """대화를 분석해서 아래 형식의 JSON을 출력하세요.
JSON 하나만 출력하세요. 마크다운·설명 금지. {{ 로 시작해서 }} 로 끝납니다.

{{"projectName": "이름", "projectDescription": "설명 1-2문장", "platform": "WEB", "techStack": [], "problemDefinition": {{"currentPainPoint": "불편함", "currentSolution": "현재방법", "idealState": "이상적상태", "businessImpact": null, "motivation": null, "competitorGap": null}}, "targetUsers": [{{"persona": "유저설명", "usageEnvironment": "사용환경", "biggestPainPoint": "주요불편함"}}], "featureDefinition": {{"commonFeatures": [], "customFeatures": [{{"featureName": "기능명", "description": "기능설명", "priority": "MUST"}}]}}}}

규칙:
- platform: WEB, APP, WEB_APP 중 하나
- priority: MUST 또는 SHOULD로 자동 할당 (사용자가 순서대로 말한 기능이므로, 처음 3개는 우선순위를 자동으로 분배)
- projectName: 대화에서 유추 (사용자가 명시하지 않으면 기능과 불편함으로부터 추론)
- projectDescription: 현재 불편함과 이상적 상태로부터 생성
- 대화에서 언급 없는 필드는 합리적으로 추론해서 채우기
- 빈 문자열("") 금지"""


class ChatMessageDto(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessageDto]


async def _call_exaone(client, messages: list[dict], max_tokens: int, temperature: float = 0.7) -> str:
    """EXAONE 호출 (재시도 3회)"""
    for attempt in range(3):
        try:
            resp = await client.chat.completions.create(
                model=_exaone_endpoint_id(),
                max_tokens=max_tokens,
                frequency_penalty=0.3,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=messages,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return resp.choices[0].message.content or ""
        except (InternalServerError, APITimeoutError, APIConnectionError) as e:
            logger.warning("EXAONE 채팅 일시 오류 (attempt %d): %s — 재시도", attempt + 1, e)
            if attempt < 2:
                await asyncio.sleep(3 * (attempt + 1))
            else:
                raise
    return ""


def _get_user_message_count(messages: list[ChatMessageDto]) -> int:
    """사용자 메시지 개수 반환"""
    return sum(1 for m in messages if m.role == "user")


def _build_context_string(messages: list[ChatMessageDto]) -> str:
    """이미 수집한 내용을 문자열로 변환 (프로젝트명 제외)"""
    questions = [
        "플랫폼",
        "현재 불편한 점",
        "현재 해결 방법",
        "원하는 이상적 상태",
        "타겟 유저",
        "핵심 기능 3가지"
    ]
    
    user_count = _get_user_message_count(messages)
    if user_count == 0:
        return "(아직 수집한 정보 없음)"
    
    lines = []
    user_idx = 0

    for m in messages:
        if m.role == "user":
            q_idx = user_idx - 1  # 첫 메시지는 초기 아이디어, 컬렉션 답변 아님
            if 0 <= q_idx < len(questions):
                lines.append(f"{q_idx + 1}. {questions[q_idx]}: {m.content}")
            user_idx += 1

    return "\n".join(lines) if lines else "(아직 수집한 정보 없음)"


def _normalize_form_data(node: dict) -> dict:
    """EXAONE 출력의 흔한 포맷 오류를 FormData 스키마에 맞게 정규화."""
    # platform 정규화
    platform_map = {
        "앱": "APP", "모바일": "APP", "MOBILE": "APP",
        "웹": "WEB", "WEB ONLY": "WEB",
        "앱+웹": "WEB_APP", "웹+앱": "WEB_APP", "BOTH": "WEB_APP",
    }
    valid_platforms = {p.value for p in PlatformType}
    p = node.get("platform", "APP")
    if isinstance(p, str) and p not in valid_platforms:
        node["platform"] = platform_map.get(p, "APP")

    # featureDefinition 정규화
    fd = node.get("featureDefinition")
    if isinstance(fd, dict):
        # priority 정규화
        priority_map = {
            "필수": "MUST", "HIGH": "MUST", "높음": "MUST",
            "권장": "SHOULD", "MEDIUM": "SHOULD", "중간": "SHOULD",
            "선택": "COULD", "LOW": "COULD", "낮음": "COULD",
        }
        valid_priorities = {m.value for m in MoSCoW}
        for cf in fd.get("customFeatures") or []:
            if isinstance(cf, dict):
                pri = cf.get("priority", "MUST")
                if pri not in valid_priorities:
                    cf["priority"] = priority_map.get(pri, "MUST")

        # commonFeatures: 문자열 아이템 → 객체 변환
        common = fd.get("commonFeatures") or []
        if common and isinstance(common[0], str):
            fd["commonFeatures"] = [{"featureName": f} for f in common]

    # problemDefinition 필수 필드 보완
    pd = node.get("problemDefinition")
    if isinstance(pd, dict):
        if not pd.get("currentPainPoint"):
            pd["currentPainPoint"] = pd.get("painPoint") or pd.get("problem") or "정보 없음"
        if not pd.get("currentSolution"):
            pd["currentSolution"] = pd.get("solution") or "정보 없음"
        if not pd.get("idealState"):
            pd["idealState"] = pd.get("goal") or pd.get("ideal") or "정보 없음"

    # targetUsers 필수 필드 보완
    for tu in node.get("targetUsers") or []:
        if isinstance(tu, dict):
            if not tu.get("persona"):
                tu["persona"] = "일반 사용자"
            if not tu.get("usageEnvironment"):
                tu["usageEnvironment"] = tu.get("environment") or "일반적인 환경"
            if not tu.get("biggestPainPoint"):
                tu["biggestPainPoint"] = tu.get("painPoint") or "정보 없음"

    return node


async def _generate_dynamic_suggestions(
    user_message_count: int,
    context: str,
    messages: list[ChatMessageDto],
    client
) -> list[str]:
    """
    사용자 컨텍스트를 기반으로 동적 suggestions 생성
    """
    questions = [
        "플랫폼",
        "현재 불편한 점",
        "현재 해결 방법",
        "원하는 이상적 상태",
        "타겟 유저",
        "핵심 기능 3가지"
    ]
    
    # 첫 번째 user 메시지는 초기 아이디어(컬렉션 답변 아님)이므로 -1 offset
    question_idx = user_message_count - 1
    if question_idx < 0 or question_idx >= len(questions):
        return []

    next_question = questions[question_idx]

    # 각 질문별 명확한 지침
    question_guidelines = {
        0: "플랫폼: WEB, APP, WEB_APP 중 하나만. '~으로 만들고 싶어요' 형식",
        1: "현재 불편한 점: 사용자가 실제로 느끼는 문제나 불편함. '~하기가 어려워요', '~가 문제예요' 형식",
        2: "현재 해결 방법: 지금 현재 어떻게 해결하고 있는지. '~를 사용해요', '~로 관리해요', '~를 수동으로 해요' 형식",
        3: "원하는 이상적 상태: 문제가 해결된다면 어떻게 되면 좋을지. '~가 자동으로 되면 좋겠어요', '~가 한눈에 보이면 좋겠어요' 형식",
        4: "타겟 유저: 누가 이 서비스를 사용할 것인가. '~세대 사람들', '~직업을 가진 사람들' 형식",
        5: "핵심 기능 3가지: 기능명만 제시 (MUST/SHOULD 없이). '재료 추가', '유통기한 알림', '쇼핑리스트 생성' 형식"
    }

    guideline = question_guidelines.get(question_idx, "")

    # 동적 suggestions 프롬프트 - 매우 명확하게
    dynamic_prompt = f"""사용자의 다음 질문에 대한 좋은 예시 3개를 JSON으로 생성하세요.

현재까지 수집한 정보:
{context}

다음 질문 #{question_idx + 1}: {next_question}

질문 가이드:
{guideline}

답변 형식:
{{"suggestions": ["완성된_답변1", "완성된_답변2", "완성된_답변3"]}}

중요한 규칙:
- 절대 이전 질문의 답변으로 혼동하지 말 것
- 사용자가 이미 말한 맥락을 고려하되, 다양한 예시 제시
- 각 suggestion은 사용자가 그대로 입력할 수 있는 완성된 문장
- 반드시 정확히 3개
- 질문에 정확히 맞는 내용만"""

    try:
        messages_for_suggestions = [
            {"role": "system", "content": "사용자의 맥락을 기반으로 각 질문에 정확히 맞는 suggestions을 생성합니다. JSON만 출력하세요."},
            {"role": "user", "content": dynamic_prompt}
        ]
        
        raw = await _call_exaone(client, messages_for_suggestions, max_tokens=500, temperature=0.8)
        node = try_parse_json(raw)
        
        if node and isinstance(node, dict):
            suggestions = node.get("suggestions") or []
            if isinstance(suggestions, list) and len(suggestions) == 3:
                logger.info("동적 suggestions 생성 성공 (질문: %s)", next_question)
                return suggestions
    
    except Exception as e:
        logger.warning("동적 suggestions 생성 실패: %s", e)
    
    return []


async def _generate_project_name_candidates(messages: list[ChatMessageDto], client) -> list[str]:
    """
    사용자의 답변을 분석해서 프로젝트 이름 3가지 후보 생성
    """
    lines = []
    for m in messages:
        prefix = "사용자" if m.role == "user" else "AI"
        lines.append(f"{prefix}: {m.content}")
    conversation = "\n".join(lines)
    
    name_prompt = f"""다음 대화를 분석해서 프로젝트의 핵심을 반영하는 프로젝트 이름 3가지를 생성하세요.

대화:
{conversation}

요구사항:
- 프로젝트의 핵심 가치나 문제 해결을 반영
- 기억하기 쉽고 멋있는 이름
- 한국어 또는 영어 모두 가능
- 실제로 사용할 수 있는 이름

응답 형식 (JSON만):
{{"names": ["이름1", "이름2", "이름3"], "reasoning": "왜 이 이름들을 제안했는지 간단한 설명"}}

예시:
- 불편함: "냉장고 뭐가 있는지 몰라서 낭비", 해결: "냉장고 재고 자동 추적"
  → "FridgeTracker", "냉동고", "FreshKeep"
"""

    try:
        name_msgs = [
            {"role": "system", "content": "사용자의 프로젝트 정보를 분석해서 창의적이고 의미있는 프로젝트 이름 3가지를 생성합니다. JSON만 출력하세요."},
            {"role": "user", "content": name_prompt}
        ]
        
        raw = await _call_exaone(client, name_msgs, max_tokens=500, temperature=0.8)
        node = try_parse_json(raw)
        
        if node and isinstance(node, dict):
            names = node.get("names") or []
            if isinstance(names, list) and len(names) == 3:
                logger.info("프로젝트 이름 후보 생성 성공: %s", names)
                return names
    
    except Exception as e:
        logger.warning("프로젝트 이름 생성 실패: %s", e)
    
    return []


async def _synthesize_form_data(messages: list[ChatMessageDto], client) -> dict | None:
    """대화 내용에서 중첩 FormData를 직접 합성 (최대 2회 시도)."""
    lines = []
    for m in messages:
        prefix = "사용자" if m.role == "user" else "AI"
        lines.append(f"{prefix}: {m.content}")
    conversation = "\n".join(lines)

    synthesis_msgs = [
        {"role": "system", "content": SYNTHESIS_PROMPT},
        {"role": "user", "content": f"대화:\n{conversation}"},
    ]

    for attempt in range(2):
        try:
            raw = await _call_exaone(client, synthesis_msgs, max_tokens=2000, temperature=0.5)
            logger.debug("Synthesis attempt %d raw (%.500s)", attempt + 1, raw)

            node = try_parse_json(raw)
            if node and isinstance(node, dict) and node.get("projectName") and node.get("featureDefinition"):
                node = _normalize_form_data(node)
                try:
                    FormData.model_validate(node)
                    logger.info("Synthesis OK (Pydantic 검증 통과) | attempt=%d", attempt + 1)
                    return node
                except Exception as ve:
                    logger.warning(
                        "Synthesis attempt %d: Pydantic 검증 실패 — %s | node keys: %s",
                        attempt + 1, ve, list(node.keys()),
                    )
                    continue

            logger.warning("Synthesis attempt %d: 파싱 실패 또는 필수 필드 없음", attempt + 1)
        except Exception as e:
            logger.error("합성 LLM 호출 실패 (attempt %d): %s", attempt + 1, e, exc_info=True)

    return None


@router.post("/message")
async def message(req: ChatRequest) -> dict:
    from main import exaone_client

    user_msg_count = _get_user_message_count(req.messages)
    context = _build_context_string(req.messages)
    
    # assistant 메시지를 완전히 래핑 (suggestions 포함)
    collection_messages = [{"role": "system", "content": COLLECTION_PROMPT_TEMPLATE.format(context=context)}]
    
    for m in req.messages:
        if m.role not in ("user", "assistant"):
            continue
        if m.role == "assistant":
            # assistant 메시지 완전 래핑
            try:
                existing = json.loads(m.content)
                wrapped = json.dumps(existing, ensure_ascii=False)
            except:
                # JSON 파싱 실패 시 그냥 메시지만 래핑
                wrapped = json.dumps(
                    {"isComplete": False, "message": m.content},
                    ensure_ascii=False,
                )
            collection_messages.append({"role": "assistant", "content": wrapped})
        else:
            collection_messages.append({"role": "user", "content": m.content})

    try:
        raw = await _call_exaone(exaone_client, collection_messages, max_tokens=800)
        logger.debug("Collection raw (%.400s)", raw)

        node = try_parse_json(raw)
        if node is None or not isinstance(node, dict):
            logger.warning("JSON 파싱 실패, plain text 사용 | raw: %.300s", raw)
            plain = raw.strip()
            return ok({"message": plain, "isComplete": False, "suggestions": [], "formData": None})

        suggestions = (
            node.get("suggestions")
            or node.get("sugations")   # EXAONE 오타 방어
            or node.get("suggestion")  # 단수형 방어
            or []
        )
        suggestions = suggestions if isinstance(suggestions, list) else []

        # suggestions가 비었으면 동적 생성 시도
        if not suggestions:
            logger.warning("Suggestions 비어있음 — 동적 생성 시도")
            suggestions = await _generate_dynamic_suggestions(user_msg_count, context, req.messages, exaone_client)
        
        # 동적 생성도 실패하면 재시도
        if not suggestions:
            logger.warning("동적 생성 실패 — EXAONE 재호출 (temperature 증가)")
            raw2 = await _call_exaone(exaone_client, collection_messages, max_tokens=800, temperature=0.9)
            node2 = try_parse_json(raw2)
            if node2 and isinstance(node2, dict):
                s2 = (node2.get("suggestions") or node2.get("sugations") or node2.get("suggestion") or [])
                if isinstance(s2, list) and s2:
                    node = node2
                    suggestions = s2

        # 모든 시도 실패 시 질문별 정적 fallback
        if not suggestions:
            logger.warning("모든 suggestions 생성 실패 — 정적 fallback 사용 (user_msgs=%d)", user_msg_count)
            _static_fallback = [
                ["웹사이트(WEB)로 만들고 싶어요", "모바일 앱(APP)으로 만들고 싶어요", "웹과 앱 둘 다(WEB_APP) 필요해요"],
                ["매번 직접 확인해야 해서 번거로워요", "기존 방법이 너무 비효율적이에요", "원하는 정보를 찾기가 어려워요"],
                ["수동으로 직접 관리해요", "스프레드시트나 메모로 기록해요", "별도 앱을 여러 개 사용해요"],
                ["한 곳에서 한눈에 볼 수 있으면 좋겠어요", "자동으로 처리되면 좋겠어요", "알림을 받을 수 있으면 좋겠어요"],
                ["20-30대 직장인", "해당 분야에 관심 있는 모든 사람", "특정 문제를 겪고 있는 사용자"],
                ["핵심 기능 1", "핵심 기능 2", "핵심 기능 3"],
            ]
            q_idx = user_msg_count - 1
            if 0 <= q_idx < len(_static_fallback):
                suggestions = _static_fallback[q_idx]

        # 초기 아이디어(1) + 6개 질문 답변 = 7개 user 메시지 후 합성 진입
        collection_done = user_msg_count >= 7

        if collection_done:
            logger.info("합성 단계 진입 | user_msgs=%d", user_msg_count)

            # 1단계: 프로젝트 이름 후보 제시 (user_msg_count == 7)
            if user_msg_count == 7:
                project_name_candidates = await _generate_project_name_candidates(req.messages, exaone_client)

                if project_name_candidates:
                    logger.info("프로젝트 이름 후보 제시 | candidates=%s", project_name_candidates)
                    return ok({
                        "message": "좋아요! 충분한 정보가 모였어요. 이 프로젝트에 어울리는 이름이 뭘까요?",
                        "isComplete": False,
                        "suggestions": project_name_candidates,
                        "formData": None,
                        "stage": "naming"
                    })

                logger.warning("프로젝트 이름 생성 실패 — 바로 합성 진행")

            # 2단계: FormData 합성 (user_msg_count >= 8 또는 이름 생성 실패)
            form_data = await _synthesize_form_data(req.messages, exaone_client)
            if form_data:
                # user_msg_count >= 8이면 마지막 user 메시지가 프로젝트명
                if user_msg_count >= 8 and len(req.messages) > 0:
                    last_user_msg = None
                    for m in reversed(req.messages):
                        if m.role == "user":
                            last_user_msg = m.content
                            break

                    if last_user_msg:
                        form_data["projectName"] = last_user_msg
                
                return ok({
                    "message": "좋아요! 충분한 정보가 모였어요. 지금 바로 프로젝트를 시작할게요!",
                    "isComplete": True,
                    "suggestions": [],
                    "formData": form_data,
                    "stage": "complete"
                })
            
            # 합성 실패 — 수집 계속
            logger.warning("합성 실패 — 수집 계속 (user_msgs=%d)", user_msg_count)
            return ok({
                "message": "조금 더 자세히 알려주시면 더 잘 기획할 수 있어요. 핵심 기능이나 목표 유저에 대해 추가로 말씀해 주세요.",
                "isComplete": False,
                "suggestions": ["핵심 기능 추가 설명", "타겟 유저 설명", "경쟁 서비스 언급"],
                "formData": None,
            })

        return ok({
            "message": node.get("message", ""),
            "isComplete": False,
            "suggestions": suggestions,
            "formData": None,
        })

    except Exception as e:
        logger.error("EXAONE 호출 실패: %s", e, exc_info=True)
        return ok({"message": "잠시 후 다시 시도해 주세요.", "isComplete": False, "suggestions": [], "formData": None})