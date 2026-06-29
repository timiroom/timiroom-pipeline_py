import logging

from openai import AsyncOpenAI

from phase2.json_utils import try_parse_json

from .models import (
    TechStackResponse,
    PersonaRecommendationRequest,
    PersonaRecommendationResponse,
    FeatureRecommendationRequest,
    FeatureRecommendationResponse,
)

logger = logging.getLogger(__name__)


class TechStackRecommendationService:

    def __init__(self, client: AsyncOpenAI, model: str):
        self._client = client
        self._model = model

    async def recommend(self, project_name: str, desc: str, platform: str) -> TechStackResponse:
        prompt = f"""프로젝트명: {project_name}
설명: {desc}
플랫폼: {platform}

아래 JSON 형식으로 파트별 기술 스택을 추천해주세요.
각 파트에 2~3개를 추천하세요.
플랫폼이 WEB이면 mobile은 빈 배열로 반환하세요.
플랫폼이 APP이면 frontend는 빈 배열로 반환하세요.

{{"frontend":[],"backend":[],"database":[],"devops":[],"mobile":[]}}"""

        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                max_tokens=512,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": "당신은 소프트웨어 아키텍처 전문가입니다.\n반드시 JSON만 반환하고 코드 블록이나 다른 텍스트는 포함하지 마세요."},
                    {"role": "user", "content": prompt},
                ],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            raw = (resp.choices[0].message.content or "").strip()
            data = try_parse_json(raw)
            if data is None:
                raise ValueError(f"JSON 파싱 실패: {raw[:200]}")
            return TechStackResponse.model_validate(data)
        except Exception as e:
            logger.warning("기술 스택 추천 실패, 기본값 반환: %s", e)
            return TechStackResponse.default_for(platform)


class PersonaRecommendationService:

    def __init__(self, client: AsyncOpenAI, model: str):
        self._client = client
        self._model = model

    async def recommend(self, req: PersonaRecommendationRequest) -> PersonaRecommendationResponse:
        pd = req.problem_definition
        prompt = f"""프로젝트명: {req.project_name}
설명: {req.project_description}
핵심 문제: {pd.current_pain_point}
현재 해결 방식: {pd.current_solution}
이상적인 상태: {pd.ideal_state}

위 서비스를 가장 필요로 할 타겟 유저 2명의 페르소나를 추천해주세요.
실제로 존재할 법한 구체적인 사람으로 작성해주세요.

{{"personas":[{{"persona":"","usageEnvironment":"","biggestPainPoint":""}}]}}"""

        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                max_tokens=512,
                temperature=0.4,
                messages=[
                    {"role": "system", "content": "당신은 UX 리서처입니다.\n서비스 정보를 보고 가장 핵심적인 타겟 유저 페르소나를 추천하세요.\n반드시 JSON만 반환하고 코드 블록이나 다른 텍스트는 포함하지 마세요."},
                    {"role": "user", "content": prompt},
                ],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            raw = (resp.choices[0].message.content or "").strip()
            data = try_parse_json(raw)
            if data is None:
                raise ValueError(f"JSON 파싱 실패: {raw[:200]}")
            return PersonaRecommendationResponse.model_validate(data)
        except Exception as e:
            logger.warning("페르소나 추천 실패: %s", e)
            return PersonaRecommendationResponse.empty()


class FeatureRecommendationService:

    def __init__(self, client: AsyncOpenAI, model: str):
        self._client = client
        self._model = model

    async def recommend(self, req: FeatureRecommendationRequest) -> FeatureRecommendationResponse:
        pd = req.problem_definition
        users = "정보 없음"
        if req.target_users:
            users = ", ".join(
                f"{u.persona} / {u.biggest_pain_point}" for u in req.target_users
            )
        tech = ", ".join(req.tech_stack) if req.tech_stack else ""

        prompt = f"""프로젝트명: {req.project_name}
설명: {req.project_description}
플랫폼: {req.platform}
기술 스택: {tech}
핵심 문제: {pd.current_pain_point}
타겟 유저: {users}

이 서비스에 필요한 기능을 MoSCoW 우선순위로 5~8개 추천해주세요.
- MUST: MVP에 반드시 필요한 기능
- SHOULD: 가능하면 포함할 기능
- COULD: 여유 있으면 포함할 기능
- WONT: 명시적으로 제외할 기능

{{"features":[{{"priority":"MUST","featureName":"","description":""}}]}}"""

        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                max_tokens=1024,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": "당신은 프로덕트 매니저입니다.\n프로젝트 정보를 보고 필요한 기능을 MoSCoW 우선순위로 추천하세요.\n반드시 JSON만 반환하고 코드 블록이나 다른 텍스트는 포함하지 마세요."},
                    {"role": "user", "content": prompt},
                ],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            raw = (resp.choices[0].message.content or "").strip()
            data = try_parse_json(raw)
            if data is None:
                raise ValueError(f"JSON 파싱 실패: {raw[:200]}")
            return FeatureRecommendationResponse.model_validate(data)
        except Exception as e:
            logger.warning("기능 추천 실패: %s", e)
            return FeatureRecommendationResponse.empty()
