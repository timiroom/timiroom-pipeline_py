from fastapi import APIRouter

from common.api_response import ok
from phase1.recommendation.models import (
    TechStackRequest,
    PersonaRecommendationRequest,
    FeatureRecommendationRequest,
)
from phase1.recommendation.services import (
    TechStackRecommendationService,
    PersonaRecommendationService,
    FeatureRecommendationService,
)

router = APIRouter(prefix="/api/v1/recommendation", tags=["recommendation"])


def _services() -> tuple[
    TechStackRecommendationService,
    PersonaRecommendationService,
    FeatureRecommendationService,
]:
    from main import tech_stack_service, persona_service, feature_service
    return tech_stack_service, persona_service, feature_service


@router.post("/tech-stack")
async def tech_stack(req: TechStackRequest) -> dict:
    svc, _, _ = _services()
    result = await svc.recommend(req.project_name, req.project_description, req.platform)
    return ok(result.model_dump(by_alias=True))


@router.post("/persona")
async def persona(req: PersonaRecommendationRequest) -> dict:
    _, svc, _ = _services()
    result = await svc.recommend(req)
    return ok(result.model_dump(by_alias=True))


@router.post("/features")
async def features(req: FeatureRecommendationRequest) -> dict:
    _, _, svc = _services()
    result = await svc.recommend(req)
    return ok(result.model_dump(by_alias=True))
