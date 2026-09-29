from enum import Enum
from pydantic import BaseModel, Field


class PlatformType(str, Enum):
    WEB = "WEB"
    APP = "APP"
    WEB_APP = "WEB_APP"


class MoSCoW(str, Enum):
    MUST = "MUST"
    SHOULD = "SHOULD"
    COULD = "COULD"
    WONT = "WONT"


class ProblemDefinition(BaseModel):
    current_pain_point: str = Field(..., alias="currentPainPoint")
    current_solution: str = Field(..., alias="currentSolution")
    ideal_state: str = Field(..., alias="idealState")
    business_impact: str | None = Field(None, alias="businessImpact")
    motivation: str | None = None
    competitor_gap: str | None = Field(None, alias="competitorGap")

    model_config = {"populate_by_name": True}


class TargetUser(BaseModel):
    persona: str
    usage_environment: str = Field(..., alias="usageEnvironment")
    biggest_pain_point: str = Field(..., alias="biggestPainPoint")

    model_config = {"populate_by_name": True}


class CommonFeature(BaseModel):
    feature_name: str = Field(..., alias="featureName")
    selected: bool = False
    description: str | None = None

    model_config = {"populate_by_name": True, "extra": "forbid"}


class CustomFeature(BaseModel):
    feature_name: str = Field(..., alias="featureName")
    priority: MoSCoW = MoSCoW.MUST
    description: str | None = None

    model_config = {"populate_by_name": True, "extra": "forbid"}


class FeatureDefinition(BaseModel):
    common_features: list[CommonFeature] | None = Field(None, alias="commonFeatures")
    custom_features: list[CustomFeature] | None = Field(None, alias="customFeatures")

    model_config = {"populate_by_name": True, "extra": "forbid"}


class FormData(BaseModel):
    project_name: str = Field(..., alias="projectName")
    project_description: str = Field(..., alias="projectDescription")
    platform: PlatformType
    tech_stack: list[str] | None = Field(None, alias="techStack")
    problem_definition: ProblemDefinition = Field(..., alias="problemDefinition")
    target_users: list[TargetUser] = Field(..., min_length=1, alias="targetUsers")
    feature_definition: FeatureDefinition = Field(..., alias="featureDefinition")

    model_config = {"populate_by_name": True}
