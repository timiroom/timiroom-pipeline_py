from pydantic import BaseModel, Field


# ── TechStack ──────────────────────────────────────────────────────

class TechStackRequest(BaseModel):
    project_name: str = Field(..., alias="projectName")
    project_description: str = Field(..., alias="projectDescription")
    platform: str

    model_config = {"populate_by_name": True}


class TechStackResponse(BaseModel):
    frontend: list[str] = Field(default_factory=list)
    backend: list[str] = Field(default_factory=list)
    database: list[str] = Field(default_factory=list)
    devops: list[str] = Field(default_factory=list)
    mobile: list[str] = Field(default_factory=list)

    @classmethod
    def default_for(cls, platform: str) -> "TechStackResponse":
        p = platform.upper()
        if p == "APP":
            return cls(frontend=[], backend=["Spring Boot", "Java 21"],
                       database=["PostgreSQL"], devops=["Docker"], mobile=["React Native"])
        if p == "WEB_APP":
            return cls(frontend=["React", "TypeScript"], backend=["Spring Boot", "Java 21"],
                       database=["PostgreSQL"], devops=["Docker"], mobile=["React Native"])
        # WEB (default)
        return cls(frontend=["React", "TypeScript"], backend=["Spring Boot", "Java 21"],
                   database=["PostgreSQL"], devops=["Docker"], mobile=[])


# ── Persona ───────────────────────────────────────────────────────

class ProblemDefinitionSimple(BaseModel):
    current_pain_point: str = Field(..., alias="currentPainPoint")
    current_solution: str = Field(..., alias="currentSolution")
    ideal_state: str = Field(..., alias="idealState")

    model_config = {"populate_by_name": True}


class PersonaRecommendationRequest(BaseModel):
    project_name: str = Field(..., alias="projectName")
    project_description: str = Field(..., alias="projectDescription")
    problem_definition: ProblemDefinitionSimple = Field(..., alias="problemDefinition")

    model_config = {"populate_by_name": True}


class RecommendedPersona(BaseModel):
    persona: str
    usage_environment: str = Field(..., alias="usageEnvironment")
    biggest_pain_point: str = Field(..., alias="biggestPainPoint")

    model_config = {"populate_by_name": True}


class PersonaRecommendationResponse(BaseModel):
    personas: list[RecommendedPersona] = Field(default_factory=list)

    @classmethod
    def empty(cls) -> "PersonaRecommendationResponse":
        return cls(personas=[])


# ── Feature ───────────────────────────────────────────────────────

class TargetUserSimple(BaseModel):
    persona: str
    usage_environment: str = Field("", alias="usageEnvironment")
    biggest_pain_point: str = Field("", alias="biggestPainPoint")

    model_config = {"populate_by_name": True}


class FeatureRecommendationRequest(BaseModel):
    project_name: str = Field(..., alias="projectName")
    project_description: str = Field(..., alias="projectDescription")
    platform: str
    tech_stack: list[str] | None = Field(None, alias="techStack")
    problem_definition: ProblemDefinitionSimple = Field(..., alias="problemDefinition")
    target_users: list[TargetUserSimple] | None = Field(None, alias="targetUsers")

    model_config = {"populate_by_name": True}


class RecommendedFeature(BaseModel):
    priority: str
    feature_name: str = Field(..., alias="featureName")
    description: str = ""

    model_config = {"populate_by_name": True}


class FeatureRecommendationResponse(BaseModel):
    features: list[RecommendedFeature] = Field(default_factory=list)

    @classmethod
    def empty(cls) -> "FeatureRecommendationResponse":
        return cls(features=[])
