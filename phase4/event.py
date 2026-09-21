from pydantic import BaseModel, ConfigDict, Field, field_validator


class PipelineResultEvent(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    schema_version: int = Field(default=1, alias="schemaVersion")
    pipeline_id: str = Field(alias="pipelineId", min_length=1)
    project_name: str = Field(alias="projectName", min_length=1)
    platform: str = ""
    tech_stack: list[str] = Field(default_factory=list, alias="techStack")
    user_query: str = Field(default="", alias="userQuery")
    feature_list: list[str] = Field(alias="featureList", min_length=1)
    prd_document: str = Field(alias="prdDocument", min_length=2)
    market_research: str = Field(default="", alias="marketResearch")
    db_schema: str = Field(alias="dbSchema", min_length=2)
    api_spec: str = Field(alias="apiSpec", min_length=2)
    feature_spec_document: str = Field(default="{}", alias="featureSpecDocument")
    feature_registry: list[dict] = Field(default_factory=list, alias="featureRegistry")
    retry_count: int = Field(default=0, alias="retryCount", ge=0)
    created_at: str = Field(alias="createdAt", min_length=1)

    @field_validator("schema_version")
    @classmethod
    def supported_version(cls, value: int) -> int:
        if value != 1:
            raise ValueError(f"지원하지 않는 이벤트 schemaVersion: {value}")
        return value
