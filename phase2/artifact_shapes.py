"""Strict container boundaries; semantic checks run only on safe shapes."""

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from phase2.validation_contract import ContractIssue
from phase2.workflow_contract import IntervalPolicy, ReadDependency, StatePolicy


class Shape(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)


class Parameter(Shape):
    name: str = ""
    schema_value: dict = Field(default_factory=dict, alias="schema")


class Endpoint(Shape):
    method: str = "GET"
    path: str = ""
    parameters: list[Parameter] = Field(default_factory=list)
    featureIds: list[str] = Field(default_factory=list)
    requestSchema: dict = Field(default_factory=dict)
    responseSchema: dict = Field(default_factory=dict)
    fieldMappings: list[dict] = Field(default_factory=list)
    effects: list[dict] = Field(default_factory=list)


class Column(Shape):
    name: str
    type: str = ""
    constraints: str = ""


class Reference(Shape):
    table: str
    column: str = "id"
    columns: list[str] = Field(default_factory=list)


class ForeignKey(Shape):
    column: str = ""
    columns: list[str] = Field(default_factory=list)
    references: Reference
    onDelete: str = "NO ACTION"


class Table(Shape):
    name: str
    columns: list[Column] = Field(default_factory=list)
    indexes: list[str | dict] = Field(default_factory=list)
    constraints: list[str | dict] = Field(default_factory=list)
    foreignKeys: list[ForeignKey] = Field(default_factory=list)
    featureIds: list[str] = Field(default_factory=list)


class Database(Shape):
    tables: list[Table] = Field(default_factory=list)
    relationships: list[str | dict] = Field(default_factory=list)
    featureMappings: list[dict] = Field(default_factory=list)


class Api(Shape):
    endpoints: list[Endpoint] = Field(default_factory=list)
    schemas: dict[str, dict] = Field(default_factory=dict)
    featureMappings: list[dict] = Field(default_factory=list)


class DbContract(Shape):
    tables: list[str | dict] = Field(default_factory=list)
    foreignKeys: list[str | ForeignKey] = Field(default_factory=list)
    fields: list[dict] = Field(default_factory=list)
    uniqueConstraints: list[dict] = Field(default_factory=list)


class Feature(Shape):
    featureId: str = ""
    sourceFeatureId: str | None = None
    nonApplicabilityReasons: dict[str, str] = Field(default_factory=dict)
    id: str = ""
    parentFeatureId: str | None = None
    priority: str = ""
    priorityScore: int = 0
    transactionRules: list[str | dict] = Field(default_factory=list)
    states: list[str | dict] = Field(default_factory=list)
    stateTransitions: list[str | dict] = Field(default_factory=list)
    outputRequirements: list[str | dict] = Field(default_factory=list)
    name: str = ""
    actions: list[str] = Field(default_factory=list)
    roles: list[str] = Field(default_factory=list)
    apiContract: list[Endpoint] = Field(default_factory=list)
    dbContract: DbContract = Field(default_factory=DbContract)
    sessionPolicy: dict = Field(default_factory=dict)
    requirements: list[str | dict] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    applicability: dict[str, bool] = Field(default_factory=dict)
    readDependencies: list[ReadDependency] = Field(default_factory=list)
    statePolicies: list[StatePolicy] = Field(default_factory=list)
    intervalPolicies: list[IntervalPolicy] = Field(default_factory=list)


class Prd(Shape):
    coreFeatures: list[Feature] = Field(default_factory=list)


class Registry(Shape):
    features: list[Feature] = Field(default_factory=list)


def shape_issues(domain, value):
    model = {"db": Database, "api": Api, "prd": Prd, "registry": Registry}[domain]
    data = {"features": value} if domain == "registry" else value
    target = "prd" if domain == "registry" else domain
    try:
        model.model_validate(data)
    except ValidationError as exc:
        return [ContractIssue("ARTIFACT_SHAPE_INVALID", target, domain,
                              "/" + "/".join(map(str, error["loc"])), error["msg"]).to_dict()
                for error in exc.errors(include_url=False, include_input=False)]
    issues = []
    def schema(node, pointer):
        if isinstance(node, bool):
            return
        if not isinstance(node, dict):
            issues.append(ContractIssue("SCHEMA_SHAPE_INVALID", target, domain, pointer, "schema object required").to_dict())
            return
        props = node.get("properties", {})
        if not isinstance(props, dict):
            issues.append(ContractIssue("SCHEMA_SHAPE_INVALID", target, domain, pointer + "/properties", "properties object required").to_dict())
            return
        required = node.get("required", [])
        if not isinstance(required, list) or not all(isinstance(name, str) for name in required):
            issues.append(ContractIssue("SCHEMA_SHAPE_INVALID", target, domain, pointer + "/required", "string array required").to_dict())
        for name, child in props.items():
            schema(child, pointer + "/properties/" + str(name))
        if "items" in node:
            schema(node["items"], pointer + "/items")
        for key in ("oneOf", "allOf", "anyOf"):
            if key not in node:
                continue
            if not isinstance(node[key], list):
                issues.append(ContractIssue("SCHEMA_SHAPE_INVALID", target, domain, pointer + "/" + key, "schema array required").to_dict())
            else:
                for index, child in enumerate(node[key]):
                    schema(child, f"{pointer}/{key}/{index}")
    if domain == "api":
        for index, endpoint in enumerate(value.get("endpoints") or []):
            for key in ("requestSchema", "responseSchema"):
                if key in endpoint:
                    schema(endpoint[key], f"/endpoints/{index}/{key}")
        for name, definition in (value.get("schemas") or {}).items():
            schema(definition, f"/schemas/{name}")
    if domain in {"registry", "prd"}:
        features = value if domain == "registry" else value.get("coreFeatures") or []
        for index, feature in enumerate(features):
            for operation_index, operation in enumerate(feature.get("apiContract") or []):
                for key in ("requestSchema", "responseSchema"):
                    if key in operation:
                        schema(operation[key], f"/features/{index}/apiContract/{operation_index}/{key}")
    return issues
