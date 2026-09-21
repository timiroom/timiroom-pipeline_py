import json
import logging
import re
from dataclasses import dataclass, field

from phase2.agents.api_agent import _invalid_paths
from phase2.feature_coverage import strictly_uncovered_features, uncovered_features
from phase2.feature_registry import missing_api_contract_features, missing_db_contract_features
from phase2.feature_scope import backend_features
from phase2.json_utils import try_parse_json

logger = logging.getLogger(__name__)

_VALID_HTTP_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}
_REQUIRED_PRD_FIELDS = {
    "projectOverview", "background", "goals", "kpi", "userPersonas",
    "mvpScope", "techStack", "releaseSchedule", "coreFeatures",
}
_CARDINALITY = r"(?:0\.\.[01NM]|1\.\.[1NM]|[01NM])"
_RELATION_CARDINALITY_RE = re.compile(rf"\(\s*{_CARDINALITY}\s*:\s*{_CARDINALITY}\s*\)", re.IGNORECASE)
_NON_RESOURCE_PATHS = {
    "auth", "login", "logout", "signup", "register", "token", "refresh",
    "health", "search", "metrics", "status", "me",
}


@dataclass
class ValidationResult:
    success: bool
    errors: list[str]
    error_codes: list[str] = field(default_factory=list)
    repair_targets: list[str] = field(default_factory=list)
    normalized_db_schema: str | None = None
    normalized_api_spec: str | None = None
    normalized_prd_document: str | None = None

    def errors_to_string(self) -> str:
        return "\n".join(self.errors)


class SchemaValidator:
    def validate(
        self,
        feature_list: list[str],
        db_schema: str,
        api_spec: str,
        prd_document: str = "",
        feature_registry: list[dict] | None = None,
    ) -> ValidationResult:
        errors: list[str] = []
        codes: list[str] = []
        targets: set[str] = set()

        def add(code: str, target: str, message: str) -> None:
            if code not in codes:
                codes.append(code)
            if message not in errors:
                errors.append(message)
            targets.add(target)

        db, normalized_db = self._parse_object(db_schema, "DB 스키마", "DB_JSON", "db", add)
        api, normalized_api = self._parse_object(api_spec, "API 스펙", "API_JSON", "api", add)
        prd, normalized_prd = self._parse_object(prd_document, "PRD 문서", "PRD_JSON", "prd", add)

        raw_features = feature_list or []
        cleaned_features = [value.strip() for value in raw_features if isinstance(value, str) and value.strip()]
        if not cleaned_features:
            add("FEATURES_EMPTY", "pm", "featureList: 기능 목록이 비어있습니다")
        elif len(cleaned_features) != len(raw_features):
            add("FEATURES_INVALID", "pm", "featureList: 빈 문자열 또는 문자열이 아닌 기능이 있습니다")
        elif len(set(cleaned_features)) != len(cleaned_features):
            add("FEATURES_DUPLICATED", "pm", "featureList: 중복 기능이 있습니다")

        if db is not None:
            self._check_db(db, cleaned_features, add, feature_registry)
        if api is not None:
            self._check_api(api, cleaned_features, add, feature_registry)
        if prd is not None:
            self._check_prd(prd, cleaned_features, add)
        if db is not None and api is not None:
            self._check_cross_artifact(db, api, add)
        if db is not None and prd is not None:
            self._check_prd_db_entities(prd, db, add)

        if errors:
            logger.warning("검증 실패 — %d개 오류: %s", len(errors), errors)
        else:
            logger.info("검증 통과")
        return ValidationResult(
            success=not errors,
            errors=errors,
            error_codes=codes,
            repair_targets=sorted(targets),
            normalized_db_schema=normalized_db,
            normalized_api_spec=normalized_api,
            normalized_prd_document=normalized_prd,
        )

    @staticmethod
    def _parse_object(value: str, label: str, code: str, target: str, add):
        if not value or not value.strip():
            add(code, target, f"{label}: 값이 비어있습니다")
            return None, None
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = try_parse_json(value)
        if not isinstance(parsed, dict):
            add(code, target, f"{label}: 최상위 JSON 객체가 아니거나 복구할 수 없습니다")
            return None, None
        return parsed, json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _check_db(data: dict, feature_list: list[str], add, feature_registry=None) -> None:
        tables = data.get("tables")
        if not isinstance(tables, list) or not tables:
            add("DB_TABLES_REQUIRED", "db", "DB 스키마: tables는 비어있지 않은 배열이어야 합니다")
            return

        table_names: list[str] = []
        for index, table in enumerate(tables):
            if not isinstance(table, dict):
                add("DB_TABLE_INVALID", "db", f"DB 스키마: tables[{index}]가 객체가 아닙니다")
                continue
            name = table.get("name")
            if not isinstance(name, str) or not name.strip():
                add("DB_TABLE_NAME_REQUIRED", "db", f"DB 스키마: tables[{index}].name이 비어있습니다")
                continue
            table_names.append(name)
            columns = table.get("columns")
            if not isinstance(columns, list) or not columns:
                add("DB_COLUMNS_REQUIRED", "db", f"DB 스키마: {name}.columns가 비어있습니다")
                continue
            column_names: list[str] = []
            has_pk = False
            for col_index, column in enumerate(columns):
                if not isinstance(column, dict):
                    add("DB_COLUMN_INVALID", "db", f"DB 스키마: {name}.columns[{col_index}]가 객체가 아닙니다")
                    continue
                col_name = column.get("name")
                col_type = column.get("type")
                if not isinstance(col_name, str) or not col_name.strip() or not isinstance(col_type, str) or not col_type.strip():
                    add("DB_COLUMN_FIELDS_REQUIRED", "db", f"DB 스키마: {name}의 컬럼 name/type이 비어있습니다")
                    continue
                column_names.append(col_name)
                has_pk = has_pk or "PRIMARY_KEY" in str(column.get("constraints", "")).upper()
            if len(column_names) != len(set(column_names)):
                add("DB_COLUMN_DUPLICATED", "db", f"DB 스키마: {name}에 중복 컬럼이 있습니다")
            if not has_pk:
                add("DB_PRIMARY_KEY_REQUIRED", "db", f"DB 스키마: {name}에 PRIMARY_KEY가 없습니다")

        if len(table_names) != len(set(table_names)):
            add("DB_TABLE_DUPLICATED", "db", "DB 스키마: 중복 테이블명이 있습니다")

        if feature_registry:
            missing = missing_db_contract_features(feature_registry, tables)
        else:
            # Backward-compatible fallback for callers that predate Feature Registry.
            backend_feature_list = backend_features(feature_list)
            table_texts = []
            for table in tables:
                if isinstance(table, dict):
                    table_texts.extend([
                        str(table.get("name", "")),
                        str(table.get("description", "")),
                        " ".join(str(column.get("name", "")) for column in table.get("columns", []) if isinstance(column, dict)),
                    ])
            missing = uncovered_features(backend_feature_list, table_texts)
        if missing:
            add("DB_FEATURE_COVERAGE", "db", f"DB 스키마: 기능을 저장할 테이블이 없어 보입니다 — {missing}")

        relationships = data.get("relationships")
        if not isinstance(relationships, list):
            add("DB_RELATIONSHIPS_REQUIRED", "db", "DB 스키마: relationships는 배열이어야 합니다")
        elif len(tables) > 1 and not relationships:
            add("DB_RELATIONSHIPS_EMPTY", "db", "DB 스키마: 테이블이 2개 이상인데 relationships가 비어 있습니다")
        else:
            normalized_relationships = [re.sub(r"\s+", " ", str(value)).strip().casefold() for value in relationships]
            if len(normalized_relationships) != len(set(normalized_relationships)):
                add("DB_RELATIONSHIP_DUPLICATED", "db", "DB 스키마: 중복 relationship이 있습니다")
            known = set(table_names)
            for relationship in relationships:
                text = str(relationship)
                referenced = [
                    name
                    for name in known
                    for _ in re.finditer(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text)
                ]
                if not referenced or not _RELATION_CARDINALITY_RE.search(text):
                    add("DB_RELATIONSHIP_INVALID", "db", f"DB 스키마: 잘못된 relationship: {text}")

    @staticmethod
    def _check_api(data: dict, feature_list: list[str], add, feature_registry=None) -> None:
        endpoints = data.get("endpoints")
        if not isinstance(endpoints, list) or not endpoints:
            add("API_ENDPOINTS_REQUIRED", "api", "API 스펙: endpoints는 비어있지 않은 배열이어야 합니다")
            return

        keys: list[tuple[str, str]] = []
        for index, endpoint in enumerate(endpoints):
            if not isinstance(endpoint, dict):
                add("API_ENDPOINT_INVALID", "api", f"API 스펙: endpoints[{index}]가 객체가 아닙니다")
                continue
            method = str(endpoint.get("method", "")).upper()
            path = endpoint.get("path")
            if method not in _VALID_HTTP_METHODS:
                add("API_METHOD_INVALID", "api", f"API 스펙: 유효하지 않은 HTTP method: {method or '(없음)'}")
            if not isinstance(path, str) or not path:
                add("API_PATH_REQUIRED", "api", f"API 스펙: endpoints[{index}].path가 비어있습니다")
            else:
                keys.append((method, path))
            for field_name in ("description", "successResponse", "errorCodes"):
                if not str(endpoint.get(field_name, "")).strip():
                    add("API_FIELDS_REQUIRED", "api", f"API 스펙: {method} {path}의 {field_name}이 비어있습니다")

        if len(keys) != len(set(keys)):
            add("API_ENDPOINT_DUPLICATED", "api", "API 스펙: method/path가 중복된 엔드포인트가 있습니다")
        bad_paths = _invalid_paths([endpoint for endpoint in endpoints if isinstance(endpoint, dict)])
        if bad_paths:
            add("API_PATH_INVALID", "api", f"API 스펙: REST 경로 형식 위반: {bad_paths}")
        if feature_registry:
            registry_ids = {
                str(item.get("featureId") or item.get("id") or "").strip()
                for item in feature_registry if isinstance(item, dict)
            }
            unmapped = [
                f"{endpoint.get('method', 'GET')} {endpoint.get('path', '')}"
                for endpoint in endpoints
                if isinstance(endpoint, dict)
                and str(endpoint.get("featureId") or "").strip() not in registry_ids
            ]
            if unmapped:
                add("API_FEATURE_ID_REQUIRED", "api", f"API 스펙: Registry featureId 매핑 누락 — {unmapped}")
            missing_contracts = missing_api_contract_features(feature_registry, endpoints)
            if missing_contracts:
                add("API_CONTRACT_MISSING", "api", f"API 스펙: apiContract endpoint 누락 — {missing_contracts}")
        else:
            missing = strictly_uncovered_features(
                backend_features(feature_list),
                [str(endpoint.get("description", "")) for endpoint in endpoints if isinstance(endpoint, dict)],
            )
            if missing:
                add("API_FEATURE_COVERAGE", "api", f"API 스펙: 기능에 대응하는 endpoint가 없어 보입니다 — {missing}")
    @staticmethod
    def _check_prd(data: dict, feature_list: list[str], add) -> None:
        missing_fields = sorted(_REQUIRED_PRD_FIELDS - set(data))
        if missing_fields:
            add("PRD_FIELDS_REQUIRED", "prd", f"PRD 문서: 필수 섹션 누락 — {missing_fields}")
        core_features = data.get("coreFeatures")
        if not isinstance(core_features, list) or not core_features:
            add("PRD_CORE_FEATURES_REQUIRED", "prd", "PRD 문서: coreFeatures가 비어있습니다")
            return
        core_texts = []
        for feature in core_features:
            if isinstance(feature, dict):
                core_texts.extend([
                    str(feature.get("name", "")),
                    str(feature.get("description", "")),
                    " ".join(str(item) for item in feature.get("requirements", []) if item is not None),
                ])
            else:
                core_texts.append(str(feature))
        missing = strictly_uncovered_features(feature_list, core_texts)
        if missing:
            add("PRD_FEATURE_COVERAGE", "prd", f"PRD 문서: coreFeatures에 반영되지 않은 기능이 있습니다 — {missing}")


    @staticmethod
    def _check_prd_db_entities(prd: dict, db: dict, add) -> None:
        """PRD의 구조화된 database 엔티티가 DBA 테이블에 존재하는지 확인한다."""
        declared = set(re.findall(r"\b([a-z][a-z0-9_]*)\s*\(id\s+PK\b", json.dumps(prd, ensure_ascii=False)))
        actual = {
            str(table.get("name")) for table in db.get("tables", [])
            if isinstance(table, dict) and table.get("name")
        }
        missing = sorted(declared - actual)
        if missing:
            add("PRD_DB_ENTITY_MISMATCH", "prd", f"PRD/DB 정합성: PRD에 정의된 테이블이 DBA 스키마에 없습니다 — {missing}")

    @staticmethod
    def _check_cross_artifact(db: dict, api: dict, add) -> None:
        endpoints = api.get("endpoints") if isinstance(api.get("endpoints"), list) else []
        authentication = str(api.get("authentication", "")).strip()
        if any(endpoint.get("authRequired") is True for endpoint in endpoints if isinstance(endpoint, dict)) and not authentication:
            add("API_AUTH_DEFINITION_REQUIRED", "api", "API 스펙: 인증 필요 엔드포인트가 있지만 authentication 정의가 없습니다")

        table_names = {
            str(table.get("name")) for table in db.get("tables", [])
            if isinstance(table, dict) and table.get("name")
        }
        for table in db.get("tables", []):
            if not isinstance(table, dict):
                continue
            for column in table.get("columns", []):
                if not isinstance(column, dict):
                    continue
                name = str(column.get("name", ""))
                constraints = str(column.get("constraints", "")).upper()
                if not name.endswith("_id") or "FOREIGN_KEY" not in constraints:
                    continue
                # 구조화된 FK가 명시되어 있으면 컬럼명 추론보다 REFERENCES를 기준으로
                # 검증한다. recorded_by_id, actor_id, job_id처럼 이름만으로 대상을
                # 결정할 수 없는 감사/폴리모픽 컬럼을 오탐하지 않기 위한 규칙이다.
                referenced = re.search(r"REFERENCES\s+([A-Z_][A-Z0-9_]*)\s*\(", constraints)
                if referenced:
                    if referenced.group(1).lower() in {name.lower() for name in table_names}:
                        continue
                    add(
                        "DB_FOREIGN_KEY_TARGET_MISSING",
                        "db",
                        f"DB 스키마: {name}이 참조할 테이블을 찾을 수 없습니다",
                    )
                    continue
                candidates = SchemaValidator._fk_target_candidates(name[:-3])
                stem = name[:-3]
                for table_name in table_names:
                    tail = table_name.rsplit("_", 1)[-1]
                    variants = {tail}
                    if tail.endswith("ies"):
                        variants.add(tail[:-3] + "y")
                    elif tail.endswith("sses"):
                        variants.add(tail[:-2])
                    elif tail.endswith("s"):
                        variants.add(tail[:-1])
                    else:
                        variants.add(tail + "s")
                    if stem in variants:
                        candidates.add(table_name)
                if not candidates & table_names:
                    add("DB_FOREIGN_KEY_TARGET_MISSING", "db", f"DB 스키마: {name}이 참조할 테이블을 찾을 수 없습니다")

    @staticmethod
    def _fk_target_candidates(stem: str) -> set[str]:
        parts = [part for part in stem.split("_") if part]
        tail = parts[-1] if parts else stem
        candidates = {stem, f"{stem}s", f"{stem}es", stem.removesuffix("y") + "ies"}
        candidates.update({tail, f"{tail}s", f"{tail}es", tail.removesuffix("y") + "ies"})
        if tail in {"user", "member", "owner", "assignee", "reviewer", "signer", "uploader"}:
            candidates.update({"user", "users", "freelancer_profiles"})
        if tail in {"file", "document", "pdf"}:
            candidates.update({"file", "files", "documents"})
        return candidates
