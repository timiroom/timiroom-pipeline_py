import json
import logging
import re
from dataclasses import dataclass, field

from phase2.agents.api_agent import _invalid_paths
from phase2.agents.dba_agent import min_table_count
from phase2.feature_coverage import uncovered_features
from phase2.json_utils import try_parse_json

logger = logging.getLogger(__name__)

_VALID_HTTP_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}
_REQUIRED_PRD_FIELDS = {
    "projectOverview", "background", "goals", "kpi", "userPersonas",
    "mvpScope", "techStack", "releaseSchedule", "coreFeatures",
}
_RELATION_CARDINALITY_RE = re.compile(r"\(\s*(?:1|N|M)\s*:\s*(?:1|N|M)\s*\)", re.IGNORECASE)
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
            self._check_db(db, cleaned_features, add)
        if api is not None:
            self._check_api(api, cleaned_features, add)
        if prd is not None:
            self._check_prd(prd, cleaned_features, add)
        if db is not None and api is not None:
            self._check_cross_artifact(db, api, add)

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
    def _check_db(data: dict, feature_list: list[str], add) -> None:
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

        relationships = data.get("relationships")
        if not isinstance(relationships, list):
            add("DB_RELATIONSHIPS_REQUIRED", "db", "DB 스키마: relationships는 배열이어야 합니다")
        elif len(tables) > 1 and not relationships:
            add("DB_RELATIONSHIPS_EMPTY", "db", "DB 스키마: 테이블이 2개 이상인데 relationships가 비어 있습니다")
        else:
            known = set(table_names)
            for relationship in relationships:
                text = str(relationship)
                referenced = [name for name in known if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text)]
                if len(referenced) < 2 or not _RELATION_CARDINALITY_RE.search(text):
                    add("DB_RELATIONSHIP_INVALID", "db", f"DB 스키마: 잘못된 relationship: {text}")

        minimum = max(2, min_table_count(len(feature_list)) - 1) if feature_list else 1
        if len(tables) < minimum:
            add("DB_TABLE_COUNT_LOW", "db", f"DB 스키마: 테이블이 {len(tables)}개뿐 — 최소 {minimum}개 필요")

        haystack = []
        for table in tables:
            if isinstance(table, dict):
                haystack.extend([str(table.get("name", "")), str(table.get("description", ""))])
        missing = uncovered_features(feature_list, haystack)
        if missing:
            add("DB_FEATURE_COVERAGE", "db", f"DB 스키마: 기능을 저장할 테이블이 없습니다 — {missing}")

    @staticmethod
    def _check_api(data: dict, feature_list: list[str], add) -> None:
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
        missing = uncovered_features(
            feature_list,
            [endpoint.get("description", "") for endpoint in endpoints if isinstance(endpoint, dict)],
        )
        if missing:
            add("API_FEATURE_COVERAGE", "api", f"API 스펙: 대응하는 엔드포인트가 없는 기능이 있습니다 — {missing}")

    @staticmethod
    def _check_prd(data: dict, feature_list: list[str], add) -> None:
        missing_fields = sorted(_REQUIRED_PRD_FIELDS - set(data))
        if missing_fields:
            add("PRD_FIELDS_REQUIRED", "prd", f"PRD 문서: 필수 섹션 누락 — {missing_fields}")
        core_features = data.get("coreFeatures")
        if not isinstance(core_features, list) or not core_features:
            add("PRD_CORE_FEATURES_REQUIRED", "prd", "PRD 문서: coreFeatures가 비어있습니다")
            return
        names = [str(item.get("name", "")) for item in core_features if isinstance(item, dict)]
        missing = uncovered_features(feature_list, names)
        if missing:
            add("PRD_FEATURE_COVERAGE", "prd", f"PRD 문서: coreFeatures에 누락된 기능이 있습니다 — {missing}")

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
        table_aliases = set(table_names)
        for table_name in table_names:
            table_aliases.add(table_name.removesuffix("s"))
            if table_name.endswith("ies"):
                table_aliases.add(table_name[:-3] + "y")

        for endpoint in endpoints:
            if not isinstance(endpoint, dict):
                continue
            segments = [segment for segment in str(endpoint.get("path", "")).split("/") if segment]
            resource_segments = [
                segment for segment in segments
                if segment not in {"api", "v1", "v2", "v3"} and not segment.startswith("{")
            ]
            if not resource_segments:
                continue
            resource = resource_segments[0].replace("-", "_").lower()
            if resource in _NON_RESOURCE_PATHS:
                continue
            resource_aliases = {resource, resource.removesuffix("s")}
            if resource.endswith("ies"):
                resource_aliases.add(resource[:-3] + "y")
            resource_matches_table = bool(resource_aliases & table_aliases) or any(
                table_alias.startswith(f"{resource_alias}_")
                or resource_alias.startswith(f"{table_alias}_")
                for resource_alias in resource_aliases
                for table_alias in table_aliases
            )
            if not resource_matches_table:
                message = f"DB/API 정합성: {endpoint.get('path')} 리소스에 대응하는 DB 테이블이 없습니다"
                add("API_DB_RESOURCE_MISMATCH", "db", message)
                add("API_DB_RESOURCE_MISMATCH", "api", message)

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
                stem = name[:-3]
                candidates = {stem, f"{stem}s", f"{stem}es", stem.removesuffix("y") + "ies"}
                if not candidates & table_names:
                    add("DB_FOREIGN_KEY_TARGET_MISSING", "db", f"DB 스키마: {name}이 참조할 테이블을 찾을 수 없습니다")
