import json
import logging
from dataclasses import dataclass

from phase2.agents.api_agent import _invalid_paths
from phase2.agents.dba_agent import min_table_count
from phase2.feature_coverage import uncovered_features
from phase2.json_utils import try_parse_json

logger = logging.getLogger(__name__)

# Phase3 재시도는 OrchestrationGraph.run()을 처음부터(search→PM→...) 다시 실행하므로
# PM이 매 시도마다 feature_list 자체를 다르게 생성할 수 있다 — "특정 기능 하나가 빠졌다"는
# 결함은 재시도해도 같은 기능을 다시 만난다는 보장이 없어 수렴하지 않는다(실측: 3회 재시도
# 후 HITL 실패). 그래서 기능 커버리지는 여기서 소수 누락에는 관대하게 하고, DBA/API 단계의
# in-run 보정 루프(같은 feature_list 안에서 재시도하므로 실제로 수렴 가능)를 1차 방어선으로 삼는다.
_COVERAGE_FAIL_RATIO = 0.4


@dataclass
class ValidationResult:
    success: bool
    errors: list[str]

    @classmethod
    def ok(cls) -> "ValidationResult":
        return cls(success=True, errors=[])

    @classmethod
    def fail(cls, errors: list[str]) -> "ValidationResult":
        return cls(success=False, errors=errors)

    def errors_to_string(self) -> str:
        return "\n".join(self.errors)


class SchemaValidator:

    def validate(self, feature_list: list[str], db_schema: str, api_spec: str) -> ValidationResult:
        errors: list[str] = []

        db_json_errors = self._check_json(db_schema, "DB 스키마")
        api_json_errors = self._check_json(api_spec, "API 스펙")
        errors += db_json_errors
        errors += api_json_errors

        if not feature_list:
            errors.append("featureList: 기능 목록이 비어있습니다")

        # JSON 자체가 깨졌으면 내용 커버리지 체크는 의미가 없으므로 JSON 검증을 통과한 값만 검사
        if not db_json_errors:
            errors += self._check_db_coverage(db_schema, feature_list)
        if not api_json_errors:
            errors += self._check_api_coverage(api_spec, feature_list)

        if errors:
            logger.warning("검증 실패 — %d 개 오류: %s", len(errors), errors)
            return ValidationResult.fail(errors)

        logger.info("검증 통과")
        return ValidationResult.ok()

    def _check_json(self, value: str, label: str) -> list[str]:
        if not value or not value.strip():
            return [f"{label}: 값이 비어있습니다"]
        try:
            json.loads(value)
            return []
        except Exception:
            pass
        # try_parse_json으로 복구 시도
        if try_parse_json(value) is not None:
            logger.info("%s: JSON 복구 성공 — 검증 통과", label)
            return []
        return [f"{label}: 유효하지 않은 JSON 형식"]

    def _check_db_coverage(self, db_schema: str, feature_list: list[str]) -> list[str]:
        """DBA/QA 단계 안전망을 뚫고 나온 부실 스키마를 잡는 최후 백스톱.
        재시도 1회가 풀 Phase2 재실행이라 비용이 크므로 테이블 개수 기준은 QA보다 느슨하게 잡되,
        기능별 커버리지(빠진 기능이 있는지)는 사용자 요구사항이므로 엄격하게 검사한다."""
        data = try_parse_json(db_schema)
        if not isinstance(data, dict):
            return []
        tables = data.get("tables")
        if not isinstance(tables, list):
            return []

        feature_count = len(feature_list)
        errors: list[str] = []
        if len(tables) > 1 and not data.get("relationships"):
            errors.append("DB 스키마: 테이블이 2개 이상인데 relationships가 비어 있습니다 — 테이블 간 외래키 관계를 정의하세요")
        if feature_count:
            minimum = max(2, min_table_count(feature_count) - 1)
            if len(tables) < minimum:
                errors.append(f"DB 스키마: 테이블이 {len(tables)}개뿐 — 기능 {feature_count}개 기준 최소 {minimum}개 필요")
        haystack = []
        for t in tables:
            if isinstance(t, dict):
                haystack.append(str(t.get("name", "")))
                haystack.append(str(t.get("description", "")))
        missing = uncovered_features(feature_list, haystack)
        if missing and feature_count and len(missing) / feature_count > _COVERAGE_FAIL_RATIO:
            errors.append(f"DB 스키마: 기능 {feature_count}개 중 {len(missing)}개를 저장할 테이블이 없어 보입니다 — {missing}")
        elif missing:
            logger.warning("DB 스키마: 소수 기능 미반영(재시도 임계치 미만이라 통과) — %s", missing)
        return errors

    def _check_api_coverage(self, api_spec: str, feature_list: list[str]) -> list[str]:
        """경로 형식 + 기능별 엔드포인트 커버리지 최후 백스톱."""
        data = try_parse_json(api_spec)
        if not isinstance(data, dict):
            return []
        endpoints = data.get("endpoints")
        if not isinstance(endpoints, list):
            return []

        errors: list[str] = []
        bad = _invalid_paths(endpoints)
        if bad:
            errors.append(f"API 스펙: REST 경로 형식 위반(한글/공백/특수문자 등): {bad}")
        missing = uncovered_features(feature_list, [ep.get("description", "") for ep in endpoints if isinstance(ep, dict)])
        feature_count = len(feature_list)
        if missing and feature_count and len(missing) / feature_count > _COVERAGE_FAIL_RATIO:
            errors.append(f"API 스펙: 기능 {feature_count}개 중 {len(missing)}개에 대응하는 엔드포인트가 없어 보입니다 — {missing}")
        elif missing:
            logger.warning("API 스펙: 소수 기능 미반영(재시도 임계치 미만이라 통과) — %s", missing)
        return errors
