import json
import logging
from dataclasses import dataclass

from phase2.json_utils import try_parse_json

logger = logging.getLogger(__name__)


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

        errors += self._check_json(db_schema, "DB 스키마")
        errors += self._check_json(api_spec, "API 스펙")

        if not feature_list:
            errors.append("featureList: 기능 목록이 비어있습니다")

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
