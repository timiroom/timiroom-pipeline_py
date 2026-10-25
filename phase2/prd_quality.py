"""Requirement-derived PRD policies and visible quality provenance."""

import copy
import json
import re


def _delivery_contracts(out):
    """Visible design defaults and event aliases, never claimed operational facts."""
    events = {}
    aliases = {alias: item["name"] for item in out.get("telemetryEvents") or [] for alias in item.get("aliases") or []}
    for item in out.get("kpi") or []:
        measurement = item.get("measurement") or {}
        event = measurement.get("event")
        if not isinstance(event, str) or not event:
            continue
        tokens = re.findall(r"[a-z0-9]+", event.lower())
        if not tokens:
            continue
        signature = tuple(sorted(token[:-1] if token.endswith("s") and not token.endswith(("ss", "us", "is")) else token for token in tokens))
        canonical = events.setdefault(signature, event)
        if canonical != event:
            aliases[event] = canonical
            measurement["event"] = canonical
    if events:
        out["telemetryEvents"] = [{"name": event, "aliases": [key for key, value in aliases.items() if value == event],
                                   "status": "planned", "source": "application instrumentation"} for event in events.values()]
    for item in out.get("kpi") or []:
        for key in ("measurementMethod", "measurement", "measurementDefinition"):
            def replace(value):
                if isinstance(value, dict):
                    return {k: replace(v) for k, v in value.items()}
                if isinstance(value, str):
                    for alias, canonical in aliases.items():
                        value = re.sub(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", canonical, value)
                return value
            if key in item:
                item[key] = replace(item[key])
    text = json.dumps({"overview": out.get("projectOverview"), "core": out.get("coreFeatures") or []}, ensure_ascii=False)
    if re.search(r"\bWEB\b|웹|브라우저|반응형", text, re.IGNORECASE):
        out.setdefault("uiAcceptanceCriteria", [
            "360px·768px·1280px 화면에서 가로 넘침 없이 목록·상세·입력 화면을 이용할 수 있다.",
            "조회·저장 중 로딩, 빈 목록, 입력 오류, 권한 거절, 재시도 상태를 화면에 구분해 표시한다.",
            "모바일 키보드·사진 선택·메시지 스크롤 중 입력과 주요 동작 버튼을 이용할 수 있다.",
            "본인만 수정 가능한 항목은 API가 제공하는 편집 가능 여부 또는 본인 전용 목록으로 제어한다.",
        ])
    file_texts = [str(value) for feature in out.get("coreFeatures") or []
                  for value in [feature.get("name", ""), feature.get("description", ""), *(feature.get("requirements") or [])]]
    if any(re.search(r"(?:사진|이미지|파일).{0,24}(?:첨부|등록|저장|업로드|교체)|\b(?:upload|attach)\b", value, re.IGNORECASE)
           and not re.search(r"제외|하지 않|책임은 없|미지원|\b(?:without|excluded)\b", value, re.IGNORECASE)
           for value in file_texts):
        out.setdefault("fileLifecyclePolicy", {
            "status": "design_default", "storage": "configured S3-compatible object storage adapter",
            "delivery": "storage_key를 공개 또는 서명 URL로 변환하며 CDN은 원본 저장소와 분리한다.",
            "validation": "서버가 실제 바이트의 MIME·크기·소유권을 검사하며 클라이언트 MIME 선언만 신뢰하지 않는다.",
            "execution": "60초 주기 worker가 임시 객체·논리 삭제 메타데이터를 대조한다. 작업 상태·재시도 횟수·다음 실행 시각은 영속 저장한다.",
            "retention": {"temporaryUploadHours": 24, "deletedFileDays": 7},
            "retry": {"maxAttempts": 5, "strategy": "exponential backoff", "idempotencyKey": "storage object key + deletion version"},
            "compensation": "DB 실패 시 업로드 파일 삭제를 보상하고 실패한 정리는 worker가 재처리한다.",
        })
        tech = out.setdefault("techStack", {})
        tech.setdefault("objectStorage", "환경 설정으로 선택하는 객체 저장소 어댑터에 원본 파일을 저장하고 CDN에는 표시용 URL을 연결합니다.")
        tech.setdefault("backgroundJobs", "파일 정리 worker와 영속 재시도 상태를 사용합니다. 메시지 큐 미사용이어도 주기 실행과 실패 복구 책임은 유지합니다.")
    if out.get("releaseSchedule"):
        out.setdefault("releasePlanningAssumptions", {
            "status": "unvalidated_estimate", "teamSize": None, "weeklyCapacityHours": None,
            "requiresCapacityEstimate": True,
            "notice": "일정은 설계 가설입니다. 팀 인원·주당 가용 시간·작업별 추정을 입력한 뒤 확정합니다.",
        })
    if aliases:
        quality = out.setdefault("qualityMetadata", {})
        quality.setdefault("adjustments", []).extend(f"TELEMETRY_EVENT_ALIAS:{key}->{value}" for key, value in aliases.items())


def reconcile_prd_quality(document):
    out = copy.deepcopy(document)
    _delivery_contracts(out)
    quality = dict(out.get("qualityMetadata") or {})
    changes = list(quality.get("adjustments") or [])
    core = out.get("coreFeatures") or []
    requirements = " ".join(str(item.get("description", "")) + " " + json.dumps(item.get("requirements") or [], ensure_ascii=False)
                            for item in core if isinstance(item, dict))
    ownership = re.search(r"사용자(?:별|에게만)|사용자 계정|본인|소유자|권한|user.?account|current user|ownership", requirements, re.IGNORECASE)
    if ownership:
        tech = out.setdefault("techStack", {})
        auth = str(tech.get("auth") or "")
        if not auth or any(token in auth for token in ("명시된 경우에만", "선택사항", "인증 없음", "선택 사항")):
            tech["auth"] = "사용자 식별과 인증을 적용하고 모든 개인 데이터 요청에서 소유권을 검증합니다. 타인의 데이터 조회·수정·삭제를 거부하며 미인증 401과 권한 부족 403을 구분합니다."
            changes.append("AUTH_POLICY_ALIGNED_WITH_OWNERSHIP")
    for item in out.get("kpi") or []:
        if not isinstance(item, dict):
            continue
        metric = str(item.get("metric") or "")
        day = re.search(r"(7|30)\s*일", metric)
        if not day or not any(term in metric.casefold() for term in ("재방문", "반복 이용", "retention", "리텐션")):
            continue
        measurement = item.get("measurement")
        if isinstance(measurement, dict):
            item["measurementDefinition"] = {
                "returnEvent": measurement.get("event"), "window": measurement.get("window"),
                "numerator": measurement.get("numerator"), "denominator": measurement.get("denominator"),
            }
            continue
        count = int(day.group(1))
        method = str(item.get("measurementMethod") or "")
        previous_definition = item.get("measurementDefinition") or {}
        cumulative = previous_definition.get("kind") == "cumulative_return" or bool(
            re.search(r"(?:7|30)\s*일\s*(?:이내|동안)|기간\s*내", method))
        definition = {
            "cohortEvent": "first_core_action_completed", "returnEvent": "core_action_completed",
            "windowStartDay": 1 if cumulative else count, "windowEndDayExclusive": count + 1,
            "timezone": "UTC", "denominator": "측정 종료일까지 관측 가능한 최초 핵심 업무 완료 사용자 수",
            "numerator": "해당 코호트 중 측정 구간에 핵심 업무를 1회 이상 다시 완료한 고유 사용자 수",
            "kind": "cumulative_return" if cumulative else "day_n_retention",
        }
        item["measurementDefinition"] = definition
        item["measurementMethod"] = (
            f"최초 핵심 업무 완료 코호트 중 관측 기간이 확보된 사용자를 분모로 하고, "
            f"UTC 기준 최초 완료일 이후 {definition['windowStartDay']}일차 00:00부터 {count + 1}일차 00:00 미만에 "
            "핵심 업무를 다시 완료한 고유 사용자를 분자로 집계한다."
        )
    feature_names = [str(item.get("name")) for item in core if isinstance(item, dict) and item.get("name")]
    warnings = list(quality.get("warnings") or [])
    for index, persona in enumerate(out.get("userPersonas") or []):
        if not isinstance(persona, dict) or not feature_names:
            continue
        if persona.get("segmentId") and persona.get("role"):
            continue
        text = " ".join(str(persona.get(key) or "") for key in ("goal", "painPoint", "usagePattern"))
        if any(name in text for name in feature_names):
            continue
        focus = feature_names[index % len(feature_names)]
        persona["usagePattern"] = f"사용자 요구에서 확인한 {focus} 흐름을 수행하고 결과와 후속 작업을 확인한다. 구체적인 이용 빈도는 사용자 조사로 검증한다."
        persona["assumption"] = True
        warnings.append(f"PERSONA_DOMAIN_CONTEXT_ENRICHED:{index}: 조사로 검증할 가설")
    quality.update({"adjustments": list(dict.fromkeys(changes)), "warnings": list(dict.fromkeys(warnings)),
                    "fallbacks": out.pop("_qualityFallbacks", quality.get("fallbacks", [])),
                    "itemDiagnostics": out.pop("_qualityDiagnostics", quality.get("itemDiagnostics", []))})
    out["qualityMetadata"] = quality
    return out
