# 추가 수정사항 1~9 적용 내역

## 범위

기존 우선사항 5개에 이어 API·DB 계약 검사, QA/Phase3 판정, repair 예산, 적용 범위, 입력 타입, MVP 정책을 보완했다. 이번에는 실제 LLM 호출, 라이브 E2E, PostgreSQL DDL 실행, Kafka 실제 저장 테스트를 수행하지 않았다.

## 1. API 필드 매핑

이전에는 requestSchema가 없는 GET이 일찍 반환되어 저장 대상 검사를 건너뛰었다. 이제 body/request, path, query, header, cookie, response, auth, server, derived를 구분하고 본문 유무와 무관하게 대상을 검사한다. 중첩 필드와 배열 경로, components/schema 참조, JSON pointer를 확인하고, DTO와 저장 타입이 다르면 명시적 transform이 있는지 확인한다. 생성 API의 필수 저장값 검사도 본문 부재 때문에 생략하지 않는다.

파일: `phase2/operation_contract.py`.

## 2. FK 정밀 검증

단순 테이블 존재 검사에서 참조 컬럼, PK/UNIQUE, 타입, 복합키 개수·대응, ON DELETE, nullable/default 충돌 검사로 확장했다. 컬럼의 REFERENCES 문자열과 테이블의 구조화 foreignKeys를 지원한다. 실제 UNIQUE 키가 아닌 partial unique index는 참조 키로 인정하지 않는다.

DBA의 타입 보정은 명시된 참조 컬럼을 따른다. 예를 들어 users(email)을 users(id)로 바꾸지 않는다. 대상이 없는 컬럼을 삭제하여 오류를 감추지 않고 보존한다. 명시적으로 선언된 양방향 FK는 무조건 삭제하거나 구조 오류로 단정하지 않으며, 순환 관계는 검토 경고로 구분한다.

생성 명세가 사용하는 일반적인 SQL 제약 표기와 구조화 계약을 검사하는 로직이며, 임의의 PostgreSQL DDL 전체를 파싱·실행하는 대체품은 아니다.

파일: `phase2/fk_validation.py`, `phase2/agents/dba_agent.py`, `phase3/schema_validator.py`.

## 3. API 파라미터 보존

파라미터 식별자를 이름 단독에서 `(in, name)`으로 바꿨다. 같은 이름의 path/query가 함께 남고 path는 required=true를 유지한다. schema의 enum/format/items/제한값과 style/explode 등의 직렬화 정보가 OpenAPI 변환까지 전달된다. header 이름은 대소문자 차이로 중복되지 않게 처리한다. content 파라미터는 schema와 중복 출력하지 않는다.

파일: `phase2/agents/api_agent.py`, `phase2/openapi_spec.py`.

## 4. 공유 endpoint 계약 충돌

method와 canonical route 기준으로 Operation Registry를 구성한다. 여러 featureId가 한 operation을 사용해도 하나의 계약을 투영한다. 서로 겹치지 않는 상세는 병합하고 request/response schema, 상태 코드, contentType 등의 값이 충돌하면 구조화된 오류로 기록한다. 마지막 feature의 계약을 자동 정답으로 덮어쓰지 않는다.

충돌 검출이 의미 있는 schema를 자동으로 선택해 준다는 뜻은 아니다. 서로 모순되는 요구사항은 upstream 계약 수정이나 검토가 필요하다.

파일: `phase2/operation_contract.py`, `phase2/agents/api_agent.py`.

## 5. QA·Phase3 통합

일반 Registry 파이프라인의 QA와 Phase3가 동일한 SchemaValidator를 사용한다. Issue에 code/domain/featureId/artifactKey/pointer/severity/repairable/revision을 기록하고 대상 식별자로 중복을 제거한다. 같은 endpoint의 서로 다른 잘못된 fieldMappings는 별도 결함으로 유지한다.

현재 산출물의 revision으로 다시 검사하므로 이전 qa_approved=false나 과거 blocker만으로 정상 결과를 막지 않는다. 생성 실패 blocker는 별도로 유지한다. 배열 응답은 responseSchema를 우선하고, GET에 requestBody 문자열을 요구하지 않는다. 검증기가 endpoint의 featureId를 몰래 수정하지 않는다.

Registry가 없는 외부·구형 수동 QA 호출은 기존 결함 전달을 위한 호환 경로를 유지한다. 이를 일반 Registry 경로와 같은 revision 보장이 있는 것으로 취급하지 않는다.

파일: `phase2/validation_contract.py`, `phase2/agents/qa_agent.py`, `phase3/schema_validator.py`, `phase3/validation_service.py`, `phase2/state.py`.

## 6. repair 예산과 반복 감지

실제 파이프라인의 예산 집계 책임을 OrchestrationGraph로 모아 Phase2/Phase3가 중복 차감하지 않게 했다. 도메인별 최대 1회는 유지한다. applied/noop/failed 및 호출 후 거절은 실제 시도에 포함하고, 호출 전 scope 거절·baseline 부재·미지원은 별도 상태로 처리한다. 병렬 결과 병합은 카운터를 낮은 이전 값으로 덮지 않는다.

한 도메인의 예산 소진이 다른 미사용 도메인의 수정을 전부 차단하지 않는다. PM/PRD 계약 수정이 필요하면 먼저 처리한다. 반복 fingerprint는 구조화된 issue 식별자를 정렬하여 계산하므로 문구·순서 변경을 새로운 결함으로 오인하지 않는다. 오래된 거절 상태가 새 호출의 예산을 숨기지 않도록 dispatch 상태도 분리한다.

파일: `phase2/repair_budget.py`, `phase2/orchestration_graph.py`, `phase3/retry_service.py`.

## 7. 기능별 산출물 적용 범위

applicability의 ui/api/db boolean을 우선 사용한다. UI-only 기능은 가짜 endpoint/table을 만들지 않고 기능 명세에 남는다. 무상태·외부 조회는 API가 필요해도 DB를 강제하지 않는다. 생성 대상과 검증 대상에 같은 기준을 사용한다. db=false인데 저장 테이블 계약이 있는 등 모순된 범위는 명시적 결함으로 처리한다.

DB coverage는 featureId가 한 테이블에 붙었다고 끝내지 않고 필수 계약 테이블 전체의 존재를 확인한다.

파일: `phase2/feature_scope.py`, `phase2/contract_guard.py`, `phase2/feature_registry.py`, API/DBA Agent, 공통 검사기.

## 8. 입력 타입 경계

기존 Pydantic 의존성을 사용해 DB/API/PRD/Registry의 중첩 컨테이너를 검증한다. tables=null, columns=null, 잘못된 parameters/sessionPolicy/참조 객체, 잘못된 JSON Schema properties/required 등이 Python TypeError 대신 위치가 있는 검증 오류로 반환된다. Feature Spec 발행 및 자체 검사/최종 검사에서 경계를 적용한다.

이는 현재 명세 계약의 입력 shape 검증이며 모든 JSON Schema 키워드의 의미 검증을 구현했다는 뜻은 아니다.

파일: `phase2/artifact_shapes.py`, Feature Spec/API/DBA Agent, `phase3/schema_validator.py`.

## 9. MVP·우선순위 정책

PRD와 Feature Spec이 같은 정책을 소비한다. 핵심 MUST=P0, supporting MUST=P1, SHOULD=P1, COULD=P2로 맞추고 명시적 P3도 보존한다. WONT/제외 범위는 releaseScope=excluded로 분리하여 P2와 혼동하지 않는다. Phase1의 명시적 포함 결정이 모델의 임의 제외보다 우선한다.

P0는 최대 5개, 활성 MVP는 최대 12개다. 작은 서비스에서 최소 P0 개수를 맞추기 위해 기능을 억지로 승격하지 않는다. 부모·선행 기능의 의존성 집합을 함께 선택하고, 없는 기능 참조와 순환 의존성은 명시적 오류로 기록한다. 최종 활성 목록을 feature_list/Registry/Feature Spec 및 PRD core 범위에 반영한다.

파일: `phase2/mvp_policy.py`, `phase2/agents/prd_agent.py`, `phase2/agents/feature_spec_agent.py`, `phase3/schema_validator.py`.

## 회귀 검증

- 전체 로컬 테스트: `562 passed`.
- CI 대상 및 신규 공통 모듈 Ruff 검사 통과, 수정 Agent의 import/미사용 코드 검사 통과.
- 라이브 E2E 및 실제 DB/Kafka 저장 검증은 미실행.

`tests/test_additional_contract_guards.py`에 API 매핑, 파라미터 보존, 공유 operation 충돌, FK 상세 제약, 입력 shape, UI-only·stateless API, QA/Phase3 revision, 예산·fingerprint, MVP 의존 관계 테스트를 추가했다. 기존 테스트 중 FK를 삭제하거나 독립된 우선순위 정책을 기대하던 항목은 새 정책에 맞게 정리했다.

실제 생성 품질, 지연 시간, Kafka consumer 저장 완료 여부는 다음 라이브 E2E에서 별도로 확인해야 한다.
