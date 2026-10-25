# Agent 프롬프트 점검 (2026-10-23)

## 점검 방법

Search, PM, PRD, Feature Spec, DBA, API, QA의 프롬프트뿐 아니라 실제 호출 경로와 parser/validator 요구 형식을 함께 확인했다. 선언만 남은 legacy 프롬프트를 현재 라이브 실패 원인으로 단정하지 않았다.

아래 본문은 최초 점검 시점의 문제와 수정 방향이다. 이후 실제 적용한 변경과 검증 범위는 마지막의 적용 기록에 구분했다.

## P1: PM 역할 설명과 실제 실행 순서의 불일치

- `phase2/agents/pm_agent.py:202`의 PM_PROMPT는 projectPlan이 이후 PRD의 기준이라고 설명한다.
- 실제 `phase2/orchestration_graph.py:86` 이후 순서는 Search → PRD → PM → Feature Spec이다.
- `phase2/agents/prd_agent.py:815`는 projectPlan을 입력에 넣지만 최초 PRD 호출 시점에는 PM이 아직 실행되지 않았다.
- PM에게 기능 계약을 만들지 말라고 하면서, 같은 프롬프트에서 기능별 PK/FK/컬럼/HTTP method/path 매핑을 요구한다.

영향: PRD가 새 PM 계획을 반영했다고 가정할 수 없고, PM의 자연어 설계 지시와 Feature Spec의 실제 계약이 서로 다른 설계를 제안할 수 있다.

수정 방향: PM을 사전 계획자로 유지한다면 PM → PRD 순서를 사용한다. 현재 순서를 유지한다면 PM을 PRD 후속 조정자로 명확히 정의하고 PRD 선행 계획이라는 문구를 제거한다. PM의 instruction은 원칙과 범위에 한정하고 물리 계약은 Feature Spec 한 곳에서 소유해야 한다.

## P1: DBA가 계약과 다른 타입·키·개수를 강제

- `phase2/agents/dba_agent.py:2321`의 worker는 Registry 필드 타입을 그대로 반영하라고 하면서 허용 타입을 제한하고 BIGINT AUTO_INCREMENT id를 일률적으로 요구한다. UUID/JSONB/TIMESTAMPTZ 등 선언 계약과 충돌할 수 있다.
- `*_id`는 모두 실제 테이블의 id를 참조하라고 한 뒤, 외부 ID와 polymorphic ID는 예외라고 한다. 비-id 고유키 참조도 일반 규칙에 가려진다.
- manager는 Registry 이름을 보존하라고 하면서 무조건 복수형 이름과 최소 테이블 개수도 요구한다.
- worker 인덱스 예시는 `INDEX ... ON ...`인데 `_repair_indexes`가 허용하는 완전한 문법은 `CREATE INDEX ... ON ...`이다. 라이브에서 이 예시 형식의 인덱스가 제거되는 것을 확인했다.

수정 방향: 명시 계약이 있을 때는 타입·키·이름·개수 휴리스틱을 적용하지 않는다. 비-id/복합 FK도 그대로 소비하고, 추가 테이블·필수 컬럼은 상위 계약 변경 없이 만들지 않는다. 인덱스 예시는 validator가 받는 문법과 일치시킨다.

이번 작업에서 최초 생성 manager/worker의 계약 밖 테이블·컬럼 유입은 코드로 제한했다. 그러나 프롬프트의 모순 자체는 별도 정리가 필요하다.

## P1: API의 계약 보존과 개수 상한 지시 충돌

- `phase2/agents/api_agent.py:108`의 plan은 모든 apiContract를 1:1로 보존하라고 요구한다.
- 같은 plan은 max_endpoints를 넘으면 낮은 우선순위 endpoint를 합치거나 제외하라고 요구한다.
- 라이브 로그에서도 계약상 필요한 수보다 soft cap이 작은 계획이 관찰됐다. 후처리가 계약을 다시 넣더라도 앞선 LLM 계획은 이미 서로 모순된 지시를 받는다.

수정 방향: 명시된 계약 경로 수를 보존의 기준으로 삼고, cap은 임의 파생 endpoint에만 적용한다. 한 계약을 제외하는 것은 해당 agent의 독자 결정이 아니어야 한다.

## P1: API의 DTO/필드 출처 설명이 실제 validator와 불일치

- `phase2/agents/api_agent.py:807`의 worker는 구조화 DTO와 fieldMappings를 보존하라고 하면서 성공 응답 필드명을 실제 DB 컬럼명과 일치시키라고 한다.
- 같은 worker의 source 예시는 request/auth/server/derived뿐이다. 실제 validator는 body/path/query/header/cookie/response 등 구분을 처리한다.

영향: camelCase DTO와 snake_case DB 매핑, 계산 결과 응답, path 파라미터를 body로 잘못 분류하는 문제가 반복될 수 있다. 앞선 라이브에서 body.plantId 오류가 확인됐다.

수정 방향: 외부 DTO는 계약대로 유지하고 저장 필드와 다를 때 명시적 매핑을 사용한다. source의 허용 형식과 예시를 validator와 일치시킨다. requestBody/successResponse 문자열은 구조화 계약에서 만든 요약이어야 하며 별도 설계 근거가 아니어야 한다.

## Feature Spec: 출력 형식·repair 식별

- `phase2/feature_generation.py`의 provider schema는 동적 JSON Schema를 포함하므로 strict=false이다. 제공자가 형식을 완전히 강제한다고 설명할 수 없다.
- 실제 실패에는 잘못된 JSON, parameters map, states map, 잘못된 기능을 고친 repair가 각각 있었다.
- 이번 작업에서 명확한 parameters map의 손실 없는 배열 변환, states/parameters 배열 예시, 정확한 featureId/sourceFeatureId repair allowlist를 추가했다.
- 유효하지 않은 JSON의 부분 복구, 범위 밖 기능 patch 수용, 무제한 재시도는 추가하지 않았다.

남은 방향: 출력 형식 예시를 실제 Pydantic schema와 동일한 단일 원천으로 관리하고, 필요한 경우 동적 schema 표현을 더 작은 typed 계약으로 나눈다. 상위 계약 검사를 줄여 억지로 통과시키지 않는다.

## PRD: 활성 경로와 legacy 경로 구분

- 현재 목록 생성은 `_build_item_prompt`/`ITEM_WORKER_SYSTEM`의 라벨 기반 단일 항목 worker를 사용한다. 따라서 파일 위쪽 JSON SECTION_PROMPTS만 수정해서 모든 현재 호출이 바뀐다고 볼 수 없다.
- 현재 KPI worker는 외부 근거가 없으면 내부 목표로 쓰도록 안내한다. legacy의 외부 출처 강제 규칙과 구분해야 한다.
- `_item_worker_node`는 형식 실패에 한 번 추가 호출하며, `_call`의 transport 실패가 빈 문자열로 바뀌면 이를 형식 실패로 다시 시도할 수 있다. 이번 502 상황에서 재시도와 fallback이 누적되는 경로다.

수정 방향: transport 실패와 출력 형식 실패를 구분해 retry 예산을 공유한다. 현재 실행 경로에 해당하는 프롬프트만 우선 고치고, 이전 JSON prompt와 기준은 명확히 분리한다.

## QA: 현재 라이브는 결정론적 경량 게이트

- `phase2/agents/qa_agent.py:229`의 `_LIGHTWEIGHT_GATE_ENABLED=True`로 live QA는 `_execute_lightweight_gate`를 호출한다.
- 전체 문서를 fixed로 다시 출력하라는 DB/API/PRD review prompt는 현재 이 경로에서 실행되지 않는다. 따라서 이번 live blocker를 QA LLM prompt 탓으로 돌리면 안 된다.
- 다만 legacy 모드를 다시 켜면 전체 문서 재출력 지시와 typed 산출물 형식의 불일치가 되살아날 수 있다.

수정 방향: 현재 결정론적 검사와 targeted repair를 유지한다. legacy 재활성화 전에는 전체 fixed 출력 지시를 제거하고 같은 계약·patch 형식을 사용하게 한다.

## Search: 근거 부족과 연결 실패는 별도 취급

- Web 검색을 사용하는 활성 경로에는 확인 불가 표기와 원문 URL 인용 지시가 있다.
- 그러나 하위 질의는 모든 주제에 국내 경쟁사 5개·퍼센트 통계·전자상거래/금융 법규 등을 요구한다. 자료가 없는 주제에도 형식을 채우도록 압박할 수 있다.
- 수집 예외를 텍스트로 합쳐 다음 단계에 넘기는 경로가 있으므로, 실패 메시지를 시장 근거로 소비하지 않게 해야 한다.

수정 방향: 서비스에 적용 가능한 조사 항목만 요청하고, 확인 불가/수집 실패/검증된 근거를 구조적으로 분리한다. 자료 개수를 충족시키려고 확인되지 않은 수치를 생성하면 안 된다.

## 프롬프트로 해결할 수 없는 오류

마지막 재실행에서는 Search와 PRD의 실제 upstream 호출에 `502 Codex request failed`가 반복됐다. 이는 제공자/게이트웨이 연결 실패이며 프롬프트 충돌과 구분해야 한다. 이 로그만으로 토큰 부족·할당량·서버 내부 원인을 단정하지 않는다. public endpoint의 upstream 서버 로그 확인이 필요하다.

## 적용 기록

### 실행 순서와 역할

- 최초 실행을 Search → PM → PRD → Feature Spec → DBA/API 병렬 → 결정론적 QA 순서로 변경했다. PRD는 이번 실행에서 만든 projectPlan을 받는다.
- PM의 물리 테이블·FK·endpoint 생성 지시를 제거했다. 목표·범위·사용자·MoSCoW와 설계 원칙만 정리하며 출력 예시에 wont를 포함했다.
- PM의 연결 실패/빈 계획, PRD의 연결 실패는 구조화 generation blocker로 남기고 후속 agent를 실행하지 않는다. 실패 단계를 완료 처리하지 않는다.
- frontend의 진행 순서와 실패 복원 순서도 PM → PRD로 맞췄다.

### 계약 우선 프롬프트

- DBA의 강제 BIGINT id, 제한된 타입 목록, 무조건 users/조직 테이블, *_id→id 추론, 복수형·최소 개수 강제를 명시 계약보다 우선하지 않게 수정했다. PostgreSQL 타입과 실제 PK/UNIQUE 참조를 보존하고 인덱스 예시는 CREATE INDEX로 맞췄다.
- API cap은 계약을 제거할 근거가 아니다. 모든 기능에 apiContract가 있으면 manager는 별도 LLM 계획 없이 계약 method/path를 직접 투영한다. 부분 legacy 입력의 cap도 선언 경로 수보다 작지 않게 한다.
- API DTO 이름을 DB 컬럼과 강제로 일치시키는 지시를 제거했다. source를 body/path/query/header/cookie/response/auth/server/derived로 구분하고 문자열은 구조화 스키마의 요약으로 취급한다. 계약의 parameters와 authRequired도 직접 보존한다.
- Feature Spec의 수작업 JSON 예시를 제거하고 GeneratedDocument의 실제 Pydantic schema를 prompt와 provider response_format이 공유한다. provider strict=false이며 엄격한 로컬 검사는 유지한다.

### 실패와 근거 분리

- PRD transport 오류를 빈 문자열로 반환하지 않는다. SDK 자동 재시도를 끄고 연결 실패를 format repair로 재호출하지 않는다. 기존 PRD가 있으면 보존한다.
- Search는 도메인에 해당하는 조사만 요청하며 경쟁사 개수·퍼센트·금융 법규를 강제하지 않는다. 상태는 market_research_status에 sourced/unavailable/failed로 분리한다. sourced는 추적 URL이 있다는 뜻이지 모든 문장을 독립 검증했다는 뜻은 아니다.
- 수집 예외의 원문은 시장 근거에 넣지 않는다. URL 없는 수치를 근거로 넘기지 않으며 확인 불가와 내부 제품 목표를 구분한다.
- 현재 QA는 계속 결정론적 경량 게이트다. 비활성 legacy prompt도 전체 fixed 재출력 대신 식별 가능한 patches만 요청하도록 정리했다. legacy parser의 기존 fixed 호환 처리는 제거하지 않았다.

### 실제 실행에서 추가 발견한 문제

첫 재실행은 Feature Spec의 중첩 responseSchema JSON 괄호 오류로 실패했다. 부분 JSON 복구나 QA 완화는 하지 않았다. auto 모드에서 이 오류가 발생하면 기존 1회 재시도 안에서만 json_object로 전환하고 같은 로컬 schema/계약 검사를 적용한다. 모드 전환은 유효한 JSON을 보장하거나 gateway를 복구한다는 의미가 아니다.

두 번째 실제 실행에서는 json_object가 정상 파싱됐지만 공유 PK default 불일치와 쉼표로 묶인 fieldMappings.column이 발견됐다. 참조 대상 필드를 중복 선언할 필요가 없고, 반복 선언 시 물리 계약을 완전히 동일하게 보존해야 한다는 지시를 추가했다. 배열 응답도 원소의 스칼라 필드별로 매핑한다. 공유 컬럼 충돌은 모든 선언 소유자를 피드백에 넣도록 검사 코드도 보강했다.

기본 출력 모드는 현재 gateway의 실제 실패를 근거로 json_object로 변경했다. auto/json_schema 명시 설정은 계속 지원하며, Pydantic 기반 prompt와 로컬 스키마 검증은 동일하다.

회귀 검증: Python PostgreSQL opt-in 포함 644건 통과, frontend 33건 통과 및 lint/production build 성공. 새 라이브 결과는 `docs/prompt-audit-live-qa-20261023.md`에 별도로 기록한다. 회귀 테스트 성공을 Phase4 저장 성공으로 간주하지 않는다.
