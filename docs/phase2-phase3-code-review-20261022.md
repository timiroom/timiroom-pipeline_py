# Phase2 / Phase3 코드 및 로직 검토

- 검토일: 2026-10-22
- 범위: Phase2/Phase3 Python 28개 파일의 구조와 주요 생성·정규화·검증·복구 실행 경로. 모든 줄의 정확성을 증명한 전수 검증은 아니다.
- 이번 작업: 코드 검토 및 읽기 전용 helper 재현. 운영 코드 수정과 신규 라이브 E2E는 수행하지 않았다.
- 기존 변경사항은 유지했다. 아래 내용은 현재 작업 트리 기준이다.

## 1. 결론

API/DBA 프롬프트만 강화해서 해결되는 구조가 아니다. 불완전한 Feature Registry가 정상 계약처럼 전달되고, 정규화가 계약의 이름을 바꾸며, repair 진입점마다 재동기화 방식이 다르다. 검증에도 누락과 오탐이 공존한다.

현재 최초 생성의 실제 순서는 `Search → PRD → PM → Feature Spec → DBA/API 병렬 → QA/repair → Phase3`이다. 과거 논의한 `PM 계획 → PRD → Feature Spec`과는 다르다. 코드: `phase2/orchestration_graph.py:66`.

Phase3를 통과시키기 위해 hard gate를 무조건 낮추거나 재시도 횟수를 늘리는 것은 권장하지 않는다. 잘못된 계약의 유입을 막고, 수정 대상과 검증 기준을 일치시키는 것이 먼저다.

## 2. 최근 라이브 실행 근거

`test-runs/live-plant-care-20261022-130903/QA_REPORT.md` 및 같은 폴더의 산출물을 참조했다.

- 주제: 초보자를 위한 반려식물 관리 서비스, 그린로그.
- pipelineId: `380afa8b-e784-42bb-9603-b20ae2508ab5`.
- Chat 8턴 성공, 기능 12개, API 23개, DB 테이블 24개.
- 350.527초 후 Phase3 실패. Phase4 미실행, 해당 pipeline의 DB 저장 청크 0건.
- Feature Spec JSON 파싱 실패 후 fallback 사용. 원시 응답 전체와 finish_reason이 없어 토큰 잘림이라고 단정할 수 없다.
- 임시 ID가 실제 경로와 테이블명으로 전파됐다. 예: `/api/v1/de0b724e30s`.
- QA blockingIssues 배열은 DB 7 / API 196 / PRD 0. Phase3 schema validator 로그는 140개 오류 메시지다. 서로 중복 가능한 다른 집계이므로 합산한 독립 결함 수가 아니다.
- 마지막에는 API/DB repair 예산 소진으로 종료됐다. Docker 연결이나 단순 호출 지연이 주된 실패 원인은 아니었다.

## 3. 최우선 수정: 계약과 상태 보존

### P0-1. Feature Spec 실패를 정상 계약으로 위장하는 fallback

위치: `phase2/agents/feature_spec_agent.py:834`, `:77`, `:203`.

현재 LLM 예외나 JSON 파싱 실패가 발생하면 최소 기능을 만들어 진행한다. API 배열과 DB tables 배열이 존재하는지만으로는 실제 도메인 계약의 완성도를 보장하지 못한다. 임시 hash ID에서 도메인 이름을 추론하면 해시가 API 경로/테이블로 퍼질 수 있다.

수정: 버전이 있는 구조화 계약을 입력 경계에서 검증한다. 의미 있는 featureId, 적용 대상, 동작, 설명/요구사항, API operation 및 DB 필드 계약을 검증하고, 잘못된 항목만 제한적으로 보정한다. 복구하지 못하면 DBA/API 호출 전에 구체적 생성 blocker로 종료한다. 기술 이름을 임의로 만들어 정상 계약인 것처럼 통과시키지 않는다.

테스트: malformed JSON, 부분 유효 응답, 임시 ID, 필수 필드 누락, UI-only 기능, 정상 계약 보존. 실패 원인과 finish_reason/usage를 비밀정보 없이 기록한다.

### P0-2. PRD 요구사항 변경 감지 누락

위치: `phase2/orchestration_graph.py:501`.

현재 비교 대상은 ID/name/priority/actions/apiContract/dbContract 중심이다. description, requirements 등만 수정하면 변경 featureId가 빈 집합이다. 읽기 전용 probe로 재현했다.

수정: 동작에 영향을 주는 요구사항·수용조건·상태·권한·입출력 변경을 정규화해 비교한다. 단순 문자열 전체 비교 대신 안정적 requirement ID와 의미 있는 필드 기준을 사용한다. 변경된 부모 기능의 operation 및 자식 기능까지 영향 범위를 계산한다.

테스트: 요구사항만 변경, 권한 변경, 입력 필수값 변경, 단순 정렬/공백 변경, 기능 추가/삭제.

### P0-3. PRD repair 진입점별 재동기화 불일치

위치: `phase2/orchestration_graph.py:121`, `:241`, `:289`.

별도 rollback 경로에는 재동기화가 있지만, Phase3에서 진입하는 PRD repair 경로는 같은 과정을 보장하지 않는다. 한 repair 경로는 DBA를 PRD보다 먼저 처리한다. 수정된 PRD와 이전 Registry/API/DB가 함께 남을 위험이 있다.

수정: 모든 진입점에서 공통 `PRD delta → Feature Spec delta → 변경 도메인 targeted resync → 재검증` 절차를 사용한다. 기존 산출물을 보존하고, 변경이 없는 도메인은 실행하지 않으며, 도메인별 timeout과 patch 범위를 유지한다.

테스트: PRD-only Phase3 repair, DB만 영향, API만 영향, 요구사항만 영향, timeout 시 기존 산출물 보존.

### P0-4. 테이블 정규화가 반복 적용 시 테이블을 증가시킴

위치: `phase2/agents/dba_agent.py:2101`, `:689`.

sanitize_tables는 유효한 이름에 있는 숫자도 제거한다. 이후 계약 테이블 보충은 원래 이름을 다시 추가할 수 있다. 동일 계약으로 두 번 처리했을 때 테이블 수가 2개에서 3개로 증가하는 입력을 재현했다. 최근 라이브 결과의 `_2/_3/_4` 테이블 증가와 연결된다.

수정: 정상 식별자의 숫자를 일괄 제거하지 않는다. 임시 이름은 Registry 경계에서 해결한다. 필요한 이름 변경은 명시적 alias map으로 한 번 수행하고 FK/index/API 매핑까지 원자적으로 갱신한다. 정상화는 반드시 멱등적이어야 한다.

테스트: `normalize(normalize(schema)) == normalize(schema)`, 숫자를 포함한 정상 이름, 이름 충돌, FK/index/mapping 참조 보존, 보충 테이블의 PostgreSQL 타입/PK 문법.

### P0-5. targeted patch의 실제 수정 범위를 코드가 강제하지 않음

위치: `phase2/agents/api_agent.py:1151`, `phase2/agents/dba_agent.py:2543`, `phase2/agents/prd_agent.py:899`, `phase2/agents/pm_agent.py:344`.

API는 허용된 blocker 대상 이외의 endpoint patch를 받을 수 있다. DBA 허용 이름은 실제 결함 테이블이 아니라 기존 전체 테이블까지 포함한다. 컬럼 배열/relationship 교체는 정상 항목도 바꿀 수 있다. PRD는 섹션 단위 교체이며, PM의 위치 인덱스 기반 수정은 여러 삭제 시 대상이 어긋날 수 있다.

수정: base revision/hash, featureId, artifactKey, 허용 JSON pointer, add/replace/remove를 담는 patch envelope를 공통화한다. 적용 전 범위 검증, 적용 후 불변식 검증, 변경 diff 비교를 수행한다. 미수정 영역 변경은 거절한다. applied/noop/rejected/failed를 실제 승인된 변경 기준으로 기록한다.

테스트: 정상 endpoint/컬럼 무단 수정 거절, 일부 컬럼 변경 시 나머지 보존, 삭제/명시적 값 제거, stale revision, no-op, missing baseline에서 전체 생성 차단.

## 4. 높은 우선순위: 검증 정확성과 복구

### P1-1. 삭제한 기능이 Registry에 다시 남음

위치: `phase2/orchestration_graph.py:582`.

변경된 ID가 새 결과에 없으면 이전 객체를 기본값으로 유지한다. 삭제 기능이 보존되는 입력을 재현했다. 새 자식 ID는 core 변경 ID 목록에 없다는 이유로 제외될 수도 있다.

수정: added/removed/modified를 구분하고 parent/dependency 관계를 포함해 갱신한다. Registry, feature_specs, featureSpecDocument, legacy feature_list의 일관성도 함께 검증한다.

### P1-2. GET의 잘못된 DB fieldMappings를 놓침

위치: `phase2/operation_contract.py:152`.

requestSchema가 없으면 즉시 반환하므로, 일반적인 GET의 fieldMappings 대상 검사까지 건너뛴다. 존재하지 않는 `ghost.id`를 가리켜도 오류가 없는 입력을 재현했다. 이전에 추가한 검증에도 남아 있는 결함이다.

수정: body 여부와 무관하게 모든 mapping을 검사한다. source를 body/path/query/header/auth/server/response/derived 등으로 구분하고 JSON pointer, 대상 컬럼, 타입/변환을 검증한다. GET/DELETE body 제한은 현재 제품 정책과 표준 적합성 문제를 구분해 정의한다.

### P1-3. FK 대상의 테이블 존재만으로는 부족

위치: `phase3/schema_validator.py:459`, `phase2/agents/dba_agent.py:1019`, `phase2/agents/qa_agent.py:1738`.

`users`가 있으면 `REFERENCES users(missing_column)`을 놓치는 교차검사 입력을 재현했다. FK 타입 보정도 명시된 참조 컬럼 대신 대상 PK 중심으로 판단해 UNIQUE 컬럼 참조를 훼손할 위험이 있다.

수정: FK를 table/columns/referencedTable/referencedColumns/onDelete 등 구조로 다룬다. 대상 컬럼 존재, PK/UNIQUE, 양쪽 타입, 복합키 순서, SET NULL과 nullable 충돌을 검사한다. 확정할 수 없는 참조는 임의 연결/삭제하지 않고 근거 있는 repair 대상으로 남긴다.

### P1-4. QA와 Phase3의 중복 검사 및 문자열 오탐

위치: `phase2/agents/qa_agent.py:611`, `:701`, `:1310`, `phase3/validation_service.py`.

responseSchema에 array가 있어도 legacy successResponse 문자열에 `[]`가 없다는 이유로 누락을 판단할 수 있다. 여러 검사기가 같은 결함을 다른 문장으로 출력한다. blocker 상세도 자연어에서 코드를 추출해 대상 featureId와 혼동할 여지가 있다.

수정: 공통 checker가 `code/domain/featureId/artifactKey/pointer/severity/repairable/evidence/revision`을 반환하게 한다. 구조화 schema를 우선하고 문체/언어 문제는 계약 오류와 분리한다. issue identity로 중복 제거한다. 현재 revision의 검사 결과로 gate를 재계산한다.

### P1-5. repair 예산 및 반복 감지의 불일치

위치: `phase3/retry_service.py:70`, `:151`, `phase2/orchestration_graph.py:241`.

대상 중 한 도메인이라도 예산이 소진되면 다른 미사용 대상도 처리하지 않고 종료한다. 시도/적용/거절의 집계가 경로마다 다르다. fingerprint에는 순서와 표현이 바뀔 수 있는 blocker 문자열/목록이 들어간다.

수정: 도메인별 최대 1회 정책은 유지하되 실제 호출과 결과 상태를 일관되게 기록한다. upstream 계약 문제를 먼저 분류하고 영향받는 downstream의 예산을 불필요하게 소비하지 않는다. 동일 revision의 code+artifactKey+pointer 정렬 집합으로 반복을 판별한다. 일시적인 upstream 실패와 불변 계약 오류를 구분한다.

### P1-6. API 파라미터 및 공유 operation schema 손실

위치: `phase2/agents/api_agent.py:2010`, `:1674`.

동일 이름의 path/query 파라미터를 함께 주면 하나가 삭제되고 schema의 enum 등 상세가 사라지는 입력을 재현했다. 복수 featureId가 같은 endpoint에 각각 다른 schema를 제공하면 마지막 계약이 앞 계약을 덮는 입력도 확인했다. 자체검사는 placeholder 오류는 내지만 이 충돌 자체를 명확히 다루지 않는다.

수정: 파라미터 키는 `(in, name)`으로 하고 schema를 보존한다. method+canonical path 단위 Operation Registry를 두고 feature들은 operationId를 참조한다. 공유 operation 계약 충돌은 덮어쓰기 대신 명시적 오류로 처리한다. 204 response/body 정책도 최종 구조화 결과에 일관되게 적용한다.

### P1-7. DB 커버리지와 기능별 적용 대상

위치: `phase2/feature_registry.py:230`, `phase2/agents/feature_spec_agent.py:203`, `phase2/agent_contract.py:203`.

DB 커버리지 helper는 한 테이블의 featureId만 있어도 나머지 필수 계약 테이블 누락을 놓칠 수 있다. 모든 기능에 API+DB를 강제하면 반응형 화면 같은 UI-only 요구에도 가짜 리소스가 생긴다. 문자열 동작 추론은 '삭제를 지원하지 않는다'에서도 DELETE를 추론하는 입력이 확인됐다. 후속 검사 일부가 이를 잡더라도 단계 간 판정은 일치하지 않는다.

수정: artifact applicability, allowedActions/excludedActions, primary/dependency table 역할을 계약에 명시한다. required resource 전체를 검사하고, 자연어 키워드 추론을 hard gate의 근거로 삼지 않는다.

### P1-8. 입력 타입 경계와 우선순위 정책 통합

위치: `phase2/state.py`, `phase2/agents/prd_agent.py:1285`, `phase2/agents/feature_spec_agent.py`, `phase3/schema_validator.py`.

딕셔너리/문자열 변환이 여러 번 반복되며 일부 검사는 중첩 값이 정상 타입이라고 가정한다. PRD 우선순위 보정과 Feature Spec의 P0 승격/12개 제한 정책도 독립적으로 적용된다.

수정: typed artifact 검증을 각 단계 입출력에 둔다. malformed 입력은 예외로 중단하지 말고 구조화된 오류로 변환한다. MoSCoW와 releaseScope를 구분하고 dependency closure를 지키는 MVP 선택을 한 번 수행한 뒤 모든 소비자가 같은 목록을 사용한다. P2를 제외 범위와 동일시하지 않는다.

## 5. 구조 및 운영 개선

1. **PM/PRD 역할 정리**: 실제 실행 순서를 제품 의도와 맞춘다. PM이 projectPlan/goals/scope/targetUsers/priorities를 만들고 PRD와 Feature Spec이 이를 소비하는 구조를 선택한다면 모든 repair 경로도 같은 소유권을 따라야 한다.
2. **불필요한 LLM 재계획 축소**: Registry에 확정된 operation/table 계획은 결정론적으로 투영한다. worker는 부족한 상세만 생성한다. DBA/API 병렬 실행은 유지하고 호출 수/토큰/대기/실행 시간을 분리 기록한다.
3. **Registry hash 실효성**: `orchestration_graph.py:561`에서 hash를 생성하지만 소비/patch의 일치 확인까지 이어지지 않는다. artifact revision과 base hash 검증을 실제 적용한다.
4. **SSE 다중 구독**: `phase2/sse_service.py`의 pipeline 단일 queue는 여러 구독자가 이벤트를 나눠 가져갈 수 있다. 구독자별 전달, event ID/replay, heartbeat, 종료/TTL 정리, bounded buffer를 추가한다. replica가 여러 개면 공유 이벤트 저장소가 필요하다.
5. **관측성과 체크포인트**: stage별 입력/출력 revision과 검증 결과를 저장한다. 파싱 실패 원인을 finish_reason/usage와 연결하고, 로그에는 키·토큰·개인정보를 남기지 않는다. 병렬 한쪽 실패 시 다른 쪽 성공 산출물의 보존/취소 정책을 명시한다.
6. **검색 근거의 명확성**: active Search 경로와 별도 PublicSourceClient가 실제 연결되는지 정리한다. 웹 검색이 없는데 웹 근거가 확인된 것처럼 표현하지 않도록 evidence availability/provenance를 전달한다.
7. **비활성/중복 경로 정리**: 경량 QA에서 사용되지 않는 오래된 reviewer graph와 중복 유틸은 정확성 수정 후 정리한다. 지금 바로 전체 재설계할 필요는 없다.

## 6. 권장 수정 순서와 완료 기준

1. Feature Registry 입력 검증 및 안전하지 않은 fallback 차단.
2. 테이블 정규화 멱등성 및 모든 참조의 원자적 이름 갱신.
3. PRD 변경 감지·삭제·자식 영향 범위와 공통 targeted resync.
4. patch 허용 범위/기준 revision 강제 및 수정 결과 상태 통합.
5. API/DB 공통 구조화 checker, FK/fieldMappings/파라미터 회귀 보강.
6. 예산/반복 감지/현재 revision gate 통합.
7. 단위·통합 회귀 후 새 주제 라이브 E2E와 저장 결과 비교.

완료 기준은 Phase3 통과 표시만이 아니다. 의미 있는 기능 ID와 경로, 충분한 기능 설명/요구사항, 필수 API/DB 계약 완성, 유효 FK, graph의 설명 가능한 연결, 변경하지 않은 artifact 보존, Phase4 발행 및 실제 consumer 저장 확인이 함께 필요하다. UI-only 기능을 억지로 DB와 연결해 그래프를 채우는 것은 성공으로 보지 않는다.

## 7. 검증 상태

이전 구현 턴의 전체 테스트 결과는 481 passed였지만 위 경계 사례의 정확성을 보장하지 않는다. 이번 검토에서는 전체 suite를 새로 실행하지 않았다.

읽기 전용 재현에서 확인한 핵심 결과:

| 입력 | 현재 결과 | 기대 |
|---|---|---|
| PRD description/requirements만 변경 | 변경 ID 없음 | 영향 기능 감지 |
| 변경 기능을 새 Registry에서 삭제 | 이전 기능 잔존 | 명시적 삭제 반영 |
| 같은 DB 정규화 반복 | 테이블 2 → 3 | 동일 결과 |
| GET mapping → ghost.id, requestSchema 없음 | mapping 오류 없음 | 존재하지 않는 대상 검출 |
| FK → users(missing_column) | 교차검사 오류 없음 | 참조 컬럼 검증 |
| path/query 동일 이름, 상세 schema | 하나만 남고 schema 정보 손실 | 둘 모두 보존 |
| 공유 endpoint에 서로 다른 schema | 마지막 schema가 덮어씀 | 충돌 검출/단일 계약 |
| 필수 DB 테이블 일부만 존재 | coverage helper 누락 없음 | 빠진 필수 테이블 검출 |

이는 일부 helper의 재현 결과이며, 각 입력이 모든 후속 검사까지 통과한다고 주장하는 것은 아니다. 실제 파이프라인에서는 다른 검사기가 추가 오류를 낼 수 있다.
