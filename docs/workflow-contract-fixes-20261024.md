# 생성 명세의 업무 계약 보완

## 대상과 범위

R27 공방 예약 결과의 `CONTENT_REVIEW.md`에서 확인한 주요 6개 문제를 새 명세 생성 로직에 반영했다. 특정 서비스의 테이블명이나 상태값을 하드코딩하지 않고, 명시적인 업무 정책을 Feature Spec → PRD/API/DBA → QA가 공유하도록 했다.

기존 저장 명세 및 승인 문서는 수정하지 않았다. 후속 라이브 R28/R29를 실행했으며 둘 다 Feature Spec 단계에서 실패했다. 이전 R27의 335/335 저장 성공과 이번 변경 후 라이브 결과는 별개다. 최신 실행의 Phase4 성공은 아직 확인하지 못했다.

## 변경 전과 후

| 문제 | 변경 전 | 변경 후 |
|---|---|---|
| 인증 전달 | 세션 저장소만 계약화하고 API 설명/OpenAPI는 JWT를 고정 출력 | sessionPolicy에 transport, tokenFormat, tokenField/cookieName, 만료·쿠키/CSRF 정책을 선언한다. Bearer 발급 응답의 필수 토큰 또는 쿠키 발급의 Set-Cookie를 검사한다. API 설명과 OpenAPI 보안 방식도 같은 정책에서 생성한다. |
| 초기 예약의 중복 방어 | 초기 상태와 점유 상태가 자연어로 따로 생성 | statePolicies의 초기 상태를 실제 DB default와 비교한다. intervalPolicies의 blockingStates가 허용 상태 집합에 속하는지, blocksOnCreate=true일 때 초기 상태를 포함하는지 검사한다. |
| 확정/취소 불일치 | 도달 불가능한 상태를 DTO에 나열하거나 취소 전이를 말로만 선언 | 초기 상태·실제 effect·명시적 외부 공급을 기준으로 상태 도달성을 검사한다. 선언한 전이가 실제 operation effect에 있는지, 응답 enum과 정책이 모순되지 않는지 확인한다. |
| 조회 진입점 누락 | 기능에 쓰기 API만 있어도 기능 커버리지 통과 | readDependencies에 조회 경로와 반환 필드 및 용도를 선언한다. 경로/응답 필드/소비자 역할 접근성을 검사한다. 새 draft의 UI 자원 UUID·정수 경로에는 식별자 탐색 경로도 확인한다. |
| 운영자 계정·권한 | 계정 공급 경계가 없고 SYSTEM 소유 범위가 공개 접근으로 변환 | defaultRole과 roleProvisioning(seed/invite/external)을 계약화한다. 역할 공급과 초대 API 존재 여부, 공개 인증 진입점의 임의 권한 선택을 검사한다. SYSTEM과 공개 접근을 분리하고 원래 access 문장을 보존한다. |
| 시간대와 예약의 연관성 | 실재하지 않는 슬롯 FK를 전제로 삭제 조건을 설명 | intervalPolicies에 실제 자원/구간/잠금 컬럼과 점유 상태, 시간대, 수정·삭제 보호 정책을 선언한다. 직접 슬롯 참조를 선택했다면 컬럼과 실제 FK를 검사하고, 아니면 자원 키 및 구간 포함 관계를 생성 지침으로 요구한다. |

## 구조화한 정책

- `sessionPolicy`: 기존 strategy/sessionTable을 확장한다. 불투명 Bearer 세션과 JWT를 구분하며 cookie 방식도 지원한다.
- `readDependencies`: 필요한 GET 경로, responseField, purpose. 다른 기능이 제공하는 조회도 재사용한다.
- `statePolicies`: table, column, initial, states 및 근거가 있는 externalStates.
- `intervalPolicies`: 점유 테이블과 자원/시작/종료/상태 컬럼, blockingStates, 가용 구간 테이블, 잠금 대상, 시간대 및 변경 보호 정책. 구간 도메인에서만 사용한다.
- `allowedRoles`, `responseHeaders`: API까지 보존하고 OpenAPI에 역할 확장 필드와 실제 응답 헤더로 투영한다.

예를 들어 자원 배정의 초기 상태가 `pending`이고 생성 즉시 자원을 점유한다면 `blockingStates`에도 `pending`이 있어야 한다. `confirmed`를 허용 상태에 넣었다면 실제 확정 effect나 명시적인 외부 공급 경계가 있어야 한다. 공방·예약이라는 이름 자체를 검사하지 않는다.

## 소비 및 자체 검증

1. 새 typed draft는 `workflowContractVersion=1`로 컴파일한다. 상태 컬럼에 default가 있는 주요 자원은 상태 정책 누락도 검사한다. 기존 legacy 계약 전체에 새로운 draft 규칙을 소급 강제하지 않는다.
2. 생성 preflight, Registry 검사, API self-check, QA/Phase3 교차 검사에서 같은 검증 함수를 사용한다. API 실제 응답 DTO의 토큰 누락도 Registry만 믿고 통과시키지 않는다.
3. DB 투영에는 선언된 상태 enum의 CHECK와 구간 start < end CHECK를 추가한다. QA는 실제 DB 명세에서 해당 CHECK 누락을 검사한다. 임의 EXCLUDE 제약이나 extension 설치를 생성한 것은 아니며 동시성은 명시한 잠금/트랜잭션 정책을 구현해야 한다.
4. 역할 제한·점유 충돌·상태 전이에서 직접 도출되는 403/409와 인증/서버 오류 응답을 계약 및 OpenAPI에 보존한다. 기존 오류 설명은 유지하고 중복 추가하지 않는다.
5. PRD 핵심 기능에도 정책을 동기화하고 DBA featureMappings/API workflowPolicies/OpenAPI 확장 필드로 전달한다. 새 정책을 이유로 기능 카드 수를 늘리지 않는다.

## 부분 수정과 재동기화

- 누락 조회는 `missingOperationKeys`에 정확한 method/path를 지정한다. 해당 경로 및 직접 참조하는 새 DTO만 추가할 수 있으며 무관한 경로/테이블/DTO 변경은 차단한다.
- 부분 수정 중 이미 약속한 readDependencies 또는 상태·구간 정책을 삭제해 검사를 우회하지 못한다.
- 최초 생성 및 기존 단일 부분 보정 예산을 유지한다. 무제한 LLM 재시도나 전체 재생성 fallback을 추가하지 않았다.
- session/read 정책 변경은 API, state/interval 정책 변경은 API/DB 재동기화 대상이 된다. 설명 문장만 변경된 경우에는 두 도메인을 재실행하지 않는다.

## 수정 파일

- `phase2/workflow_contract.py`: 정책 모델, 공통 검증 및 투영 메타데이터/오류 응답
- `phase2/feature_draft.py`, `feature_generation.py`, `artifact_shapes.py`: typed draft, bounded repair 및 shape 경계
- `phase2/agents/feature_spec_agent.py`: 생성 지침과 PRD 계약 동기화
- `phase2/agent_contract.py`: SYSTEM 접근 정책 보존
- `phase2/agents/api_agent.py`, `openapi_spec.py`: 실제 인증 방식·헤더·역할·정책 전달
- `phase2/agents/dba_agent.py`, `registry_contract_guard.py`, `operation_contract.py`: DB 정책 투영과 공유 검증
- `phase2/orchestration_graph.py`: 변경 정책의 targeted 재동기화 판별
- `tests/test_workflow_contracts.py`: 위 결함 및 보존/범위 제한 회귀 테스트

## 검증

- PostgreSQL 통합 검증을 포함한 전체 테스트: **824개 통과**.
- 새 업무 계약 테스트: **43개**. 인증, 권한 공급, 상태 도달성/전이/응답 enum, 점유 상태, 조회/권한 의존성, DB CHECK, 정확한 경로의 단일 부분 보정, PRD 및 OpenAPI 투영, 정책만 변경한 재동기화를 포함한다.
- CI 범위 Ruff 및 수정한 에이전트/계약 파일의 F/I 검사 통과.
- 회귀 검증은 단위·통합 테스트와 mocked provider의 생성/부분 수정 검증이다. 실제 LLM 생성은 아래 후속 실행에서 별도 검증했다. 생성된 서비스의 업무 실행·동시 예약 부하 및 프론트 화면은 검증하지 않았다.

## 후속 라이브 E2E

| 실행 | Pipeline ID | 시간 | 결과 |
|---|---|---:|---|
| R28 | ae9bbf00-d5a7-4f45-a449-6e2ab1fbff2d | 216.102초 | 최초 중복 DTO, 재응답의 상태 전이/공통 정책 검증 오탐으로 Feature Spec 실패. 저장 0건 |
| R29 | 61d42ac1-a36b-49fd-830b-3d8c17cedb81 | 206.089초 | 오탐 수정 후 새 Chat부터 재실행. 최초 JSON 구문 오류, 재응답 auth.session의 apiContract 누락으로 Feature Spec 실패. 저장 0건 |

R28에서 set 및 요청 enum 기반 update도 실제 상태 전이의 구현 근거임을 확인해 검사기를 보완했다. 읽기 전용 기능은 이미 선언된 테이블/컬럼의 공통 상태 정책을 소비할 수 있도록 고쳤다. 해당 회귀 3개를 추가해 전체 824개가 통과했다. R28 실패 원본을 다시 발행하거나 성공으로 바꾸지는 않았다.

R29는 새로운 생성 오류가 남았다. 회원가입·로그인·로그아웃이 있는 auth.account 외에 인증 미들웨어 역할의 auth.session을 API 기능으로 따로 만들고 필수 API 계약을 누락했다. 현재는 생성 프롬프트의 인증 여정 응집성과 JSON 출력 안정성 보완이 필요하다. 상세 실행 근거는 `test-runs/live-workshop-booking-20261024-230658/QA_REPORT.md`에 있다.

자연어 요구사항의 모든 의미를 코드가 증명하는 것은 아니다. 주요 6개 결함을 생성 지침과 명시적 계약 검사로 보강한 것이며, 모든 주제의 무조건적 성공을 보장하지 않는다.
