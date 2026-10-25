# Phase4 및 산출물 품질 안정화 기록

## 완료 기준

새 Chat에서 요구사항을 수집해 Phase1~4를 실제 실행하고, Kafka 발행뿐 아니라 consumer의 PostgreSQL 저장까지 확인한다. 생성된 기능 설명/요구사항, PRD 추적, API 입출력과 인증, 실제 DB 필드/FK 및 업무 상태/예약 구간을 함께 검토한다. 이전 성공 결과를 최신 코드의 성공 근거로 사용하지 않는다.

## 최종 상태 (10월 25일)

사용자의 최신 우선순위인 **Phase4 실제 저장을 완료**했다. 단, 검증된 실제 생성 체크포인트에서 DBA/API부터 재개한 결과이며 새 Chat 단일 E2E 성공으로 집계하지 않는다. Pipeline `546f740c-f25f-45ec-be70-cbfe1a13e8ef`, 재개 후 76.061초, Kafka consumer의 PostgreSQL 저장 **449/449**. 별도 storage API와 직접 PostgreSQL 조회도 확인했다. 기능 6/API 15/테이블 5/FK 4, 계약·FK·OpenAPI 오류와 연결 없는 관계 항목 0. 전체 테스트 **862개 통과**, Ruff 통과.

상세 기록과 전체 산출물은 `test-runs/checkpoint-phase4-20261024-184939Z/QA_REPORT.md`, `ALL_SPECIFICATIONS.md`에 있다. 아래 실행 중 표시는 당시 경과 기록이며 이 최종 상태를 우선한다.

마지막 주요 수정은 SDK뿐 아니라 공유 LlmRuntime에도 계약 전용 timeout을 적용한 것, 실제 요청값의 set 효과와 역할별 조회 소비자 검사, 본문 없는 POST 취소를 신규 행 생성으로 오인하던 QA 검사 수정이다. 새 Chat R41/R42/R43 실패와 첫 체크포인트 재개 실패도 그대로 보존했다. 저장 성공을 위해 검증 gate를 끄거나 수동 가짜 API/테이블을 넣지 않았다.

## 최근 실패와 수정

| 실행 | 실제 결과 | 원인과 후속 수정 |
|---|---|---|
| R30 / d7cdd1c8-1742-4acd-afb7-0b9efe28fc92 | 200.637초, Feature Spec 실패, 저장 0 | 최초 응답의 중복 키와 재응답의 시간 컬럼에 대한 잘못된 상태 정책. 생성 지침에서 시간/null 기반 만료와 문자열 상태를 구분했다. 실제 다른 기능이 구현한 상태 전이와 부모 조회로 얻을 수 있는 식별자를 검사기가 인정하도록 수정했다. |
| R31 / 75394b0b-46b1-44f8-9c1f-a23ab1dd6e75 | 166.035초, Feature Spec 부분 수정 범위 검사 실패, 저장 0 | 도달 불가능한 상태 enum을 공유 DTO에서 고치는 부분 수정이 무관한 변경으로 오인됐다. 해당 enum을 사용하는 실제 operation도 결함 범위에 넣고 참조 DTO를 재귀적으로 포함했다. 다른 기능에 부분 범위가 있을 때 전체 영향 기능의 DTO까지 잘못 보호하던 조건을 수정했다. |
| R32 / 466771ad-c1ad-4144-a367-5d18f0f44b08 | 224.451초, Feature Spec 실패, 저장 0 | 최초 JSON 구문 오류 후 재응답에서 인증 기능의 ownership 및 supporting 부모/이유 누락. ownership/access를 새 draft의 필수 필드로 하고 core/supporting을 구분한 타입으로 부모 규칙을 스키마에 표현했다. |
| R33 / 4846eb3f-0566-43be-b172-d2acd78c1cb4 | Feature Spec 실패, Phase4 미진행 | 최초 응답이 전역 테이블/DTO를 features 내부에 중첩했다. 재응답은 실제 items[] DTO를 [].필드로 잘못 매핑했고, 명시된 운영자 seed를 상태 검사기가 인정하지 못했다. 최상위 출력 구조를 다시 안내하고 유일한 배열 래퍼의 기계적 경로 정규화 및 계정 공급 근거의 상태 검사 반영을 추가했다. |
| R34 / 5ede8edd-b594-476a-b6ab-10cdc066a522 | Feature Spec 부분 수정 보존 검사 실패, Phase4 미진행 | 새 조회 의존성을 추가하면서 기존 시간대 조회 의존성을 대체한 응답을 거절했다. readDependencies를 키 단위 부분 병합으로 변경해 기존 약속을 보존했다. 목록 GET이 실제 id를 반환하는 경우에도 경로 변수의 DB 매핑 유무 차이로 누락으로 판단하던 조건을 수정했다. 허용된 기존 DTO의 새 하위 DTO 참조가 부분 수정 범위에서 빠지던 경계도 수정했다. |
| R35 / bfd389ca-6a5a-4793-b0f7-833af6a0b3d0 | Feature Spec 보정 후 boolean 정책 타입 오류, Phase4 미진행 | DB/요청의 활성 플래그는 boolean인데 구간 상태는 문자열 true/false로 생성됐다. boolean 상태의 typed 정책, 실제 boolean update의 도달성 및 TRUE/FALSE SQL CHECK를 지원했다. SQL BOOL/BOOLEAN 컬럼으로 확정된 정책만 명시적 true/false 문자열을 boolean으로 정규화한다. |
| R36 / 109bf21e-7e71-4e6e-b64d-212a9514bda5 | Feature Spec 보정 병합 실패, Phase4 미진행 | 실제 boolean 정책과 문자열 active/inactive 전이가 불일치했고 보정 응답의 경로/effects만 있는 API 항목이 전체 교체로 처리됐다. 기능 단위 보정도 기존 method/path 키의 API에 부분 병합하도록 수정했다. 전이 DTO는 boolean 값을 지원하고 잘못된 전이 값의 오류에 실제 허용 상태를 제공한다. |
| R37 / a205d912-b9c7-476f-99ee-f3abd9b9b948 | 최초 JSON 구문 오류, 재응답 요청 필드 매핑 오류, Phase4 미진행 | 최초 응답은 의미 검사 전 JSON 단계에서 실패했다. 재응답의 운영자 상태 처리에서 요청 DTO는 status인데 이력 to_status 매핑의 입력을 toStatus로 작성했다. 본문 필드가 없는 실제 결함이므로 임의 추정 보정이나 gate 우회로 통과시키지 않았다. |
| R38 / 9b7afe48-3155-43f1-a6e9-c209c83afd68 | 도달 불가능한 상태 보정 후 인증 입력 검사 오탐, Phase4 미진행 | Bearer 정책이 있는 비공개 operation의 Authorization 헤더를 별도 parameter가 없다는 이유로 거절했다. 명시된 인증 transport의 표준 헤더/지정 쿠키만 실제 입력으로 인정하도록 수정했다. 실제 응답 model도 GPT-5.4 Mini임을 진단에서 확인했다. |
| R39 / 67617e04-fd3e-4514-9940-05c36b24a196 | 전역 자산 미생성 후 재응답 필드/전이 불일치, Phase4 미진행 | 최초 응답의 전역 tables/schemas가 비어 있어 재시도가 소비됐다. 재응답에 임의 sessionToken 헤더와 boolean 정책에 active/inactive 문자열 전이가 남았다. 출력 JSON을 추정 복구하지 않고 거절했다. |
| R40 / a2fe0ce1-91fd-4936-9a40-f691f804ed17 | 226.572초, Feature Spec 실패, Phase4 미진행 | medium Mini에서도 최초 JSON 구문 오류와 재응답의 operation 내부 readDependencies 오배치가 남았다. 모델 분리와 집중 실제 생성 검증을 진행했다. |

R31 원본과 실제 부분 수정 응답을 수정된 병합기로 읽기 전용 재검사했을 때 preflight 결함은 0건이었다. 이는 회귀 검증이지 R31 파이프라인의 성공이나 재발행이 아니다. R32의 API/DB가 비어 있는 것은 Feature Spec에서 중단됐기 때문이며, 독립적인 API/DB 생성 실패로 집계하지 않는다.

## 생성 및 검증 경계

- JSON Schema 안내를 기본 모드로 사용한다. 현재 provider 호출은 strict=false이므로 스키마 준수나 구문 무결성을 보장하지 않는다. 기존 한정된 fallback/부분 수정 예산을 유지한다.
- draft 프롬프트의 JSON Schema annotation title을 줄이되, 실제 title이라는 필드명과 literal/default 값 및 required/타입 제약은 유지한다.
- 회원가입/로그인/로그아웃/세션은 하나의 일관된 인증 기능으로 생성하도록 안내한다. 미들웨어 설명만 있는 별도 기능 카드로 API 계약을 생략하지 못한다.
- 상태 전이는 실제 set/state_transition effect 또는 해당 물리 컬럼에 매핑된 요청 enum update를 근거로 검사한다. 임의의 일반 문자열 update를 근거로 상태 도달성을 통과시키지 않는다.
- 식별자는 접근 가능한 조회의 실제 PK 응답 매핑을 따라 탐색한다. 부모 식별자에서 자식 목록으로 도달하는 경우도 인정하지만, 출발 조회가 없는 순환이나 권한이 없는 조회는 통과시키지 않는다.
- 부분 수정은 영향 기능/operation/참조 DTO만 허용한다. 계약 삭제나 전체 재생성으로 결함 검사를 우회하지 않는다.
- 응답 매핑이 `[].id`이나 실제 DTO가 단 하나의 대응 배열 속성 `items[].id`를 가진 경우에만 기존 매핑의 경로를 정규화한다. table/column/source는 바꾸지 않고 진단에 전후 경로를 기록한다. 후보가 여러 개이거나 해당 필드가 없거나 명시적 pointer가 있으면 수정하지 않는다. request/body 매핑과 JSON 구문 오류는 이 정규화 대상이 아니다.
- 운영자 seed/external 또는 실제 초대 API가 있는 roleProvisioning은 세션 테이블 FK가 가리키는 계정의 role 상태 공급 근거로 인정한다. 무관한 테이블의 상태나 구현 없는 초대를 공급 근거로 인정하지 않는다.
- readDependencies의 부분 응답은 method/path/responseField로 병합하며 기존 조회 약속은 보존한다. 명시적 빈 배열 삭제와 중복 키는 거절한다. 수정 허용 DTO가 새 DTO를 참조할 때만 그 참조 경로의 새 정의를 허용하고, 연결되지 않은 정의는 거절한다.
- boolean 상태는 실제 boolean 초기값/enum/점유값과 SQL TRUE/FALSE를 공유한다. 해당 컬럼에 매핑된 boolean update는 true/false의 전이 근거이며 const가 있으면 그 값만 허용한다. 일반 문자열 요청이나 무관한 컬럼의 update를 boolean 전이 근거로 삼지 않는다. 알 수 없는 문자열 상태를 임의로 수정하지 않는다.
- feature 전체가 영향 범위에 있더라도 API 부분 응답은 기존 method/path로 병합한다. 응답에서 생략한 action, 성공 상태, DTO, 다른 경로를 잃지 않는다. 추가 경로는 기존 missing-method/key 범위에서만 허용한다.
- 보정 입력에 `protectedSchemaNames`를 별도 제공한다. 정상 operation이 소비하는 공유 DTO는 보호하고, 추가 필드가 필요한 operation은 새 전용 DTO를 사용하도록 안내한다. 동일한 보호 DTO를 그대로 반환하는 것은 허용하지만 변경은 거절한다.
- 검증된 workflow draft의 설명/요구사항에는 이전 PRD의 미확정 문장을 무조건 다시 덧붙이지 않는다. 입력 PRD는 생성 프롬프트로 전달하고, 확정된 기능 계약의 설명/요구사항을 PRD/API/DB에 동기화한다. supporting의 검증된 P1/P2/P3 구분을 단어 점수만으로 다시 강등하지 않는다. 기존 legacy 보강 경로는 유지한다.
- 효과의 컬럼/전이 의미 검사는 타입 파싱을 예외로 중단시키는 대신 기존 전체 preflight에서 수집한다. 컬럼 누락과 from/to 누락은 여전히 blocker이며, 다른 기능의 결함도 같은 보정 입력에 전달된다. 불완전한 effect가 최종 계약으로 승인되지는 않는다.
- 복잡한 공유 계약 생성 호출에만 `FEATURE_SPEC_REASONING_EFFORT`를 추가했다. low로 R38/R39를 비교한 후 medium으로 보완했다. Mini의 장문 생성 불안정이 남아 후속 집중 테스트부터 Feature Spec만 `FEATURE_SPEC_MODEL=gpt-5.4`로 분리했다. Chat/PM/PRD 등 다른 단계는 기존 모델 설정을 유지하며 빈 모델 설정은 기존 공통 모델로 되돌릴 수 있다. reasoning을 쓰는 호출에는 temperature를 섞지 않는다. 요청/응답 모델명과 적용 effort를 진단에 기록한다. reasoning과 최종 계약 출력을 같은 토큰 예산 안에 담을 수 있도록 Feature Spec 출력 상한은 24,000으로 조정했다. 계약 전용 request timeout은 240초이며 Chat timeout은 유지했다. 호출/보정 횟수는 늘리지 않았다. 이는 성공 보장이 아니다. 설정 근거: [공식 모델 문서](https://developers.openai.com/api/docs/models/gpt-5.4), [공식 파라미터 호환 지침](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-5.4).

## 검증 상태

- R32 후 수정한 코드의 PostgreSQL 통합 포함 전체 테스트: 831개 통과(14.00초).
- config/settings.py, tests, workflow_contract.py, feature_draft.py Ruff 검사 통과.
- R33 후 전체 테스트: 840개 통과(13.02초). 해당 실패 응답의 읽기 전용 재검사는 남은 결함 0건이었다. 실패 실행을 재발행한 것은 아니다.
- R34 원본과 실제 부분 응답의 읽기 전용 재검사는 후속 수정 후 결함 0건이었다. R34를 성공으로 재발행하지 않았다.
- R35 원본과 실제 부분 응답의 읽기 전용 재검사도 후속 타입 보완 후 결함 0건이었다. 이 결과는 새 E2E 성공을 대신하지 않는다.
- R36 후 전체 테스트 직전 기준: 846개 통과(13.69초), 이후 기능 단위 경로 병합 회귀 1개 추가.
- R37 후 전체 테스트: 847개 통과(13.51초).
- R38 후 전체 테스트: 853개 통과(15.74초).
- 후속 검증은 `scripts/audit_live_feature_generation.py`로 실제 계약 생성을 먼저 좁혀 확인한다. 이 도구는 새 모델 응답을 생성하지만 저장된 PRD를 입력으로 사용하며 Kafka에 발행하지 않는다. 성공해도 Chat→Phase4 E2E로 집계하지 않는다. 이후 새 Chat에서 전체 E2E를 다시 실행한다.

## 집중 생성 검증

- `feature-probe-20261024-180342Z`: Mini medium의 실제 새 생성과 단일 보정이 79.450초에 통과했다. 이후 전체 R40은 여전히 실패했으므로 한 번의 성공으로 안정화를 선언하지 않았다.
- GPT-5.4 첫 집중 요청은 기존 120초에서 timeout됐다. 구 설정의 재시도 프로세스만 종료하고 별도 240초 제한으로 새 검증했다. 짧은 실제 GPT-5.4 요청은 정상 응답해 모델 접근 불가와 긴 생성 지연을 구분했다.
- `feature-probe-20261024-181437Z`: 68.592초. from/to가 없는 effect 하나가 초기 구조 검사를 중단시켜 다른 결함이 첫 보정에 전달되지 않았다. 효과 의미 검사를 기존 전체 preflight에서 함께 수집하도록 수정했다.
- `feature-probe-20261024-181753Z`: GPT-5.4 medium, **158.540초, 최초 생성 1회 통과, 보정 0회**. 실제 응답의 모델명도 gpt-5.4였다. 이 검증은 기존 PRD 입력의 새 Feature Spec 생성이며 Phase4 발행은 하지 않았다.
- 최신 전체 회귀: PostgreSQL 통합 포함 **858개 통과(13.20초)**. CI Ruff 및 변경 에이전트/조립 코드 F/I 검사 통과.
- R41: 위 수정이 반영된 서버에서 새 Chat부터 전체 E2E 실행 중. 최신 Phase4 성공 및 내용 검토는 아직 아래에 기록되지 않았다.

아직 최신 코드의 Phase4 성공을 확인한 기록은 아니다. 실제 서비스 구현의 동시성/보안 동작까지 생성 문서 검토만으로 증명할 수는 없다.
