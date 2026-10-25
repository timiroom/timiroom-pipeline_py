# 실제 E2E 반복 검증 및 수정 기록 (2026-10-24)

## 검증 기준

- 실제 Chat에서 FormData를 수집하고 Phase1 검색부터 Phase4 Kafka 발행 및 PostgreSQL 저장까지 실행한다.
- 단위 테스트, 저장된 응답의 재검사, Kafka 발행만으로 전체 성공을 주장하지 않는다.
- 실패 실행의 원본 응답과 terminal 산출물을 보존한다. 검증 통과를 위해 hard gate를 끄거나 repair 예산을 늘리지 않는다.
- RAG 검색 결과 0건인 실행은 사용자 입력만 사용한 경우로 구분한다. PDF 입력과 프론트엔드 그래프 시각화는 이번 실행의 검증 범위가 아니다.

## 변경 전 문제와 변경 후 동작

### 1. 최종 기능 ID와 원본 PRD 식별자 분리

기존에는 PRD의 임시 `intent.*` 식별자가 최종 Feature Spec 기능 ID로 복사되거나 `supporting.auth`처럼 분류명이 ID가 됐다. prompt의 예시 슬롯에도 임시 ID가 있어 같은 오류를 재생산했다.

이제 prompt 입력에서는 임시 식별자를 `sourceFeatureId`에만 전달하고 최종 ID에는 `plant.register`, `auth.session` 같은 의미 있는 dotted ID를 요구한다. 생성 경계에서도 검사한다. PRD와의 원본 추적 정보는 삭제하지 않는다.

### 2. 명시적인 물리 DB 계약을 소비하는 DBA

기존 DBA 후처리는 실제 계약의 `users`를 별도 `user` 테이블로 추정하거나 UUID FK를 BIGINT로 바꾸고, 설명의 키워드에서 미선언 `status`, `content`, `image_url` 컬럼을 추가했다. 올바른 worker 응답도 이후 정규화에서 달라질 수 있었다.

완전한 물리 계약과 PK가 있으면 공통 계약에서 테이블·컬럼·PK·FK·타입을 결정론적으로 투영한다. 명시된 ownerEntity/ownerKey를 사용하며 이름으로 principal 테이블을 만들지 않는다. 관계는 실제 FK에서 도출한다. 불완전한 legacy 입력은 기존 호환 경로를 유지한다.

공통 테이블의 필드 병합을 끝낸 뒤 FK constraint를 렌더링해, 다른 기능의 같은 컬럼 선언이 REFERENCES를 지우지 않게 했다. 인덱스의 `DESC`, `ASC NULLS LAST`는 컬럼 오탈자가 아니라 정렬 지시로 보존한다.

### 3. 하위 Agent 실행 전 계약 검증

기존에는 ownerEntity 오타, 잘못된 ownerKey, 약속한 수정/삭제 operation 누락, DTO에 없는 출력값, 비어 있는 write transaction 규칙을 DBA/API 생성 후에 발견했다.

Feature Spec 생성 단계에서 같은 결정론적 검사를 먼저 실행한다. 정확한 feature/operation pointer를 남기고 기존 1회 targeted repair 안에서 보정한다. generic 동작 단어만으로 PATCH/DELETE를 강요하지 않고 명시적인 수정/삭제 약속만 확인한다.

### 4. 저장값과 API transport 검사

기존 insert 검사는 primaryTable에 한정되어 부수 테이블의 필수값 누락을 놓쳤고, `user_id`, `password_hash` 같은 이름을 서버 자동값으로 오인할 수 있었다.

명시된 모든 insert/create effect 테이블을 검사한다. typed source가 request이면 이름 기반 자동값 추정보다 우선하며, 각 NOT NULL 값의 입력·매핑·생성 근거가 필요하다. body가 없는 계약에서는 worker가 임의로 만든 requestSchema를 제거하고 계약을 보존한다.

### 5. JSON 출력 구조와 재시도

같은 DB 키가 중복되거나 `uniqueConstraints: [[...]]`의 배열 끝이 누락되어 JSON 자체가 깨진 실제 응답이 있었다. provider가 json_object 모드를 수락해도 문법을 보장하지 않았다.

중복 키는 정확한 JSON pointer로 거부한다. provider에 안내하는 UNIQUE 구조를 `[{"columns": [...]}]`로 평탄화하고 기존 배열 입력은 로컬 호환만 유지한다. 파싱 불가능한 assistant 응답은 다음 prompt에 재주입하지 않는다. 잘못된 JSON을 추측해서 복구하거나 성공 처리하지 않는다.

### 6. 부분 patch 병합과 정상 중첩 출력의 QA 오탐

R4에서는 모델이 `featureId`와 변경할 ownership만 반환했는데, 병합기가 생략된 sourceFeatureId를 삭제로 취급해 patch를 거부했다. 동시에 `items[].message`, `guidance.recommendedActions`를 flat property 이름과 비교해 정상 중첩 응답을 누락으로 판정했다.

기존 기능의 sparse patch는 생략한 필드를 보존하며 중첩 객체만 재귀 병합한다. 배열은 명시적으로 반환했을 때만 교체한다. 명시적인 source 변경/삭제와 무관한 endpoint·DTO 수정은 계속 거부한다. 출력 요구사항은 실제 JSON Schema 경로로 확인하며, 잘못된 부모 경로는 같은 leaf가 있어도 통과하지 않는다.

R4 원본 응답 및 원래 repairScope를 그대로 사용한 읽기 전용 재검사에서 기존 오류 6개 중 4개는 오탐으로 제거됐고, 실제 오류 2개는 원래 sparse patch로 수정되어 생성 경계 오류가 0개가 됐다. 이것은 새 E2E 성공을 의미하지 않는다.

### 7. KPI/페르소나 정책의 일관성

개수 채우기 fallback을 제거하고 worker·조립·QA·repair가 같은 정책으로 검사하도록 바꾼 앞선 수정도 함께 검증한다. 내부 목표와 외부 근거, 실제 사용자 세그먼트와 가정 페르소나를 구분한다. KPI의 이벤트·분모·측정 단위·기간, 연령 범위, 제외 범위를 보존한다.

`releaseScope: "mvp"`나 정상 프레임워크 이름은 형식 취향만으로 hard blocker/문장 오염으로 판정하지 않는다. 실제 계약·측정 정의 누락 검사는 유지한다.

### 8. 경로 이름 대신 명시적 세션 계약으로 인증 진입점 판별

R5의 `/auth/users`, `/auth/sessions`는 가입·로그인으로 세션을 발급하는 공개 진입점이지만, Phase3는 `/signup`, `/login` 같은 경로 접미사만 예외로 인정했다. API Agent가 canonical `authRequired=false`를 보존하므로 잘못된 QA 피드백을 받아도 오탐이 반복됐다.

명시적 sessionPolicy·세션 테이블·POST action·세션 insert/create effect·요청 스키마를 갖춘 계약은 실제 경로와 무관하게 세션 발급 진입점으로 인정한다. 임의의 일반 쓰기 endpoint나 로그아웃은 예외가 아니다. 기존 R5 산출물은 수정 없이 읽기 전용 재검사에서 구조·FK·OpenAPI·기능 그래프 오류가 0개였다. R5 자체는 실패 실행이며 Kafka 발행/저장은 없었다.

같은 계약을 supporting 우선순위에도 사용한다. 이름에 로그인/회원가입 단어가 없는 인증 기능을 P2로 낮추지 않고 P1 출시 필수로 분류한다. 명시적 프로젝트 제외 범위는 그대로 존중한다.

### 9. endpoint 단위 sparse patch와 진단 집계

R7은 누락된 reason만 고친 patch가 envelope version을 생략해 거부됐고, 그 오류 뒤에는 정의되지 않은 DTO 참조도 숨어 있었다. R8에서는 API 배열에 수정 대상 endpoint만 반환한 정상적인 부분 응답을 전체 배열로 해석해 정상 형제 endpoint 삭제 시도로 판정했다.

envelope의 미변경 version은 원본에서 상속하고 다른 버전·알 수 없는 top-level 키는 거부한다. 허용된 method/path에 대응하는 부분 operation patch는 원래 슬롯에 병합해 나머지 endpoint의 순서·내용을 보존한다. 새로운 method는 명시적으로 누락된 method만 추가한다. route를 명확히 매칭할 수 없거나 허용 범위 밖의 endpoint를 변경하면 여전히 거부한다. 잘못된 optional 필드를 포함한 기존 operation은 완전한 유효 replacement가 온 경우에만 모델 기본값을 사용할 수 있다.

필수 필드 검사에서 실패해도 이름 있는 DTO·테이블 참조 검사를 함께 집계한다. reason 누락을 고친 다음에야 missing DTO가 드러나 한 번의 repair 예산을 낭비하던 경로를 회귀 테스트로 막았다.

### 10. 구조화된 action과 본문/메타데이터 검사 분리

R8의 조회 기능은 '변경·취소 내역'을 표시하는 요구사항이었지만, 단순 변경 키워드가 불필요한 PATCH endpoint를 요구했다. 이제 PRD에 구체적인 actions가 있으면 이를 우선 사용한다. legacy 문서처럼 구조화 actions가 없는 경우에만 이전 자연어 보조 검사를 사용한다.

R9에서 고정된 `persona.6c0889087300`의 숫자 구간이 'source identifier leak'에 걸렸다. 모델이 정상 본문을 다시 써도 코드가 같은 ID를 붙여 같은 오류가 반복됐다. 본문 오염 검사는 구조화된 segmentId/featureId 등 기계 식별자와 분리했다. 본문에 노출된 출처 식별자는 여전히 검출하고, 기능 ID·참조 유효성은 별도 계약 검사로 유지한다.

생성 프롬프트도 동일 규칙에 맞춰 보강했다. 실제 경로 변수명과 column 이름을 구분하고, 부모를 경유하는 소유권은 부모 테이블 및 자식 FK로 명시하며, 기존 DB를 읽어 계산하는 기능도 DB 적용 대상으로 표현한다. 신규 서비스에서 로컬 로그인만 있고 계정 생성 경로가 없는 결과를 줄이도록 계정 공급 경계도 명시했다.

### 11. 저장 성공 후 내용 QA에서 추가로 고친 부분

자동 QA 통과를 내용 검토의 종료로 보지 않았다. R10은 실제 저장까지 성공했지만, 읽어보니 인증 후처리의 중복 FK와 평균 KPI 분모 누락이 있었다.

- 완전한 물리 계약에서는 인증 후처리가 테이블명으로 principal 컬럼을 추정해 추가하지 않는다. 기존의 불완전한 legacy 계약 보강은 유지한다.
- 평균 KPI에는 완료 과업/세션 건수 분모가 필요하다. 생성 prompt의 '비율 아니면 분모 없음' 지시를 고치고 worker·조립·QA 공통 정책에 평균 검사를 추가했다. 가설 기준선과 실제 측정값도 구분한다.
- 임의 테넌트 추가를 막고 실제 소속 컨텍스트가 필요하면 사용자·소속 연결 및 초기 공급 경계를 요구한다. 변경 가능한 status를 이력 UNIQUE에 넣지 않도록 하고 활성 구간 충돌 검사와 자원 잠금을 명시하게 했다.
- 기술 스택은 영문 고유 기술명이 정상적인 내용이다. 구조화 techStack 영역과 개별 기술 선택 문장에서는 넓은 mixed-language 휴리스틱만 제외하고, 실제 prompt/runtime 유출·깨진 문자열 검사는 유지한다.
- R13의 추가 내용 검토에서 최초 데이터 생성 경계와 상태/역할 계약의 명확성이 부족한 부분을 발견했다. 모든 사용자별 저장 테이블의 초기 insert/상위 생성/자동 생성 책임, 도달 가능한 상태만의 enum, 역할별 약속의 실제 operation, 부모 삭제에 따른 외부 파일 보상 경계를 공통 프롬프트에 보강했다.

마지막 항목은 프롬프트가 모든 자연어 정합성을 증명한다는 뜻이 아니다. 실제 산출물 내용을 별도 QA로 읽고 확인한다.

### 12. 최종 재검증에서 확인된 세션 QA와 보정 응답 경계

R14의 LLM 보정 응답은 DTO의 schemas 배열을 features 배열 안에 넣었다. 잘못된 JSON 구조를 임의 복구해 승인하지 않고 기존 거부 규칙을 유지했다. 보정 prompt에 세 전역 배열이 형제라는 점과 각 entry의 식별키를 명시하고, 실패 진단은 실제 `/features/3/featureId` 같은 위치를 포함하도록 개선했다.

R15는 refresh API가 없는 bearer_session 모델이었다. UUID 세션 PK와 expires_at/revoked_at가 실제 존재하지만, legacy QA가 logout 경로만 보고 refresh token_hash를 요구했다. 이제 명시적 bearer_session.sessionTable이 실제 UUID PK·만료·폐기 필드를 갖춘 경우 로그아웃 저장 계약을 인정한다. 만료/폐기 누락, 다른 테이블 지정, 순차 BIGINT 세션 식별자 및 실제 refresh operation에는 이 예외를 허용하지 않는다. 회귀 테스트 6가지로 경계를 확인했다.

같은 실행의 본문 없는 typed POST logout에 API worker가 requestSchema=null을 반환해 불필요한 LLM repair가 발생했다. 완전한 typed 계약에 요청 DTO가 없으면 HTTP method와 무관하게 임의 worker body/schema를 제거한다. 불완전한 legacy POST 계약은 이 규칙으로 임의 변경하지 않는다.

R16은 최초 JSON에 문서 envelope 전체를 features 원소로 잘못 넣어 전역 tables/schemas 및 기능 식별자가 없었다. 주소 지정이 불가능한 응답을 부분 수정 baseline으로 보존하면서 복구할 수 없는 원소도 계속 남았다. 이제 전역 배열과 고유 식별키가 없으면 유효 계약이 아직 없는 최초 생성 실패로 처리한다. 기존 단 한 번의 재시도에서 잘못된 assistant JSON을 재주입하지 않고 완전한 초안을 다시 요청한다. 정상 식별키가 있는 계약은 여전히 targeted patch만 허용한다. 잘못된 중첩 JSON을 추출·구제하거나 호출 예산을 늘리지 않는다.

### 13. 실제 저장에서 발견된 legacy 전역 hash 인덱스

R17은 Phase3를 통과하고 Kafka 발행까지 완료했지만, 같은 내용 청크가 다른 문서에도 있어 저장 트랜잭션이 rollback됐다. 재시도 3회 후 DLQ로 이동했고 저장 확인은 expected=452 / stored=0 / pending으로 정확히 실패를 보고했다.

원인은 source_key 기반 저장 코드는 이미 적용됐지만, migration이 자동 이름의 content_hash UNIQUE constraint만 제거한 것이었다. 실제 로컬 DB에는 이름이 다른 일반·partial UNIQUE 인덱스 `idx_chunks_content_hash`, `uq_chunks_content_hash`가 남아 있었다. PostgreSQL catalog에서 대상 RAG 테이블의 content_hash 단일 키 unique 인덱스와 constraint를 찾아 제거하도록 바꿨다. 다른 테이블, PK, 복합 키 및 source_key UNIQUE는 건드리지 않는다. 원문·벡터·metadata 행은 삭제하지 않는다.

격리 PostgreSQL 테스트로 세 종류 legacy 인덱스/constraint 제거의 멱등성, 다른 테이블 보존, 동일 내용의 pipeline/type별 3건 저장 및 동일 source 재실행 시 0건 추가를 검증했다. 실제 로컬 DB에도 `python migrate_db.py`를 적용했다. R17 실패 원본을 재발행해서 새 성공으로 기록하지 않고 새 Chat부터 검증한다.

### 14. 부모 FK 경유 설정과 소유권 누락의 생성 경계

R18의 care_schedules는 enabled와 plant_id를 갖고 plants.user_id를 통해 사용자를 참조했지만, legacy QA는 설정 테이블에 user_id가 직접 있어야 한다고 판단했다. 이제 실제 FK 그래프를 따라 인증 주체 테이블로 도달하는 설정도 인정한다. 누락된 FK나 인증 주체로 연결되지 않는 순환은 인정하지 않으며, 방문 집합으로 무한 순환을 방지한다. 특정 서비스나 테이블명을 하드코딩하지 않는다.

R19의 equipment에는 owner_id → users.id가 명시돼 있었으나 Feature Spec의 ownership 자체가 빠져 있었다. 최초 계약 검사에서 통과한 뒤 legacy 정규화가 user_id를 기본 소유권으로 만들어 정상 물리 계약과 충돌했다. 인증된 저장 기능에는 명시적인 ownership scope/access 계약이 필요하다는 검사를 Feature Spec preflight에 추가했다. 이제 누락은 허용된 단일 부분 보정의 대상으로 명확히 제시하며, 테이블명으로 owner_id/user_id 중 하나를 추측해 승인하지 않는다.

R20에서는 모델이 잘못된 ownership을 고치는 대신 null로 지웠고, R21에서는 반환 DTO의 status 누락에 대해 같은 outputRequirements만 다시 적었다. 둘 다 실패로 유지했다. 마지막 보강은 보정 입력에 요청된 기능의 실제 primaryTable/FK ownerKey·ownerEntity 후보를 구조화해 제공하는 것이다. 모델이 선택하도록 돕되 코드가 소유권을 임의 확정하지 않는다. 출력 필드 누락은 operation.responseType/schemas.nodes 및 적절한 응답 status로 수정해야 하며 outputRequirements 반복이나 삭제로 우회하지 못하도록 보정 지시를 명확히 했다. 이 마지막 보강은 전체 회귀 테스트까지 확인했으나 이후 새 라이브 E2E는 아직 실행하지 않았다.

### 15. R26 이후 보정 범위와 supporting 분류 보완

같은 PRD source에 core가 둘 있으면 기존 범위 계산은 이를 missing core로 취급해 세 번째 core 추가를 유도했다. 이제 실제 source별 개수를 세어 0개인 경우만 missing으로 분류하고, 중복이면 기존 기능들만 대상으로 지정한다. 하나의 기존 ID만 core로 유지하고 나머지는 원래 source와 계약을 보존한 supporting으로 재분류하도록 지시한다.

빈 apiContract를 채우는 수정은 missing method와 별도로 범위에 표시한다. 해당 operation이 직접 참조하는 신규 DTO 및 그 참조 의존성만 허용하고, 무관한 DTO나 새 테이블 추가는 거부한다. 기존 기능 선택은 정확한 featureId를 우선하며 source 대응은 유일한 경우에만 사용한다.

명시적으로 supporting으로 보정한 기능이 원래 source가 core라는 이유로 다시 core로 승격되는 후처리도 차단했다. 생성 횟수와 단일 부분 보정 예산 및 hard gate는 유지한다.

보강 후 집중 테스트 143개, PostgreSQL 통합 포함 전체 781개가 통과했다. 새 Chat부터 실행한 R27 공방 예약은 214.225초에 Phase4 및 실제 Kafka/PG 335/335 청크 저장을 완료했다. 기능 6개, API 14개, DB 5개, FK 4개이며 Phase3는 첫 검증에 통과했다. 상세 근거는 `test-runs/live-workshop-booking-20261024-190417/QA_REPORT.md`에 있다. 중복 core/빈 API 경로는 회귀 테스트로 확인했고 이번 라이브 실행의 보정 대상은 인증 계약 두 건이었다.

## 실제 실행 이력

후속 실제 재검증 R22~R27 및 코드 수정, 개발 착수 판단은 `docs/live-e2e-verification-20261024-r22.md`에 기록했다. R24 반려식물은 실제 340/340 청크 저장에 성공했고, R26 공방 실패를 보완한 뒤 새 R27 공방도 335/335 저장에 성공했다. 단일 실행 성공을 모든 주제의 안정적 완주 보장으로 일반화하지 않는다.

| 실행 | Pipeline ID | Chat 포함 시간 | 도달 단계 | 실제 결과 |
|---|---|---:|---|---|
| R1 | 395d997b-bd29-44fd-88ad-9292c93035dc | 203.993초 | Feature Spec | 중복 JSON key 및 생성 계약 오류. 저장 0건 |
| R2 | 55f23ff3-7ac5-4051-8795-ed6befaaa660 | 253.647초 | Phase3 | 기능 6, API 16, DB 9. phantom owner/FK 및 계약 불일치. 저장 0건 |
| R3 | 462eae9b-6e75-47c8-9dbf-71c6b12d69ad | 215.524초 | Feature Spec | 두 응답 모두 nested UNIQUE 배열의 JSON 구문 오류. 저장 0건 |
| R4 | 366266df-2113-468d-aea2-08e1687c9e9f | 170.879초 | Feature Spec | sparse patch를 source 삭제로 오판. 저장 0건 |
| R5 | 0c94d75b-0bdd-4216-b935-d948ca347b74 | 244.235초 | Phase3 | 기능 6, API 19, DB 9, FK 12. 세션 진입점 인증 오탐 2건. 저장 0건 |
| R6 | ef01ec7b-61dd-4e20-b671-3abb3c63eeba | 278.838초 | Phase3 | 기능 6, API 23, DB 9, FK 8. 세션 create action 인증 오탐 1건. 저장 0건 |
| R7 | dbf6705c-a976-46a2-a61f-c275072ac98b | 174.949초 | Feature Spec | reason 부분 patch의 version 생략 거부. 저장 0건 |
| R8 (공방) | 8f8349c9-2c36-4003-98a1-e2e32597b139 | 170.080초 | Feature Spec | 부분 endpoint 배열 병합 및 action QA 오탐. 저장 0건 |
| R9 | 2d69ce9e-a248-46e2-9880-ea7858a41690 | 164.922초 | Feature Spec | 최초 소유권/범위 문제, 보정 응답의 JSON 구문 오류. 저장 0건 |
| R10 (공방) | ef7148d9-b2c4-4da1-a219-5048ea718551 | 225.158초 | Phase4 저장 완료 | 기능 6, API 15, DB 6, FK 6. 8종 문서 333/333 청크 저장 |
| R11 | 12907a74-3686-48cc-bf99-41bca200b503 | 222.746초 | Feature Spec | 중복 JSON key 이후 보정본의 supporting/operation 누락. 저장 0건 |
| R12 (공방) | 6dcfef87-21cf-440c-9525-69007bde87da | 239.506초 | Phase4 저장 완료 | 기능 6, API 15, DB 6, FK 10. 8종 문서 410/410 청크 저장 |
| R13 | d7f385aa-41dd-47cd-a182-894431fd1c61 | 220.351초 | Phase4 저장 완료 | 기능 6, API 18, DB 7, FK 6. 8종 문서 352/352 청크 저장 |
| R14 | ba238013-e323-4bb1-a6e9-84e85ba3cbc9 | 199.184초 | Feature Spec | DTO patch가 features 안에 잘못 중첩됨. 저장 0건 |
| R15 (공방) | 7a5c47e6-a4b2-4a59-ac06-ea10aa93c625 | 243.310초 | Phase3 | bearer 세션 로그아웃에 refresh 저장소를 요구하는 오탐. 저장 0건 |
| R16 | 01ea437b-a6ab-4fda-8782-d7cf2cfdc8b9 | 212.705초 | Feature Spec | 주소 지정 불가능한 최초 초안을 부분 수정 기준으로 유지해 실패. 저장 0건 |
| R17 (공방) | 0ab82c70-0eed-4d00-b7b6-27ceb11b826c | 367.271초 | 실제 저장 확인 | Phase3 통과·Kafka 발행 후 legacy 전역 hash UNIQUE 충돌. 452개 기대 / 0개 저장 |
| R18 | d1a0f373-28f2-4a06-b44b-94d3567a1dda | 229.470초 | Phase3 | 부모 FK 경유 사용자 알림 설정을 인정하지 않는 오탐. 저장 0건 |
| R19 (공방) | eaac1d66-24f9-432c-a6a5-74d930bdf351 | 251.269초 | Phase3 | 생성 ownership 누락 후 user_id 기본값과 실제 owner_id 충돌. 저장 0건 |
| R20 | c0a59944-f289-45c2-b5fa-3d85f0597efb | 167.244초 | Feature Spec | 보정이 잘못된 ownership을 null로 지워 필수 계약 검사 실패. 저장 0건 |
| R21 (공방) | 592992aa-a0c0-4866-b610-39cc1115675e | 198.788초 | Feature Spec | 취소 응답 status DTO 누락을 보정하지 못함. 저장 0건 |

실행별 원본 및 분리 산출물은 `test-runs/live-plant-care-20261024-*`에 있다. 로컬 raw diagnostics는 opt-in이며 SSE에는 노출하지 않는다.

## 회귀 검증

- 부분 patch/중첩 출력/DB 인덱스 등을 포함한 집중 테스트: 119개 통과.
- PostgreSQL 저장 통합 검증을 활성화한 전체 테스트: 후속 수정 포함 781개 통과(12.91초).
- `ruff check config/settings.py tests phase2/feature_draft.py phase2/prd_policy.py`: 통과.

## 이전 성공 근거와 현재 미완료 항목

R12/R13 모두 Phase3에서 추가 targeted repair 없이 통과했다. Feature Spec 자체의 기존 최대 1회 보정 예산은 유지되며 R12는 취소 응답 출력 필드 1건을 보정했고 R13은 최초 생성에서 통과했다.

R12는 서버 재시작 후 독립 저장 상태 API에서도 `stored`, `expectedChunks=410`, `storedChunks=410`, `kafkaPublished=true`가 복구됐다. R12/R13의 읽기 전용 재검사에서 FK·OpenAPI·기능 계약 누락은 없고 고립 기능/API/테이블도 0개였다.

전체 실제 산출물은 각 실행 폴더의 `ALL_SPECIFICATIONS.md`, 개별 JSON, `review_metrics.json`에서 확인한다. 수동 내용 QA는 성공 폴더별 `QA_REPORT.md`로 별도 보존하며, 자동 통과와 제품 운영 정책 확정 및 실제 생성 앱 구현 검증을 구분한다.

후속 R24는 실제 로컬 legacy 인덱스 migration 이후 Kafka+DB 저장까지 통과했고 서버 재시작 후에도 340/340 저장이 복구됐다. R26 공방의 신규 DTO 보정 범위 및 core 중복 문제는 위 15절과 같이 보완했으며, 최신 R27 공방은 실제 335/335 저장까지 완료했다. 이번 우선 목표인 Phase4 저장 완료는 달성했다. 모든 업무 정책의 완결성이나 모든 주제의 성공을 보장하지는 않는다. 테스트 서버는 종료했고 계속 실행 중인 E2E는 없다.
