# 전체 산출물 QA 보완 — 2026-10-25

대상은 전공물품 중고거래 테스트에서 확인한 QA 항목이다. 기존 발행 문서를 수정하는 작업이 아니라 파이프라인의 다음 생성·검증 동작을 수정했다. 이전 변경이 많은 작업 폴더에서 관련 코드만 보완했으며 커밋·push하지 않았다.

## 적용 내용

| 문제 | 변경 |
|---|---|
| 문서 전체의 인증 요구가 공개 GET까지 차단 | 명시적 operation 인증 계약은 전역 추정보다 우선. 공개 읽기와 OWNER/PARTICIPANT 변경을 구분하며 legacy private 기본 검사는 유지 |
| API별 처리 테이블과 기능 대표 테이블 혼동 | 명시적 operation 주 테이블 또는 실제 effects/fieldMappings로 단일 처리 테이블을 정함. 존재·변경 대상 불일치 검사 추가 |
| worker가 잘못 만든 sessionPolicy 잔존 | typed contract에 없는 worker 정책을 제거하고 인증 업무 API는 공통 Registry 세션 정책 소비 |
| 사진 수정 입력·효과가 설명에만 존재 | operation에 requiredInputs/requiredEffectTables 추가. 필드·효과가 실제 없으면 최초 계약 검사에서 정확한 operation 위치로 repair 전달 |
| 물품명·상대방·등록 시각 등 화면 필드 누락 | requiredOutputs를 operation별 응답 경로로 명시. 응답 DTO 누락 검사 추가. repair가 수용 요구를 지워 통과시키는 동작 차단 |
| PRD와 Feature Spec 본문 상세도 차이 | Feature Spec 확정 후 해당 PRD 기능의 설명·요구사항·출력 요구도 계약과 함께 동기화 |
| PostgreSQL이 거절하는 시간 의존 인덱스 | 문자열 literal을 제외하고 volatile 함수·시간 키워드를 검사. 잘못된 일반 인덱스는 제거·경고, UNIQUE 인덱스는 의미를 임의로 바꾸지 않고 구조화 오류로 유지 |
| 입력 길이·빈 문자열·빈 PATCH 정책 차이 | 실제 필드 매핑이 있는 입력을 VARCHAR 길이에 맞추고 PATCH에 생성 입력의 값 제약을 보존. 빈 객체 PATCH는 거절 |
| 값 범위·상태 enum이 DB에 반영되지 않음 | Draft DB 필드에 minimum/maximum/enum 지원. 명시된 값만 SQL CHECK로 투영하고 교차 검증. 도메인 필드를 이름으로 추측해 추가하지 않음 |
| 목록 무제한 조회 | 별도 페이지 계약이 없는 생성 목록에 limit(1~100, 기본 20)/offset을 추가하고 기존 필터·소유권·정렬과 PK 보조 정렬 보존. 기존 cursor/page 선언은 덮어쓰지 않음 |
| 이미지 원본 저장·정리·재시도 책임 미정 | PRD에 명시적 파일 수명 정책과 저장 어댑터·URL 변환·worker·영속 재시도·보상 경계의 설계 기본값 추가. CDN과 저장소 구분 |
| 반응형·로딩·오류·본인 편집 UI 기준 부족 | PRD uiAcceptanceCriteria 및 생성 프롬프트로 화면 폭·상태·모바일 파일 선택·편집 여부 계약 보강 |
| KPI 이벤트 이름 불일치 | 동작/대상 토큰이 같은 단복수·어순 별칭을 하나의 이벤트로 연결하고 telemetryEvents에 별칭과 planned 상태 기록. 임의로 다른 동작을 합치지 않음 |
| 팀 정보 없는 일정을 확정처럼 표시 | releasePlanningAssumptions에 unvalidated_estimate 및 미정 팀 인원·가용 시간 명시. 실제 인원·근거는 만들지 않음 |
| 그래프·외부 도구에서 새 계약 추적 불가 | OpenAPI에 accessScope·primaryTable·pagination·수용 입력/출력/효과 확장 필드 유지 |

## 핵심 코드

- phase2/contract_quality.py: 공통 operation 정책·필드 제약·페이지 조회·DB CHECK·인덱스 규칙.
- phase2/feature_draft.py 및 agents/feature_spec_agent.py: 구조화 생성 모델·보정 보호·생성 프롬프트·PRD 동기화.
- phase2/operation_contract.py, registry_contract_guard.py: 초기 계약 투영·검증과 산출물 교차 검사.
- phase2/agents/api_agent.py, dba_agent.py, qa_agent.py 및 phase3/schema_validator.py: 병렬 생성·보정·최종 검증 연결.
- phase2/prd_quality.py, agents/prd_agent.py 및 openapi_spec.py: 운영·UI·측정·일정 정책과 OpenAPI 투영.

## 검사 결과

- PostgreSQL 통합 포함 전체 회귀: 878개 통과(57.14초).
- CI 대상 및 변경 공통 모듈 Ruff 통과.
- 기존 실패 기록과 명세 JSON은 보존한다. 새 Chat부터 시작한 라이브 E2E가 Phase4 및 Kafka 비동기 저장까지 완료됐다.
- 성공 실행: `test-runs/live-campus-major-market-20261025-035017Z`, pipelineId `69f7b10b-e223-4bd1-ac8d-c4b7194dd444`.
- 소요 시간 445.459초, 기능 6개(P0 4/P1 2), API 20개, 테이블 8개, FK 8개. 예상/실제 저장 chunk 611/611, kafkaPublished=true. 독립 저장 상태 조회와 PostgreSQL 유형별 개수도 일치한다.
- QA 승인, 독립 SchemaValidator, FK 및 OpenAPI 검증 통과. 계약 기반 그래프의 고립 기능/API/테이블 0개. 실제 프론트 그래프 화면을 열어 확인했다는 뜻은 아니다.
- 검증 실행에서는 Feature Spec 출력 모드를 환경 변수로 `json_object`로 지정했다. 코드의 기본값은 `auto`이며 이 설정 차이를 숨기지 않는다.
- 전체 고유 산출물 내용과 남은 제품 결정사항은 성공 실행 폴더의 `QA_REPORT.md`에 기록했다.

라이브 1차 `3a6c8343-9797-41c9-b233-e0fafaee89fa`는 첫 Feature Spec 호출 240초 timeout, 두 번째 응답의 중복 parentFeatureId 키로 종료했다. 중복 키를 임의 선택해 통과시키지 않았다. 기록은 `test-runs/live-campus-major-market-20261025-032826Z`에 있다.

다음 실행은 requiredInputs가 경로 변수도 본문에서만 찾는 신규 오탐을 발견해 Feature Spec 단계에서 수동 중단했다. 본문·path·query·명시된 인증 입력을 구분하도록 수정하고, 파일 업로드가 제외된 서비스에 파일 인프라 기본값이 추가되지 않는 회귀도 보완했다. 수동 중단은 성공/자동 검증 실패와 구분해 기록했다.

파일 수명 정책과 화면 기준은 실제 구현이 완료됐다는 주장이 아니라 생성 명세의 설계 기본값이다. 팀 인원·실제 시장 근거·재학생 인증을 꾸며내지 않는다. 외부 시장 자료는 추가 확인되지 않은 상태를 유지한다.
