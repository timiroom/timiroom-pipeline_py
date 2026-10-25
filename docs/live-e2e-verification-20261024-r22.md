# Chat → Phase4 재검증 및 개발 착수 판단

## 이번 실제 실행 결과

| 실행 | 주제 | Pipeline ID | Chat 포함 시간 | 결과 |
|---|---|---|---:|---|
| R22 | 반려식물 | 6446b6f5-5179-4e31-830b-b0391915a6e0 | 195.139초 | Feature Spec: DTO-only patch를 features 배열 누락으로 거부 |
| R23 | 공방 예약 | 7827a26a-648c-4cfd-831e-fae8794fc336 | 195.990초 | Feature Spec: 누락 API 및 소유권을 단일 보정에서 해결하지 못함 |
| R24 | 반려식물 | a1a63c79-ea1d-44cf-a568-a7d59088b15e | 231.283초 | Phase4 통과, Kafka 소비 및 실제 PG 340/340 청크 저장 |
| R25 | 공방 예약 | af48999e-af18-4b52-a108-9d49e54e8e67 | 197.542초 | Feature Spec: 복합 결함의 허용 수정 범위가 과도하게 좁게 계산됨 |
| R26 | 공방 예약 | 03033e9b-2d3c-4b60-b09d-9ec8b6df61ba | 172.952초 | Feature Spec: ReservationValidationRequest 신규 DTO patch가 허용 범위에서 누락됨 |
| R27 | 공방 예약 | 4d7648de-c0e6-4963-9c92-987f2afb8661 | 214.225초 | Phase4 통과, Kafka 소비 및 실제 PG 335/335 청크 저장 |

실패 실행은 저장 0건이다. 검사에서 보이는 빈 API/DB나 임시 intent ID는 채택된 최종 산출물이 아니라 Feature Spec 이전 상태이며, 이를 실제 API/DB 생성 실패와 혼동하지 않는다. 실패한 원본은 재발행하거나 저장 성공으로 변경하지 않았다.

## 이번에 수정하고 확인한 코드 결함

1. `strict_payload`가 전체 생성과 부분 수정에 같은 features 필수 규칙을 적용했다. 유효한 baseline이 있는 draft repair에서만 schemas/tables 단독 응답을 허용한다. 최초 생성, 잘못된 배열 타입, 중복 JSON 키, 허용되지 않은 자산 변경 검사는 유지한다.
2. missing GET과 기존 API의 응답 누락이 함께 있으면 operationIndices가 빈 배열이 되어 필요한 기존 API 수정까지 막았다. 기능 수준 출력 결함이 명시된 기능의 기존 응답 계약과 누락 method를 함께 수정할 수 있도록 했다. endpoint 위치가 명시된 결함의 제한과 다른 기능 보존 규칙은 유지한다.

PostgreSQL 통합 검증을 활성화한 전체 테스트 775개 통과(12.90초), Ruff 및 git diff --check 통과. R25의 실제 원본/보정 응답도 수정 후 읽기 전용 preflight에서 오류 0건이었다. 이는 원본 실행의 E2E 성공을 뜻하지 않는다.

## 반려식물 성공본 확인

- 기능 6개(P0 5/P1 1), API 17개, DB 9개, FK 8개, OpenAPI 3.1.0
- 8종 명세를 340개 청크로 실제 저장했다. source_key도 340개로 중복이 없다.
- 독립 저장 API와 서버 재시작 후 저장 API 모두 stored / expected=340 / stored=340 / kafkaPublished=true였다.
- FK·OpenAPI·기능 계약 누락·임시 기능 ID·빈 설명/요구사항·고립 기능/API/테이블 0건
- PRD의 fallbacks 및 itemDiagnostics 0건
- 전체 원문: `test-runs/live-plant-care-20261024-184011/ALL_SPECIFICATIONS.md`
- 상세 내용 QA: 같은 폴더의 `QA_REPORT.md`

## 개발 착수 판단

반려식물 성공본은 인증, 식물 CRUD, 물주기 기록, 사진 업로드/조회 등부터 개발을 시작할 수 있는 연결된 초안이다. 그러나 전체를 확정된 개발 계약으로 사용하기에는 다음 내용이 부족하다.

1. 일정/알림 최초 데이터 생성의 실제 insert 또는 시스템 작업 계약, 실행 주기 및 중복 방지 기준
2. 물주기 기록이 어느 일정을 완료시키는지 날짜/시간대/허용 구간을 포함한 계산식
3. 증상 안내의 초기 데이터 공급·갱신·입력 매칭·미일치 처리
4. 부모 삭제와 외부 파일 정리의 DB 제출/비동기 재시도/보상 경계

위 항목을 확정하기 전에 해당 기능을 개발하면 개발자가 정책을 추측해야 한다. 파이프라인 저장 성공과 생성된 서비스의 업무 명세 완결성은 다른 기준이다. PRD auth 선택의 모호함과 의미상 반복된 설명/범위도 정리할 여지가 있다.

## R26에서 확인한 결함과 후속 해결

R26은 equipment.manage와 availability.manage를 같은 PRD sourceFeatureId의 prd_core로 각각 만들었다. 핵심 1:1 대응이 깨졌으며 reservation.validation supporting은 API 적용 대상인데 operation이 비어 있었다. 목록의 requestedAt/updatedAt/cancelledAt 응답 매핑과 인증 ownership도 누락됐다.

R26 보정은 비어 있던 API에 POST /reservations/validate와 새 DTO를 추가했으나, 당시 신규 DTO 허용 규칙은 missing core 또는 missing method에 집중돼 이 보정의 ReservationValidationRequest를 거부했다. 또한 중복 core를 missing source처럼 처리하는 범위 계산과, 같은 source가 중복된 기능의 병합/재분류 방식도 문제가 있었다. R26 자체는 정상 Feature Spec/API/DB를 만들지 못한 실패로 보존한다.

후속 수정에서는 중복과 누락을 분리하고, 중복은 기존 기능만 재분류하도록 범위를 지정했다. 빈 API를 채우는 operation에 직접 필요한 신규 DTO/참조 의존성만 허용하며 무관한 변경은 차단한다. 명시적으로 supporting으로 재분류된 기능을 기존 core source 때문에 재승격하던 후처리도 고쳤다.

집중 테스트 143개, PostgreSQL 통합 포함 전체 781개(12.91초), Ruff를 통과했다. 새 Chat부터 실행한 R27은 기능 6개(P0 5/P1 1), API 14개, DB 5개, FK 4개를 생성했고 Phase3 첫 검증 및 Phase4 실제 저장까지 통과했다. FK/OpenAPI/계약/임시 ID/빈 설명 및 고립 기능·API·테이블 검사는 모두 0건이었다. 이번 실제 보정은 인증 지원 기능의 부모/이유 누락과 소유권 FK 두 건이었다. 중복 core/빈 API 경로는 테스트로 검증했으며 이번 라이브에서 재발했다고 주장하지 않는다.

독립 저장 API와 실제 PostgreSQL 조회로 8종 문서 335/335 청크 저장을 교차 확인했다. 상세 근거와 전체 명세는 `test-runs/live-workshop-booking-20261024-190417/QA_REPORT.md`, `ALL_SPECIFICATIONS.md`에 있다. 우선 목표인 Phase4 실제 저장 완료는 달성했으며 앞 절의 업무 명세 보완과는 구분한다.

## 범위와 한계

실제 API 호출 기반 테스트이며 기존 승인 명세를 덮어쓰지 않았다. 두 주제 모두 RAG 결과 0건이고 PDF 입력은 없었다. 외부 시장 근거 및 생성된 앱 자체의 실행/부하/브라우저 UI는 검증하지 않았다. 그래프 고립 0건은 산출물 데이터 검사이며 프론트 렌더링 검증이 아니다. 모든 실행은 종료됐고 테스트용 8081 서버도 종료했다. Docker와 사용자의 다른 서버는 유지했다.
