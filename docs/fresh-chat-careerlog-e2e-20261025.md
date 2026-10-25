# 커리어로그 새 Chat → Phase4 실제 E2E

검증일: 2026-10-25 (Asia/Seoul). 주제는 개인 취업 지원 현황·전형 상태·면접 일정·준비 메모 관리 웹 서비스다. 기존 공방 예약 체크포인트와 다른 주제이며, 매 실행마다 실제 Chat 8턴과 새 pipelineId를 사용한다. 실제 LLM API 호출, PostgreSQL 5433, Kafka 9092를 사용한다.

## 인프라 준비

Docker Desktop 초기 기동에서 `dockerInference` 임시 소켓 접근 오류를 확인했다. 임시 소켓 이름 변경은 실패했고 데이터나 볼륨은 삭제하지 않았다. Docker 재기동 후 엔진 동작을 확인했으며, 사용하지 않는 Model Runner는 공식 `docker desktop disable model-runner` 명령으로 비활성화했다. 공유 PostgreSQL·Redis·Zookeeper·Kafka는 기존 compose/볼륨으로 실행했다.

## 실패 기록과 수정

1. 첫 실행: `73875af1-49cb-4c23-82ff-d31fd9909e4d`, `test-runs/live-job-application-tracker-20261025-004826Z`.
   - 382.354초 후 Phase3 차단. `API_AUXILIARY_ONLY_CONTRACT`가 지원 상태 변경 기능을 오탐했다.
   - 기존 검사는 `parentFeatureId`만 따라가고 명시적 `dependencies`의 동일 자원 관리 경로는 확인하지 않았다. 두 참조를 순환 방지와 함께 탐색하도록 수정했다. 관계 없는 인증 경로나 선언하지 않은 다른 기능은 여전히 허용 근거로 사용하지 않는다.
   - Chat이 마침표 뒤 `상태는 ...`의 상태 열거를 쉼표·가운뎃점으로 분리해 기능 9개로 확대했다. 명시적 세부 설명 문장을 기능명 목록과 구분하고 원문을 projectDescription에 보존하도록 수정했다. 다음 실행에서 실제 기능 4개와 상태 조건 보존을 확인했다.
   - 후속 코드로 읽기 전용 재검사한 결과는 통과하지만, 이 실행 자체는 실패이며 Kafka 저장 0건이다.
2. 두 번째 실행: `58876c24-dd74-4b7f-9ef3-d625499fb744`, `test-runs/live-job-application-tracker-20261025-005837Z`.
   - 동일 `GET /api/v1/applications`의 parameters·fieldMappings·errorCodes가 두 기능에서 다르게 선언돼 QA에서 충돌했다. 후속 PRD repair/resync도 실패했으며 Phase4에 도달하지 않았다.
   - 원인: Feature Spec preflight가 featureId로 시작하는 교차검사 메시지만 연결하면서, 전역 경로 충돌 메시지를 최초 repair 피드백에서 빠뜨렸다.
   - 같은 method/path를 선언한 모든 기능의 정확한 `/features/{i}/apiContract/{j}` 위치를 preflight 오류에 추가했다. 충돌을 무시하거나 마지막 선언으로 덮어쓰지 않고 기존 1회 targeted repair로 전달한다.

## 회귀 검증

- Chat/operation 관련 테스트: 53개 통과.
- Chat/의존 관계 수정 후 PostgreSQL 통합 포함 전체: 864개 통과.
- 공유 경로 preflight 추가 후 flat draft 테스트: 80개 통과.
- 변경 파일의 해당 Ruff 검사 통과.

## 최종 성공

세 번째 전체 실행 `858bc5c1-7c6e-4e07-a9d6-44c152d2dac9`가 새 Chat부터 Phase4까지 성공했다. 결과 경로는 `test-runs/live-job-application-tracker-20261025-010749Z`이다. 449.391초에 완료했으며 Feature Spec 최초 호출 1회·오류 0, QA/Phase3 추가 repair 없이 실제 Kafka/consumer/PostgreSQL 저장 399/399건을 확인했다.

최종 전체 테스트는 PostgreSQL 통합 포함 865개 통과다. 결과 기능은 P0 핵심 4개와 P1 인증 1개, API 18개, 테이블 5개, FK 4개다. 자동 구조 검사와 연결 데이터의 orphan 검사는 통과했다.

사람이 읽은 내용 검토에서는 메모 API primaryTable 오지정 4개, 일부 endpoint sessionPolicy 불일치, 자유 상태 수정과 제한 전이표 모순, PRD 면접/메모 상세 요구 부족, KPI/페르소나 정제 항목을 찾았다. 근거와 우선순위는 결과 폴더의 `QA_REPORT.md`, 읽기 쉬운 명세 요약은 `RESULT_SUMMARY.md`에 기록했다. 성공 이벤트와 실제 저장을 확인했다고 내용 완전성을 과장하지 않는다.
