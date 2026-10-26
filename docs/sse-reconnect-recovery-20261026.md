# 생성 진행 화면의 연결 끊김·무한 재연결 보완

## 확인한 원인

1. 프론트가 재전송된 과거 progress 이벤트를 받을 때도 재시도 횟수를 0으로 초기화했다. 연결 → 과거 이벤트 재생 → 끊김이 반복되면 최대 6회 제한에 도달하지 못했다.
2. ProgressScreen의 SSE effect가 부모의 inline onComplete에 의존했다. 부모가 다시 렌더링되면 연결과 진행 상태를 다시 만들 수 있었다.
3. 기존 상태 조회는 STORAGE_PENDING에서만 실행됐고, FAILED를 만나면 화면에 실패를 전달하지 않은 채 멈췄다. 조회 실패·결과 없는 COMPLETED·저장 대기에도 종료 한도가 없었다.
4. 파이프라인은 LLM 처리 중 15초 단위 heartbeat가 없었다. 알려지지 않은 pipelineId도 빈 구독 상태로 기다렸다.
5. 백엔드 상태 조회는 DB의 RUNNING만 반환했다. SSE 완료·실패를 놓친 경우 이를 회복하지 못했고, FAILED의 사유·재시작 가능 여부도 화면에 전달하지 않았다.

## 수정

### Frontend

- 새 진행 이벤트와 replay를 구분하는 재연결 예산을 추가했다. 기본 최대 6회/복구 구간 60초이며 동일 replay는 예산을 초기화하지 않는다.
- 완료 콜백을 ref로 분리해 부모 렌더링만으로 SSE를 다시 만들지 않는다. 수동 재연결 시 같은 실행의 진행 상태를 보존한다.
- 45초간 신호가 없으면 연결 복구를 시도한다. heartbeat는 생성 진행률을 올리는 이벤트로 사용하지 않는다.
- 실행 상태 조회를 연결 오류에도 사용한다. 요청마다 10초 timeout을 두며 자동 조회는 최대 24회 후 멈춰 사용자 동작을 기다린다.
- FAILED는 사유와 다시 시작 동작으로, 저장 대기는 재조회 동작으로, 불확실한 연결은 다시 연결 동작으로 구분한다. 재연결/재조회는 새로운 생성 API를 호출하지 않는다.
- 연결 확인·저장 확인 상태에서 단계 카드가 계속 실행 중으로 표시되지 않도록 구분했다.
- 완료·실패를 확인하면 예약된 재연결이나 늦은 응답이 화면을 다시 바꾸지 못하게 정리했다.

### Backend

- 파이프라인의 상태 snapshot을 5초 제한으로 조회해 놓친 완료·실패를 기존 처리기로 반영한다. 요청자의 권한을 먼저 확인한다.
- RUNNING을 찾지 못했거나 연결이 불확실한 경우 FAILED로 단정하거나 재생성을 허용하지 않는다.
- 상태 응답에 FAILED 사유와 retrySafe를 추가했다. STORAGE_PENDING은 재생성하지 않고 해당 receipt만 즉시 재확인할 수 있다.
- 기존 receipt 없는 COMPLETED 실행도 저장 artifact에서 결과를 복원할 수 있게 했다.
- 서버 감시 복구 기본 주기를 5분에서 30초로 줄이고, 진행 Flux의 무응답 제한을 45초로 설정했다.
- SSE 응답에 Cache-Control: no-cache, no-transform 및 X-Accel-Buffering: no를 설정해 proxy 버퍼링을 방지한다.

### Pipeline

- 실행 대기 이벤트를 등록한 후 background task를 시작하고 task의 강한 참조를 유지한다.
- 진행 이벤트가 없어도 15초마다 heartbeat를 보낸다. 기존 완료·실패 이벤트 보관과 다중 구독은 유지한다.
- 없는 실행은 connection/unavailable 이벤트로 종료하며 생성 실패라고 거짓으로 기록하지 않는다.
- 읽기 전용 상태 snapshot endpoint를 추가했다. RUNNING, COMPLETED, FAILED, UNKNOWN을 구분하고 늦게 도착한 조회에서 terminal event를 복구한다.

## 회귀 검증

- Frontend: Node 테스트 46개 통과. replay에 의한 무한 재연결, 복구 시간 제한, FAILED 통지, 저장 대기 조회 상한, 결과 없는 완료 상태, unmount 뒤 응답을 포함한다. lint 및 production build 검사도 수행했다.
- Pipeline: 전체 912개 통과, DB 통합을 opt-in으로 사용하는 기존 2개 제외. heartbeat 중 진행률 불변, 없는 실행의 즉시 종료, 놓친 terminal snapshot 및 HTTP 상태 endpoint를 검증했다. Ruff/compileall 통과.
- Backend: 전체 빌드에서 341건 중 실패·오류 0, 기존 환경 조건부 34건 제외. 그 뒤 추가한 proxy 응답 헤더를 포함한 최종 관련 회귀 22건(상태 흐름 19, 실제 HTTP/SSE 2, controller 1)이 모두 통과했고 최신 실행 JAR도 다시 생성했다. 실제 로컬 HTTP fixture를 통해 heartbeat/complete SSE와 실패 snapshot 파싱을 검사했다.

## 반영 상태

작업 시작 시 원격 develop을 조회하고, backend b8e43f4 / frontend 85a22ec / pipeline 0a32988까지 fast-forward했다. 기존 권한 검사를 보존했다. 수정·검증 후 사용자의 요청에 따라 세 저장소의 develop에 fix 커밋으로 반영한다. 원격 push와 운영 배포 완료는 구분하며, 이 문서는 운영 사이트 재검증을 완료했다는 주장이 아니다.

배포 순서는 pipeline의 heartbeat/state endpoint → backend의 복구/버퍼링 방지 → frontend의 복구 UI가 권장된다. 실행 상태 자체는 파이프라인 프로세스의 메모리 snapshot이므로 프로세스 재시작이나 다른 worker에서 실행을 찾지 못할 수 있다. 이 경우 무한히 생성 중으로 표시하거나 중복 생성을 자동 시작하는 대신 연결 확인이 필요하다고 표시한다. 영속 저장 완료는 기존 DB receipt를 기준으로 확인한다.

이번 검증은 연결 복구와 상태 계약을 대상으로 하며, AI를 호출하는 새 Chat→Phase4 라이브 생성이나 운영 장애 로그 확인을 대신하지 않는다.
