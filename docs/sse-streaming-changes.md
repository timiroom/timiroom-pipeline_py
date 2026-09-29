# Chat 및 Pipeline SSE 스트리밍 개선

## 변경 배경

기존에는 채팅 응답과 파이프라인 진행 상태가 화면에 안정적으로 전달되지 않는 문제가 있었다.

- 파이프라인 진행 이벤트가 프론트에 늦게 연결되면 초기 이벤트를 놓칠 수 있었다.
- 파이프라인이 완료되거나 실패한 뒤 SSE 구독이 연결되면 terminal event를 받지 못하고 계속 대기할 수 있었다.
- 프론트 진행 화면이 이전 파이프라인 단계인 `DRAFT`, `COLLAB` 기준으로 동작했다.
- `Feature Spec`, targeted resync, QA repair, Phase3, Phase4 이벤트가 화면 단계와 연결되지 않았다.
- SSE `error` 이벤트가 네이티브 연결 오류인지 서버가 전달한 실제 파이프라인 실패인지 구분되지 않았다.
- 서버는 실제 blocker 상세 정보를 전달했지만 프론트가 일반 오류 문구만 표시했다.
- 채팅 요청은 Spring의 30초 timeout과 Python/OpenAI 120초 timeout이 맞지 않아 정상 응답도 중간에 끊길 수 있었다.
- OpenAI SDK 자동 재시도와 애플리케이션 재시도가 중복되어 채팅 응답이 불필요하게 늦어질 수 있었다.

## 기존 동작

```text
Pipeline 실행
  -> progress event를 queue에 전달
  -> 늦게 연결된 SSE 구독자는 초기 이벤트를 놓칠 수 있음
  -> 완료/실패 event를 받지 못하면 프론트가 대기 상태 유지

Frontend
  -> EventSource로 일부 단계만 매핑
  -> 실패 시 "관리자 검토가 필요합니다"만 표시
  -> 실제 validation blocker, DB/API/PRD 원인은 표시하지 않음

Chat
  -> Spring RagPipelineClient가 30초 timeout
  -> Python OpenAI 호출은 최대 120초 및 재시도 수행
  -> SDK 재시도와 애플리케이션 재시도가 중복될 수 있음
```

## 변경 후 구조

```text
PipelineProgressService
  -> progress event를 buffer와 active queue에 동시에 저장
  -> complete/error event를 terminal event로 TTL 동안 보관
  -> 늦게 연결된 구독자에게 terminal event 즉시 재생
  -> terminal event 뒤 __done__ 신호로 스트림 종료

Backend SSE
  -> Phase1 / Search / PM / PRD / Feature Spec
  -> DBA/API 병렬 설계
  -> QA / Phase3 / Phase4
  -> repair/resync 이벤트까지 동일 스트림으로 전달

Frontend EventSource
  -> 서버 step을 현재 파이프라인 단계로 정규화
  -> 완료 단계, 실행 중 단계, 실패 단계를 구분
  -> 최근 이벤트와 경과 시간 표시
  -> blocker 상세 내용을 실패 화면에 표시
```

## 서버 SSE 개선

### 진행 이벤트 보존

`PipelineProgressService.send()`가 이벤트를 active queue뿐 아니라 pipeline별 buffer에도 저장한다. 따라서 프론트가 SSE를 늦게 구독해도 Phase1부터 현재 단계까지의 이벤트를 재생할 수 있다.

### 완료/실패 이벤트 보존

`complete()`와 `error()`는 terminal event를 별도 저장한다.

- 늦게 연결된 클라이언트도 완료 결과 또는 실패 원인을 받을 수 있다.
- terminal event는 TTL 동안 보관한다.
- terminal event 이후 `__done__`을 전달해 구독을 종료한다.
- 대형 완료 결과가 progress buffer를 밀어내 terminal event가 사라지는 문제를 분리했다.

### 실패 정보 보존

Phase3 실패 시 다음 정보를 SSE error payload에 포함한다.

- `validationErrors`
- `validationMessage`
- `validationBlockers`
- `resolvedBlockers`
- `validationBlockerComparison`
- `blockerDetails`
- DB/API/PRD별 blocking issues와 warnings

## 프론트 진행 화면 개선

### 현재 파이프라인 단계 반영

```text
RAG 준비
  -> 시장 조사
  -> 기능 분석
  -> PRD 작성
  -> 기능명세
  -> DB · API 설계
  -> QA 검수
  -> 최종 검증
  -> 결과 저장
```

DBA/API 단계는 `DBA · API 병렬`로 표시한다.

### SSE 이벤트 매핑

```text
PHASE1_START / PHASE1_DONE / PHASE1_SKIP -> PHASE1
SEARCH -> SEARCH
PM / PM_REPAIR -> PM
PRD / PRD_REPAIR / PRD_ROLLBACK -> PRD
FEATURE_SPEC / FEATURE_SPEC_RESYNC -> FEATURE_SPEC
DBA_API / PHASE2_REPAIR / PHASE2_TARGETED_RESYNC -> DBA_API
QA / QA_RETRY / QA_REPAIR -> QA
PHASE3 -> PHASE3
PHASE4 -> PHASE4
```

### 진행 상태 표시

- 현재 단계 카드에 완료/실행 중/대기 상태 표시
- 실패한 단계는 `실패` 상태로 표시
- 파이프라인 시작 후 경과 시간 표시
- 최근 실행 이벤트 8개 표시
- 좁은 화면에서는 다이어그램이 깨지지 않도록 가로 스크롤 적용
- 기존 4단계/5단계 행 분리 오류 수정
- 이전의 보라색 상태 안내 바 제거

### 실패 원인 표시

기존에는 서버의 error payload에서 `message`만 표시했다. 현재는 다음 데이터를 조합해 화면에 실제 blocker를 표시한다.

```text
validationBlockers
blockerDetails
qa.blockingIssues.db
qa.blockingIssues.api
qa.blockingIssues.prd
```

이제 사용자는 단순히 “생성 실패”가 아니라 어떤 DB FK, API endpoint, PRD 정합성 문제가 남았는지 확인할 수 있다.

## Chat 응답 개선

### Timeout 정렬

Spring의 채팅 timeout을 환경변수 기반 180초로 변경했다.

```yaml
rag-pipeline:
  chat-timeout-seconds: ${RAG_PIPELINE_CHAT_TIMEOUT_SECONDS:180}
```

기존 Spring 30초 제한 때문에 Python/OpenAI 응답이 정상적으로 진행 중이어도 프론트에는 일시적인 오류로 표시될 수 있었던 문제를 해결했다.

### 중복 재시도 제거

채팅 요청에서는 OpenAI SDK 자동 재시도를 끄고 애플리케이션에서 최대 1회만 명시적으로 재시도한다.

또한 일반 채팅 응답에서 다음 보강 호출을 제거했다.

- 잘못된 질문 응답의 추가 LLM 재생성
- suggestions 부족 시 추가 LLM 생성

출력이 유효하지 않은 경우 기존에 준비된 결정론적 fallback을 사용해 사용자 응답을 빠르게 반환한다.

## 검증 결과

- Python phase2/phase3 테스트: `57 passed`
- 프론트 `ProjectChatWizard.jsx`, `CreateProjectWizard.jsx` ESLint 통과
- `PipelineProgressService`의 terminal event 보존 구조 확인
- DB FK 추론 테스트 추가: `buyer_id`, `seller_id`가 별도 테이블이 없을 때 `users`를 참조

## 관련 코드

- `phase2/sse_service.py`
- `routers/orchestration.py`
- `src/main/java/com/timiroom/infra/ragpipeline/RagPipelineClient.java`
- `timiroom-frontend/src/components/dashboard/ProjectChatWizard.jsx`
- `timiroom-frontend/src/components/dashboard/CreateProjectWizard.jsx`
- `timiroom-frontend/src/lib/chatApi.js`
