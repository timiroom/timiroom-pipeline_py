# 계약·저장 복구 라이브 QA (2026-10-23)

## 실행 조건

- 주제: 초보 사용자의 반려식물 등록·물주기 기록·생육 사진·관리 일정·이상 징후 안내 서비스.
- 각 실행은 새 Chat 8턴으로 FormData를 만든 뒤 Phase1을 건너뛰지 않고 시작했다.
- 실제 모델 endpoint와 로컬 PostgreSQL 5433, Kafka 9092를 사용했다.
- 임시 Python 서버의 실제 HTTP API를 호출한 E2E이다. 브라우저 로그인부터 Java backend를 경유하는 전체 UI E2E와는 구분한다.
- backend 원자 저장과 복구는 별도의 실제 PostgreSQL 통합 테스트로 검증했다.

## 중간 실패와 적용한 보정

| 실행 | Pipeline ID | Chat 포함 시간 | 도달 단계 | 확인된 원인 |
| --- | --- | ---: | --- | --- |
| 1 | 21a92b33-606e-4c20-acd4-d6c23125f328 | 300.441초 | Phase3 | GET requestSchema=null 및 JSON pointer가 endpoint 경로처럼 연결된 repair 피드백 |
| 2 | 3582a5e4-d5bc-4e3b-bd69-fc9336fb12cd | 267.566초 | Phase3 | GET 본문, 204 응답 본문, path/body 매핑 오류가 원본 Registry에 남아 downstream repair 후 다시 복사됨 |
| 3 | 6dec139d-58c0-4e08-8858-537d4cd6f7c7 | 293.838초 | Phase3 | 계약 밖 별칭 테이블 symptom_guides의 featureId 부재, growth_photos.checksum의 필수 입력 출처 부재 |
| 4 | ea5f5c99-a329-4b9b-af80-31f2f567ed78 | 200.804초 | Feature Spec | 최초 JSON 구문 오류, 재시도에서 parameters를 표준 배열 대신 path/query 객체로 반환 |
| 5 | 1ee2efa2-19c6-4fed-a3bc-efc5e506c53a | 183.691초 | Feature Spec | states 형식 오류를 고치라는 요청에 다른 featureId의 patch를 반환해 범위 검사에서 거부 |
| 6 | 7f681923-c20b-49b8-9efa-fd95ebe8c6a2 | 320.546초 | Feature Spec | upstream 502 Codex request failed 반복, Feature Spec 두 호출 모두 InternalServerError |

실행 1의 shape 오류를 고친 뒤 기존 결과를 재검사하니 추가 의미 오류 32개가 드러났다. 이 재검사는 새 E2E 성공으로 계산하지 않았고 Kafka에 재발행하지 않았다.

실행 2에서는 한 번의 API repair가 원본의 잘못된 계약을 다시 투영해 문제가 반복됐다. 따라서 HTTP 본문·응답·필드 매핑 검사도 Feature Spec 생성 경계로 올렸다. 실행 3에서는 기능명세의 파라미터 형식 오류 12개를 한 번의 targeted patch로 해결했고, 후속 API/DB 생성까지 진행했다.

실행 3에 남은 두 원인은 기존 FK 참조 대상 탐색 실패가 아니었다. 참조 대상과 Registry 계약 커버리지는 맞았지만, DBA가 Registry에 없는 테이블·필수 컬럼을 추가했다. 완전한 필드 계약이 있는 최초 생성의 manager/worker 범위를 제한했으며, 기존 문서 repair나 legacy 입력은 변경하지 않았다. 제외된 worker 컬럼은 경고에 기록한다.

각 실패 실행의 실제 명세서와 terminal event는 다음 폴더에 보존했다.

- `test-runs/live-plant-care-20261023-012205`
- `test-runs/live-plant-care-20261023-013202`
- `test-runs/live-plant-care-20261023-014003`
- `test-runs/live-plant-care-20261023-014858`
- `test-runs/live-plant-care-20261023-015543`
- `test-runs/live-plant-care-20261023-020121`

실행 4는 DBA/API를 호출하지 않았다. 유효하지 않은 JSON의 부분 복구는 추가하지 않았다. 대신 유효한 JSON의 명확한 location/name 파라미터 map만 이름·위치·스키마·필수 여부를 보존하여 배열로 바꾸고, 변환 위치를 generationDiagnostics에 기록하도록 보강했다. 모호한 map은 여전히 차단한다.

실행 5 이후에는 repair 피드백에 실제 수정 가능한 featureId/sourceFeatureId/name을 명시했다. 배열 위치는 supporting 기능을 포함한 원본 기준이라는 점도 전달한다. 범위 밖 patch 거부 규칙은 유지했다. 인증 후처리가 명시된 logout 입력 계약을 삭제하지 않는 보정도 포함했다.

## 마지막 실행 판정

실행 6의 Chat 포함 시간은 320.546초였다. Search와 PRD에도 upstream 502가 관찰됐고 Feature Spec 두 호출은 응답 본문 없이 InternalServerError로 실패했다. 따라서 DBA/API는 실행하지 않았고, Phase4 Kafka 발행과 결과 저장도 실행하지 않았다. 해당 pipelineId의 실제 문서 청크는 0건이다.

Phase4까지 정상 완료했다는 결론을 내릴 수 없다. 프롬프트 충돌 점검 결과는 `docs/agent-prompt-audit-20261023.md`에 별도로 기록했다. 연결 장애와 프롬프트 문제는 별도 원인이며, 마지막 502를 프롬프트 변경만으로 해결했다고 주장하지 않는다.

## 저장 복구 검증

- 실제 PostgreSQL에서 manifest를 같은 내용으로 재등록할 수 있고, 다른 청크 목록으로 덮지 못함을 확인했다.
- consumer 트랜잭션이 커밋되기 전에는 완료로 인정하지 않았다. 일부 청크만 커밋된 상태도 pending을 유지했다.
- verifier 인스턴스를 새로 만들어도 영속 manifest와 실제 저장 key를 비교해 stored로 확인했다.
- Java의 마지막 알림 INSERT에 강제 DB 오류를 일으켜 문서·요구사항 상태·알림·실행 완료가 모두 rollback됨을 확인했다.
- 결과 checkpoint는 보존됐고, AI 재호출 없이 기존 결과 저장을 복구했다.
- 동시 완료 수신 두 건이 문서 묶음과 알림을 한 번만 저장했다.
- 완료 결과 조회는 소유자만 허용했다.

## 회귀 테스트

- Python: 실제 PostgreSQL opt-in 테스트 포함 627개 통과.
- backend: 285개 통과, 기존 조건부 제외 29개, 실패·오류 0. 새 PostgreSQL 통합 테스트 5개 포함.
- frontend: 32개 통과, lint 및 production build 성공.

단위·통합 테스트 통과는 새 라이브 생성의 품질이나 Phase4 저장 성공을 자동으로 의미하지 않는다. 최종 라이브 결과는 별도로 기록한다.
