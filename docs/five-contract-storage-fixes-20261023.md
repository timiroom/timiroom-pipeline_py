# 계약 검증·저장 복구 5개 보강 (2026-10-23)

## 1. Feature Spec 공유 DB/API 계약 검증

변경 전에는 개별 기능의 계약 형식이 맞더라도 다른 기능이 선언한 참조 컬럼이 실제 고유키인지, 같은 컬럼의 타입과 필수 조건이 서로 일치하는지 downstream 생성 전에 확인하지 못했다.

`phase2/registry_contract_guard.py`에서 전체 Registry의 필드와 고유키를 먼저 모은 뒤 FK를 검증한다. 참조 대상의 존재·타입·PK/UNIQUE 여부와 공유 필드 충돌을 검사한다. API의 HTTP 본문 규칙, 204 응답, 필드 매핑과 상태 변경 대상도 동일한 물리 필드 목록을 사용해 검사한다. 오류에는 기능 및 operation의 JSON 위치를 남겨 한 번의 제한된 patch 대상으로 전달한다.

이 검사는 LLM 최초 결과와 최종 정규화 결과 양쪽에 적용된다. 잘못된 FK를 통과시키려고 반복 기록 컬럼에 임의 UNIQUE를 추가하지 않는다.

## 2. 인증 세션 저장소 단일화

변경 전에는 모델이 `sessions`를 설계해도 후처리에서 별도의 `auth_sessions`를 추가할 수 있었다.

명시된 feature/operation의 `sessionPolicy`와 선언된 토큰 해시·만료 필드로 저장소를 찾고 재사용한다. 여러 저장소나 전략이 충돌하면 조용히 하나로 덮지 않고 계약 결함으로 남긴다. 사용자 ID 타입도 Registry에서 가져와 UUID를 BIGINT로 바꾸지 않는다. 빈 feature 정책이 명시적인 operation 정책을 가리지 않게 했다.

## 3. API 설명과 상태 변경 효과 분리

변경 전에는 worker가 자연어 `effects` 배열이나 계약에 없던 상태 변경을 추가할 수 있었다.

`effects`는 객체 배열이며 실제 table/column/kind를 검증한다. 행 삽입·삭제와 컬럼 변경을 구분하고 명시적인 상태 전이를 지원한다. 최종 API에는 Registry에 선언된 효과만 복사한다. worker의 임의 효과는 경고로 기록하고 채택하지 않는다. 자연어 설명은 `transactionRules`에 유지한다.

## 4. 저장 확인 대기·재시작 후 복구

변경 전에는 Kafka ACK 이후 소비자 저장이 늦으면 대기 결과를 재확인할 영속 상태가 부족했다.

Python의 `pipeline_storage_receipts`에 예상 청크의 정확한 source key 목록과 Kafka 발행 여부를 보관한다. 동일 pipelineId로 다른 manifest를 덮어쓰지 못하게 한다. 확인 API는 메모리 ACK가 아니라 DB에 커밋된 청크를 확인한다.

backend는 결과 payload를 `pipeline_result_receipt`에 보관하고 실행을 `STORAGE_PENDING`으로 유지한다. 기본 5초 간격의 복구 작업이 Python 저장 상태를 확인한 뒤 기존 결과를 저장한다. 새 모델 호출이나 Kafka 재발행으로 복구하지 않는다. frontend도 저장 대기 시 본인 실행 상태를 조회해 완료 화면으로 전환한다.

- 내부 저장 상태: `GET /api/v1/orchestration/storage/{pipeline_id}`
- 본인 실행 결과: `GET /api/v1/pipeline/state/{pipelineId}`

## 5. backend 완료 처리의 원자성

변경 전에는 문서 저장 오류를 삼키거나 실행 완료 처리와 문서 저장이 분리되어 일부 결과만 남을 위험이 있었다.

첫 번째 트랜잭션은 결과와 저장 대기 상태를 checkpoint로 남긴다. 두 번째 트랜잭션은 실행 행을 잠근 뒤 문서 묶음·요구사항 완료 상태·완료 알림·실행 완료·receipt 완료를 함께 커밋한다. 마지막 알림 저장이 실패해도 앞서 저장한 문서와 상태가 모두 rollback되고, 첫 checkpoint는 남아 재시도할 수 있다. 여러 수신자가 같은 완료를 받아도 하나만 반영한다.

Kafka/벡터 저장과 Java 업무 DB를 하나의 분산 트랜잭션으로 묶었다는 의미는 아니다. 외부 저장 확인 이후 Java 내부 결과 반영을 원자화한 것이다.

## 추가 회귀 보정

- GET의 `requestSchema: null`은 본문 없음으로 정규화한다. 실제 비어 있지 않은 본문 오류는 삭제하지 않고 검증한다.
- shape 오류의 JSON pointer를 endpoint 경로와 분리한다. repair는 기존 endpoint 또는 Registry 계약 경로만 대상으로 삼는다.
- 새 migration의 PostgreSQL 전용 상태 제약 변경은 Java migration으로 분리해 H2 테스트와 호환시켰다.
- frontend 상태 조회 중 일시 오류가 나도 재생성하지 않으며, unmount 이후 도착한 응답은 화면을 변경하지 않는다.
- 완전한 DB 계약이 있는 최초 생성에서는 manager가 Registry 밖 별칭 테이블을 추가하지 못하게 하고, worker의 계약 밖 필수 컬럼을 채택하지 않는다. 제외 항목은 generationWarnings에 남긴다. 기존 문서 repair나 필드 계약이 없는 legacy 생성에는 이 범위 제한을 적용하지 않는다.
- 유효한 JSON의 파라미터 location/name map은 의미를 보존하여 표준 배열로 변환하고 위치를 diagnostics에 기록한다. JSON 구문 복구나 모호한 map 추측은 하지 않는다.
- Feature Spec repair 피드백에는 배열 위치뿐 아니라 실제 허용된 featureId/sourceFeatureId를 전달한다. 인증 후처리도 명시적인 logout 입력 계약을 삭제하지 않는다.

## 검증 범위

- Python 전체 627개 통과. 실제 PostgreSQL manifest 검증을 opt-in으로 포함한 실행이다.
- backend 전체 314개 중 285개 통과, 기존 조건부 테스트 29개 제외, 실패·오류 0. 실제 PostgreSQL 임시 schema의 5개 저장 통합 테스트 포함: 마지막 쓰기 실패 rollback, 지연 소비 복구, 중복 완료 경합, 변경 payload 거부, 기존 상태 제약 migration.
- frontend 32개 테스트, lint, production build 성공.
- 실제 모델·Chat·Phase1·Phase2·Phase3·Kafka/DB 경로를 라이브 테스트로 확인하며, 각 실행의 terminal event와 명세서는 `test-runs`에 보관한다.

라이브 실패를 단위 테스트 성공이나 저장 결과 재검사 성공으로 대체해 표현하지 않는다. 실제 생성 E2E의 최종 도달 단계는 해당 실행의 QA 보고서를 기준으로 판단한다. backend/프론트 운영 프로세스는 이 작업에서 재배포하지 않았다.

## 적용 순서

1. Python DB에 `python migrate_db.py`를 실행해 저장 manifest 테이블을 만든다. 이번 로컬 DB에는 실행 완료했다.
2. Python 서비스를 새 코드로 실행한다.
3. backend를 재시작해 Flyway migration과 저장 복구 작업을 적용한다.
4. frontend를 새 코드로 실행한다.

검증용 PostgreSQL schema만 정리하며 기존 프로젝트 산출물은 삭제하지 않는다. migration rollback을 이유로 저장된 receipt나 문서를 역삭제하지 않는다.
