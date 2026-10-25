# 우선순위 5개 수정 내역

## 적용 범위

2026-10-22 Phase2/Phase3 검토에서 정한 우선사항 5개를 먼저 반영했다. API fieldMappings 정밀 검사, FK 참조 컬럼/타입 검증, 파라미터 보존, QA 검사기 통합, repair 예산 통합, SSE 개선 등 추가 항목은 이번 범위에 포함하지 않았다.

## 1. Feature Spec 계약 검증

이전: JSON 파싱/모델 호출 실패를 최소 기능·추측 경로·추측 테이블로 대체했고, 그 결과를 정상 계약처럼 downstream에 전달할 수 있었다.

변경: 실제 모델 응답을 입력 경계에서 검사한다. 의미 있는 고유 featureId, 설명/요구사항, method/path 및 응답 schema, primaryTable과 typed DB fields를 확인한다. PRD core 누락도 검사한다. 실패하면 기존 산출물을 보존하고 구조화된 생성 blocker로 반환하여 DBA/API 호출을 차단한다.

유효한 계약을 받은 뒤에도 추측한 POST나 하위 기능을 자동으로 보충하지 않는다. 기존 helper는 호환 테스트를 위해 남아 있지만 실제 Feature Spec execute에서는 사용하지 않는다. 추가 기능은 LLM이 명시적 계약을 작성해야 발행된다. 모델 호출을 무한 재시도하거나 실패를 성공으로 처리하지 않는다.

파일: `phase2/contract_guard.py`, `phase2/agents/feature_spec_agent.py`, `phase2/orchestration_graph.py`.

## 2. PRD 변경 감지

이전: name/priority/actions/API·DB 계약만 주로 비교하여 description/requirements/권한/수용조건만 바뀌면 변경 기능이 없다고 판단했다.

변경: 기능 객체의 내용 전체를 canonical JSON으로 비교한다. ID 별칭과 문자열 공백을 정규화하고 설명·요구사항·권한·상태 등의 변경을 감지한다. 기능 추가/삭제와 부모 기능의 하위 기능 영향 범위도 반영한다.

파일: `phase2/orchestration_graph.py`.

## 3. PRD repair 후 공통 재동기화

이전: rollback 경로와 Phase3 repair 경로의 동작이 달랐다. Phase3 PRD repair 이후 Feature Spec/API/DB가 이전 계약에 남을 수 있었다.

변경: 초기 rollback, Phase2 QA repair, Phase3 repair가 공통 delta resync를 사용한다. PRD를 먼저 수정하고 변경 기능의 Feature Spec을 반영한 뒤 실제 계약이 바뀐 도메인만 targeted repair한다. 기존 API/DB 산출물은 PRD agent가 반환한 빈 값으로 덮지 않는다. Feature Spec 및 API/DB별 timeout 실패는 기존 결과를 보존한 blocker로 전달한다.

삭제 기능과 자식 기능을 이전 Registry에서 다시 공급하지 않도록 필터링한다. 신규 자식은 반영하고 변경하지 않은 형제 기능은 유지한다. repair를 지원하지 않는 agent는 전체 execute로 대체하지 않는다.

파일: `phase2/orchestration_graph.py`.

## 4. 테이블 정규화 멱등성

이전: 정상 테이블명에서도 숫자를 제거했다. 계약 테이블 보충과 결합하면 원래 테이블이 반복 추가되고 `_2/_3` 테이블이 누적될 수 있었다.

변경: 유효한 식별자의 숫자를 유지한다. `oauth2_tokens`, `assets_2` 등을 반복 정규화해도 이름과 테이블 수가 유지된다. 기존 명시적 계약 alias 정규화는 유지한다. 임시 hash 리소스는 앞 단계 계약 검사에서 차단한다.

파일: `phase2/agents/dba_agent.py`.

## 5. targeted repair 범위 제한

이전: prompt의 '문제 부분만 수정' 지시 외에 실제 patch 대상을 제한하는 장치가 부족했다. 정상 endpoint/table/feature와 기존 컬럼이 함께 바뀔 수 있었다.

변경:

- 피드백의 명시적 artifact key와 featureId에서 허용 대상을 계산한다. 대상을 판단할 수 없으면 호출하지 않고 rejected_scope로 반환한다.
- API: 허용 endpoint만 적용하고, 후처리 뒤에도 대상 밖 endpoint는 원본 객체로 복원한다.
- DBA: 허용 테이블만 수정하며 컬럼은 이름별 부분 병합한다. table.column이 특정되면 해당 컬럼만 적용한다. 명시하지 않은 컬럼/index와 대상 밖 테이블은 보존한다. 관계도 대상 테이블과 관련된 범위로 제한한다.
- PRD: 허용된 섹션만 반영한다. coreFeatures는 항목별 병합하고 featureId가 특정되면 다른 기능 patch는 무시한다. 원본이 없으면 전체 생성을 차단한다.
- PM: 대상 기능만 수정하고, 여러 삭제는 최초 목록 인덱스를 기준으로 계산하여 위치 이동 오류를 막는다. 관련 없는 DBA/API instruction 변경은 반영하지 않는다.
- baseHash를 prompt에 전달하고 응답에 다른 hash가 있으면 거절한다. 기존 patch 응답 호환을 위해 hash 생략은 허용하므로 엄격한 외부 동시성 제어/CAS까지 구현한 것은 아니다.

파일: `phase2/repair_scope.py`, `phase2/agents/api_agent.py`, `phase2/agents/dba_agent.py`, `phase2/agents/prd_agent.py`, `phase2/agents/pm_agent.py`.

## 검증

`tests/test_priority_contract_repairs.py`에 실패 차단, 정상 계약 발행, 요구사항 변경, 기능/자식 삭제, 숫자 테이블명 멱등성, patch 범위, 컬럼 보존, stale hash, PM 복수 삭제, PRD resync와 timeout 회귀를 추가했다. 이전의 무조건 fallback/전체 재생성을 기대하던 관련 테스트는 새 정책에 맞게 변경했다.

라이브 E2E와 Kafka 실제 저장은 이번 작업에서 실행하지 않았다. 단위/회귀 통과가 모든 생성 주제의 Phase4 도달을 보장하지는 않는다.
