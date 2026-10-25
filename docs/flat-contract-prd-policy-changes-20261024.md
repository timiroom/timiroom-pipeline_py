# Feature Spec 계약 조립 및 PRD 정책 수정 (2026-10-24)

## 적용 범위

앞선 1~6번 수정 방향과 KPI·페르소나 fallback 원인 분석을 함께 적용했다. 변경 대상은 Python 생성·조립·QA·repair와 OpenAPI 변환이며, Java/프론트엔드의 저장 형식은 변경하지 않았다.

## 1. Feature Spec 출력 구조

기존에는 LLM이 모든 기능의 중첩 requestSchema/responseSchema와 반복 DB 필드를 한 JSON에 직접 작성했다. 유효 JSON을 출력하라는 지시와 json_object 모드만으로 실제 구문 오류를 막지 못했다.

이제 실제 FeatureSpecAgent는 typed DraftDocument를 사용한다.

```text
version: contract-draft-v1
features: 설명·요구사항·기능 ID·작업·테이블/DTO 참조
tables: 테이블별 물리 컬럼·PK·FK·default·고유 제약을 한 번 정의
schemas: 이름 있는 DTO별 평탄한 nodes 배열
```

DTO 노드는 id/parent/name/kind로 구조를 표현한다. 코드가 객체·배열·필수값·enum·nullable·파일 binary·참조·union을 JSON Schema로 조립한다. 잘못된 부모·고아 노드·순환 참조·중복·비정상 범위는 추측하지 않고 오류로 반환한다.

```text
LLM typed draft
→ strict JSON / Pydantic 검사
→ compile_document
→ 기존 GeneratedFeature 및 공유 계약 검사
→ 기존 Feature Registry 정규화와 재검사
→ DBA/API 병렬 생성 및 QA
```

prompt와 provider schema는 동일한 DraftDocument 모델에서 나온다. 기존 canonical 형식의 테스트/호환 helper는 명시적으로 유지하되, 실제 FeatureSpecAgent는 draft=True를 사용한다. provider 모드가 유효 문법이나 의미를 보장한다고 설명하지 않는다.

## 2. 공통 DB 및 매핑

- 테이블·컬럼을 LLM이 기능마다 다시 선언하지 않는다. 각 기능은 dbTables와 primaryTable을 참조한다.
- 소비용 기존 dbContract에는 코드가 같은 공통 정의를 복사한다. 생성 원천은 하나지만 저장 형식은 기존과 같다.
- 명시된 FK 대상은 공통 tables 안에서만 찾아 의존성을 투영한다. 이름으로 새 테이블이나 FK를 추론하지 않는다.
- ownership은 실제 ownerEntity/ownerKey/access를 보존하며 고정 user_id를 강제하지 않는다.
- MappingDraft는 실제 스칼라 DTO 필드 하나와 테이블 컬럼 하나를 연결한다. column="id,name" 같은 묶음은 생성 경계에서 거부한다.
- union·nullable·binary와 구조화 파라미터가 OpenAPI 변환에서 유지되도록 보강했다.

## 3. 제한된 repair

- 최초 생성과 기존 1회 보정 예산은 유지한다. 기능마다 추가 생성 호출을 늘리지 않았다.
- 유효하게 파싱된 draft는 featureId·table name·DTO name allowlist로만 교체한다. 성공한 형제 기능과 공통 정의는 보존한다.
- 특정 apiContract 항목 오류이면 같은 기능 안의 정상 method/path까지 바꾸는 patch도 거부한다.
- 오류 pointer로 정확한 DTO 자산을 지정한다. 짧은 이름이 오류 코드의 문자열 일부와 우연히 일치해 수정 범위가 넓어지지 않게 했다.
- JSON 자체가 유효하지 않으면 부분 복구하지 않는다. 수정할 featureId를 추측할 수 없으므로 같은 최초 생성 단위에 대해 기존 1회 재시도만 허용한다.

## 4. KPI 정책

기존 활성 KPI 검사는 특정 한글 단어 포함 여부로 판정했으며, fallback KPI 7개 중 3개도 그 검사에 맞지 않았다. 최종 조립은 시간 단위 지표를 삭제하고 고정 7개로 다시 채웠다.

수정 후:

- dimension을 명시하고 측정 정의를 event/numerator/denominator/unit/window로 구조화한다.
- worker·조립·QA·targeted repair가 같은 검증 함수를 사용한다. 비율에는 분모가 필요하다.
- 초·분·원 등 단위를 임의로 배제하지 않는다. target의 숫자 기준값→목표값 형식은 검사한다.
- 제공된 외부 근거와 내부 운영 목표를 구분한다. 근거 없는 외부 주장을 내부 목표로 슬쩍 재라벨링하지 않는다.
- 개수를 채우는 기본 KPI는 제거했다. 생성 실패 항목은 제외하고 사유를 기록한다. KPI가 하나도 유효하지 않으면 QA에서 실패한다.
- 리텐션 후처리가 이미 명시된 측정 이벤트와 방법을 고정 이벤트로 덮어쓰지 않는다.

## 5. 페르소나 정책

기존에는 같은 연령·직업이면 다른 목표와 사용 흐름도 삭제하고 일반 템플릿으로 3개를 채웠다. 20~30대 안의 20대 초반도 문자열 불일치로 탈락할 수 있었다.

수정 후:

- PM 계획/입력의 실제 사용자 세그먼트 수에 맞춰 worker를 배정한다. 무조건 3명이나 가상의 운영자를 만들지 않는다.
- segmentId와 제공된 역할을 보존한다. 중복 키는 역할·목표·불편·사용 패턴의 정규화된 조합이며, 연령·직업이나 표시 이름만으로 삭제하지 않는다.
- 연령은 범위로 비교한다. 세부 연령대를 허용하고 실제 범위 밖이면 거부한다.
- 가상 사용자 모델은 assumption=true로 구분한다. 근거 없는 직업·연령을 사실로 표시하지 않는다.
- 요구된 세그먼트가 누락됐으면 expectedPersonaSegments와 대조해 QA에서 지적한다. 실제 두 역할이 필요한데 하나만 생성됐다고 통과시키지 않는다.
- 실패 후 가상 인물 템플릿으로 채우는 fallback은 제거했다.

이 중복 키는 결정론적 비교이며 모든 자연어 동의어를 의미적으로 판별하는 AI라는 뜻은 아니다.

## 6. 범위와 진단 이력

- PRD mvpScope.excluded에 PM scope.out/priorities.wont와 사용자 제외 기능을 반영한다. 기존 기능 목록에 이름이 없다는 이유로 제외 범위를 지우지 않는다.
- KPI/페르소나 탈락 사유는 qualityMetadata.itemDiagnostics에 항목과 단계별로 보존한다.
- 현재 문서와 진단 이력을 분리해 검사한다. 과거 오류 메시지의 unknown 같은 단어만으로 고친 문서가 다시 placeholder blocker가 되지 않는다.
- 유효한 targeted repair 후 관련 진단은 resolved_by_targeted_repair로 표시한다. 무효한 측정/페르소나 patch는 원본을 보존하고 거부한다.
- legacy PRD 프롬프트의 7개 KPI/3개 페르소나 강제 지시와 내부 목표 금지 지시도 정리했다.

## 7. 추가 보완: repair 안전성과 연령 형식

추가 코드 점검에서 확인된 네 항목을 순서대로 보완했다.

1. 잘못된 응답의 수정 범위 계산: 숫자형 parameters, 잘못된 apiContract/dbTables, null 테이블/DTO 항목을 순회하다 TypeError가 발생하던 경로를 제거했다. 원본 데이터 자체를 유효하다고 바꾸지 않고, 고칠 수 있으면 기존 1회 repair를 사용한다. 안정적인 범위를 정할 수 없는 배열 구조는 명시적인 FEATURE_SPEC_REPAIR_SCOPE 오류로 반환한다.
2. 누락 핵심 기능의 새 자원: missingSourceFeatureIds에 지정된 새 core를 같은 patch에 추가할 때 그 기능의 dbTables/API 타입이 참조하는 새 정의와 명시적인 FK/schema-ref 의존 정의만 허용한다. 기존 정의 덮어쓰기와 무관한 자원 추가는 계속 거부하며, 병합 후 전체 계약 검증을 다시 적용한다.
3. endpoint 단위 수정 보호: 응답 스키마 누락, 경로/메서드 오류, operation 소유자 오류에 정확한 /features/N/apiContract/M 위치를 남긴다. operationIndices로 오류가 난 슬롯만 교체하고 정상 endpoint의 개수·순서·내용을 보존한다. 정상 endpoint가 참조하는 DTO와 중첩 ref 정의를 우회해서 변경하는 것도 거부한다.
4. 페르소나 연령: 20대~30대, 20~30대, 20세~30세, 만 25세 및 초반/중반/후반 표기를 공통 범위로 해석한다. 양쪽 단위를 섞거나 역순인 범위는 거부하고, 사용자 입력의 범위 비교에도 같은 파서를 적용한다.

전체 생성이나 repair 호출 예산을 늘리지 않았고, QA hard gate도 완화하지 않았다.

## 검증과 한계

새 테스트는 typed draft의 기존 계약 변환, FK 의존성, scalar 매핑, DTO 그래프/union/파일, scoped repair와 정상 endpoint 보존, KPI/페르소나의 worker→조립→QA 일치, 실제 세그먼트 누락, 진단 해결 이력을 포함한다.

- 최초 보완 후 PostgreSQL 저장소 통합 검증을 활성화한 전체 테스트: 677 passed (20.82초).
- 1~4번 추가 보완 후 같은 통합 검증을 활성화한 최종 전체 테스트: 714 passed (45.68초). 회귀 테스트 37건을 추가했다.
- `ruff check config/settings.py tests phase2/feature_draft.py phase2/prd_policy.py`: 통과.
- `ruff check --select F,I phase2/contract_guard.py phase2/feature_generation.py`, `compileall`, `git diff --check`: 통과. 기존 일부 파일의 LF/CRLF 안내는 남아 있다.
- 아래 결과는 로컬 코드 검증이며 실제 모델 응답 품질을 보장하지 않는다.

이 작업에서는 새 모델 호출을 사용하는 Chat→Phase4 라이브 실행을 하지 않았다. 단위·통합 검증을 실제 모델 생성 성공이나 Kafka의 새 결과 저장 성공으로 간주하지 않는다. 10월 23일의 실패 실행 3회는 이전 코드의 결과이며 새 구현의 성공 증거가 아니다.
