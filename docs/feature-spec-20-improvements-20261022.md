# Feature Spec 및 파이프라인 보완 20개 항목

## 범위
2026-10-22 그린로그 라이브 실행 `0212ec8f-a9cd-4791-8a49-1dea0c927b4d`의 실패 원인과 PRD 내용 검토를 바탕으로 적용했다. 기존 미커밋 작업은 유지했으며 이 문서의 변경은 별도 커밋이나 push를 하지 않았다.

## 변경 전과 변경 후
| 번호 | 기존 문제 | 변경 내용 | 코드 |
| --- | --- | --- | --- |
| 1 | 자유형 JSON 출력 지시만 사용 | Pydantic에서 만든 json_schema 응답 형식과 엄격한 로컬 검증. 명시적 미지원 응답에만 json_object로 전환 | phase2/feature_generation.py |
| 2 | 필수 계약 요구와 빈 계약 예시가 모순 | 유효한 endpoint/request/response/fieldMappings/DB 필드를 포함한 예시와 비적용 사유 규칙 | phase2/agents/feature_spec_agent.py |
| 3 | 무조건 하위 기능 2~4개 및 10개 이상 생성 지시 | 최대 12개 MVP, 최소 개수 채우기 금지, CRUD는 같은 기능의 여러 operation으로 표현. 출력 토큰 상한 설정 가능 | feature_spec_agent.py, config/settings.py |
| 4 | core 기능 이름이 바뀌면 누락 오인 | sourceFeatureId로 원본을 연결하고 원본 ID와 최종 의미형 ID 분리 | feature_generation.py |
| 5 | 설명/요구사항/적용 영역 일부 누락 | 필수 생성 모델과 계약 검사. UI/API/DB 비적용 사유 및 supporting 부모/이유 검증 | feature_generation.py, artifact_shapes.py |
| 6 | 부분 JSON 복구 결과가 생성 검사에 유입 | Feature Spec에는 json.loads 완전 파싱만 사용. 중복 키, 비표준 상수, 코드펜스, 잘림도 거부 | feature_generation.py |
| 7 | 파싱/형식/내용 실패 원인이 혼재 | JSON 위치, shape, 필수 필드, core coverage, scope, provider finish/refusal/transport 구분 | feature_generation.py |
| 8 | 생성 오류는 보정 없이 Phase3에서 반복/차단 | 최초 1회 + 최대 1회 보정. SDK 중첩 재시도 비활성. 일시 통신 오류는 같은 예산에서 제한 재시도 | feature_generation.py |
| 9 | 보정이 정상 기능도 덮어쓸 가능성 | 오류가 있는 기능 또는 누락된 core만 교체/추가. 정상 sibling 반환/변경 거부, 전체 결과 재검증 | feature_generation.py |
| 10 | intent 임시 ID가 최종 ID처럼 노출 | 원본 참조에만 임시 ID 허용. 최종 Registry는 의미형 ID 검증 후 발행 | feature_spec_agent.py, contract_guard.py |
| 11 | 상위 실패 후 API/DB 빈 값이 독립 원인처럼 표시 | Feature Spec 실패 시 최초 원인만 blocker로 보고하고 하위 미실행을 별도 메타데이터로 전달 | phase3/validation_service.py |
| 12 | 마지막 Phase3 이벤트가 실제 실패 단계를 가림 | failedStage/lastCompletedStage/skippedStages 전달, 생성 단계에서 실패하면 Phase3 진행률로 이동하지 않음. frontend 단계 순서도 실제 PRD→PM 순서에 맞춤 | routers/orchestration.py, state.py, ProjectChatWizard.jsx |
| 13 | 일시 오류/보정 가능 오류/최종 오류를 동일 취급 | capability fallback, 통신 재시도, 기능 보정, 예산 소진 후 Phase3 재시도 금지를 구분. 거부/인증 오류는 반복하지 않음 | feature_generation.py, retry_service.py |
| 14 | 사용자별 접근 요구와 선택적 인증 문구 충돌 | 소유권 요구가 있으면 인증·요청별 접근 검사 명시. 공개 서비스에는 계정 강제하지 않음 | phase2/prd_quality.py |
| 15 | 재방문율 분모/기간이 불명확 | 코호트/재방문 이벤트, UTC 측정 구간, 관측 가능한 분모와 고유 사용자 분자, 누적/일자 retention 구분 | prd_quality.py |
| 16 | fallback 페르소나가 도메인과 무관 | 사용자 기능명을 사용해 실제 이용 흐름 보완. 조사로 검증해야 하는 가설임을 표시 | prd_agent.py, prd_quality.py |
| 17 | fallback 텍스트 사용 사실이 결과에서 사라짐 | worker 및 의미 검수의 대체·개수 보충 기록을 qualityMetadata 및 artifactQuality로 전달 | prd_agent.py, state.py, orchestration.py |
| 18 | 원시 응답·사용량·파싱 원인 진단 부족 | 응답 hash/크기, finishReason, usage, 시간, 오류 위치 기록. 원문은 설정 시에만 로컬 파일에 토큰/비밀값 마스킹 후 기록 | feature_generation.py |
| 19 | 실제 실패 사례 회귀 테스트 부족 | JSON 잘림/중복키, 계약 자료형, rename/누락, 범위 외 보정, provider fallback, refusal/예산/저장 대기 및 SSE 다중 구독 테스트 | tests/test_feature_generation_quality.py 등 |
| 20 | Kafka 발행을 DB 저장 완료로 표시 | consumer와 같은 문서 manifest로 모든 예상 source_key의 실제 DB 존재 여부 확인. 저장 확인 후에만 complete. pending은 명시하고 중복 생성 차단 | phase4/documents.py, kafka_producer.py, document_ingestion.py, frontend/backend |

## 중요한 한계와 운영 설정
- provider 응답 형식은 **strict=false JSON Schema guidance**다. 동적인 requestSchema/responseSchema 객체를 지원하기 위한 선택이며, strict=true 서버 강제 보장을 의미하지 않는다. 최종 계약은 로컬 Pydantic 및 의미 검사를 반드시 통과해야 한다.
- 실제 configured upstream에 작은 json_schema(strict=false) 요청을 보내 `{"ok":true}` 응답을 확인했다. 큰 계약도 항상 생성된다는 보장은 아니므로 별도 라이브 E2E가 필요하다.
- `FEATURE_SPEC_OUTPUT_MODE=auto` 기본. `json_schema`, `json_object`를 명시할 수도 있다. auto는 400/422 응답이 명확하게 response_format 미지원이라고 할 때만 형식을 낮춘다. 인증/서버 장애를 capability 오류로 오인하지 않는다.
- 생성 내용은 최대 두 번 요청한다. capability 미지원 요청이 한 번 추가될 수 있다. 기존 API/DBA targeted repair 예산은 증가시키지 않는다.
- 최초 응답 전체가 깨져 유효한 기능을 식별할 수 없으면 Feature Spec 문서만 한 번 완전 재요청한다. 유효한 JSON에서 일부 계약이 잘못됐으면 해당 기능만 patch한다. 파싱 불가능한 응답의 정상 sibling 보존을 보장한다고 주장하지 않는다.
- 대형 출력을 무제한 분할 생성하지 않는다. 최대 기능 수와 프롬프트 중복 지시를 줄이고 `FEATURE_SPEC_MAX_COMPLETION_TOKENS=16000` 상한을 사용하며, length 종료는 성공으로 받지 않는다.
- `FEATURE_SPEC_DIAGNOSTICS_DIR` 기본 빈 값: 원문 파일 저장 안 함. 설정하면 생성된 문서 내용이 로컬에 남으므로 접근 제한, 개인정보 검토, 운영자 보존/삭제 정책이 필요하다. 정규식 마스킹은 모든 민감정보를 탐지한다고 보장하지 않는다. 원문/파일경로는 SSE에 보내지 않는다.
- `PHASE4_STORAGE_TIMEOUT_SECONDS=120` 기본. 타임아웃이면 Kafka 이벤트가 이미 발행되어 이후 저장될 수 있다. DB 저장 실패라고 단정하거나 자동 재발행하지 않는다. backend는 STORAGE_PENDING으로 기록하고 재시작을 막는다. 지연 저장의 자동 상태 복구 작업은 별도 운영 기능이며 이번 변경에 포함하지 않는다.
- 오래된 Java pipeline처럼 storage 필드가 아예 없는 완료 이벤트는 기존 호환성을 유지한다. 새 Python pipeline의 완료는 receipt가 있는 경로로 검증한다.
- backend watcher와 브라우저가 같은 SSE 큐를 나눠 소비하지 않도록 구독자별 큐를 사용한다. 프로세스 재시작을 견디는 분산 이벤트 저장소로 바뀐 것은 아니다.
- 페르소나의 직업/연령/이용 빈도를 실제 사용자 조사 결과처럼 만들어내지 않는다. 보완한 내용은 가설로 표시한다.

## 검증 기록
- Python 전체: **600 passed** (2026-10-23 최종 실행). Ruff와 git diff --check 통과(기존 CRLF 경고 제외).
- frontend 전체 테스트: **30 passed**, lint 및 production build 성공. lockfile 기준 의존성 설치 완료.
- backend PipelineRunFlowTest / PipelineFailureMessageTest: **13 tests, failures/errors 0**, Gradle BUILD SUCCESSFUL.
- 첫 보완 후 라이브 `92974d5c-fcc9-4d7c-9d26-c86bd1c761c3`: JSON은 정상이나 parentFeatureId를 실행 의존성으로 취급한 기존 검사 때문에 실패. 이를 분리하고, dependency 오류에 기능 포인터를 붙여 targeted patch 범위를 판단하도록 보완했다. 최초 실제 응답을 수정 검사로 재검증하면 이슈 0건이다. 이는 저장된 응답의 검사 결과이며 새 라이브 성공을 의미하지 않는다.
- 라이브 E2E: 아래 실행 보고서로 최종 결과를 별도 기록한다. 단위 테스트 통과는 라이브 성공을 뜻하지 않는다.

## 최종 라이브 결과와 후속 보완
- 새 Chat 실행 `07ad7be7-71a1-4117-bed9-6c928da1df2d`: 263.685초. Feature Spec은 첫 호출에서 8개 기능의 계약 검증에 통과했다. API 27개, DB 테이블 11개, OpenAPI 3.1.0이 생성됐다.
- 최종 Phase3에서는 API effects 문자열 배열 때문에 실패했으며 Phase4에는 진입하지 못했다. 실제 DB 저장은 0건이다. 이 실행의 예전 failedStage 값 QA_REPAIR는 정확한 실패 단계가 아니므로 보고서에서는 실제 Phase3로 구분한다.
- 이 실행을 보고 API worker의 effects 객체 형식 지시, shape 오류의 endpoint key/포인터 표시 및 repair allowlist 해석, Phase3 실패 단계 표시를 추가 보완했다.
- DBA constraint 정규화가 non-id 참조를 id로 바꾸던 문제와 명시적 필드 타입 반영 후 FK 타입이 다시 어긋나던 순서를 수정했다. UUID에 잘못 남은 GENERATED_IDENTITY도 제거한다.
- 저장된 DB를 최신 self-check로 재검사하면 FK 타입 불일치는 사라지고, `symptom_guidance.symptom_type -> symptoms.symptom_type`의 PK/UNIQUE 부재가 남는다. 이 관계는 생성된 계약 자체를 검토해야 하며 통과를 위해 임의 UNIQUE를 추가하지 않았다.
- 마지막 보완 이후 새 Chat부터 전체 라이브를 다시 실행한 것은 아니다. API 보정 및 Phase4 실제 저장 성공을 최종 확인했다고 주장하지 않는다.
- 상세 결과: `test-runs/live-plant-care-20261022-194842/QA_REPORT.md`. 원본 산출물은 같은 폴더의 개별 JSON 및 ALL_SPECIFICATIONS.md에 보존했다.
