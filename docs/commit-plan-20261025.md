# 커밋 대상 정리 — 2026-10-25

로컬 Git 작업 트리 기준이다. 커밋·stage·push·pull·fetch는 수행하지 않았다. 저장된 origin 추적 ref 기준 세 저장소의 ahead/behind는 모두 0/0이며, 새 원격 조회 결과는 아니다. 세 저장소 모두 staged 파일이 없다.

| 저장소 | 브랜치 | 코드·테스트·실행 스크립트 | 기존에 미추적이던 작업 문서 |
|---|---|---:|---:|
| timiroom-backend | develop | 수정 8 + 신규 5 = 13 | 0 |
| timiroom-frontend | develop | 수정 2 + 신규 2 = 4 | 0 |
| pipeline_py | codex/pipeline-contract-validation | 수정 42 + 신규 34 = 76 | 신규 15 |

이 정리 문서는 위 집계 이후 생성했으므로 pipeline 문서 수에 포함하지 않았다. 원한다면 문서 커밋에 함께 포함할 수 있다.

## backend

경로: `C:/Users/jsh68/Desktop/timiroom-backend`

제안 커밋: `fix: confirm pipeline storage and recover pending results`

Kafka 발행과 최종 저장을 구분하고 STORAGE_PENDING 상태, 결과 receipt, 저장 상태 조회, 주기 복구, 트랜잭션과 중복 처리 보호를 추가한다. 생성자 의존성 변경에 따른 기존 테스트 갱신도 함께 포함한다. 신규 migration 2개를 누락하지 않는다.

수정:
```
src/main/java/com/timiroom/domain/pipeline/controller/PipelineController.java
src/main/java/com/timiroom/domain/pipeline/entity/PipelineExecution.java
src/main/java/com/timiroom/domain/pipeline/repository/PipelineExecutionRepository.java
src/main/java/com/timiroom/domain/pipeline/service/PipelineService.java
src/main/java/com/timiroom/infra/ragpipeline/RagPipelineClient.java
src/test/java/com/timiroom/domain/notification/NotificationLinkTest.java
src/test/java/com/timiroom/domain/pipeline/PipelineRunFlowTest.java
src/test/java/com/timiroom/domain/project/ProjectDocumentSaveTest.java
```

신규:
```
src/main/java/com/timiroom/domain/pipeline/entity/PipelineResultReceipt.java
src/main/java/com/timiroom/domain/pipeline/repository/PipelineResultReceiptRepository.java
src/main/java/db/migration/V20261023_02__pipeline_execution_storage_pending.java
src/main/resources/db/migration/V20261023_01__pipeline_storage_receipt.sql
src/test/java/com/timiroom/domain/pipeline/PipelineStoragePostgresTest.java
```

제외: 부모 저장소에 미추적으로 보이는 pipeline_py/, timiroom-frontend/, timiroom-consistency_py/, run-all.bat. AGENTS.md는 로컬 작업 지침이므로 별도 팀 합의 없이 제품 변경 커밋에 넣지 않는 것을 권장한다. 중첩 저장소를 gitlink(160000)로 추가하지 않는다.

## frontend

경로: `C:/Users/jsh68/Desktop/timiroom-backend/timiroom-frontend`

제안 커밋: `fix: show pipeline failures and wait for confirmed storage`

실패 단계와 미실행 단계를 구분하고, 저장 대기 상태 조회·중복 생성 차단·저장 확인 후 완료 전환을 추가한다. PM 단계 명칭과 DBA/API 병렬 표시를 수정한다. package.json 변경은 신규 테스트 실행 목록 추가이며 의존성 변경은 아니다. 이번 diff에는 예전에 작업했던 챗봇 UI 전체 변경이 새로 남아 있는 것은 아니다.

```
package.json
src/components/dashboard/ProjectChatWizard.jsx
src/lib/pipelineProgress.mjs
src/lib/pipelineProgress.test.mjs
```

## pipeline_py

경로: `C:/Users/jsh68/Desktop/timiroom-backend/pipeline_py`

제안 커밋: `fix: stabilize specification contracts and verify phase4 storage`

공유 Feature/operation 계약, 기능 식별자·MVP 범위, PRD/Feature Spec 동기화, API/OpenAPI 입력·응답·권한·처리 테이블, DBA FK·값 제약·인덱스, 결정론적 QA와 제한된 targeted repair, 실패 단계 SSE, Kafka 저장 receipt·실제 저장 확인, Chat 입력 파싱을 포함한다. 변경된 agent만 올리고 신규 공통 모듈을 빠뜨리면 import 및 동작이 깨질 수 있으므로 소스와 테스트를 함께 커밋한다.

기존 수정 42개:
```
config/settings.py
main.py
migrate_db.py
phase1/document_ingestion.py
phase1/semantic_chunking.py
phase2/agent_contract.py
phase2/agents/api_agent.py
phase2/agents/dba_agent.py
phase2/agents/feature_spec_agent.py
phase2/agents/pm_agent.py
phase2/agents/prd_agent.py
phase2/agents/qa_agent.py
phase2/agents/search_agent.py
phase2/feature_registry.py
phase2/feature_scope.py
phase2/llm_runtime.py
phase2/openapi_spec.py
phase2/orchestration_graph.py
phase2/quality_rules.py
phase2/sse_service.py
phase2/state.py
phase3/retry_service.py
phase3/schema_validator.py
phase3/validation_service.py
phase4/kafka_consumer.py
phase4/kafka_producer.py
routers/chat.py
routers/orchestration.py
scripts/run_live_pipeline_audit.py
tests/test_agent_contract.py
tests/test_agent_quality.py
tests/test_contract_pipeline_regressions.py
tests/test_explicit_qa_contracts.py
tests/test_phase2.py
tests/test_phase2_orchestration.py
tests/test_phase3.py
tests/test_phase4.py
tests/test_prd_chat_runtime_regressions.py
tests/test_prd_item_workers.py
tests/test_sse_service.py
tests/test_structural_contracts.py
tests/test_targeted_repair_contracts.py
```

신규 코드·테스트·스크립트 34개:
```
phase2/artifact_shapes.py
phase2/contract_guard.py
phase2/contract_quality.py
phase2/feature_draft.py
phase2/feature_generation.py
phase2/fk_validation.py
phase2/mvp_policy.py
phase2/operation_contract.py
phase2/prd_policy.py
phase2/prd_quality.py
phase2/registry_contract_guard.py
phase2/repair_budget.py
phase2/repair_scope.py
phase2/validation_contract.py
phase2/workflow_contract.py
phase4/documents.py
phase4/storage_receipt.py
scripts/audit_live_feature_generation.py
scripts/recheck_saved_pipeline.py
scripts/resume_verified_feature_checkpoint.py
tests/test_additional_contract_guards.py
tests/test_delivery_contract_quality.py
tests/test_feature_generation_quality.py
tests/test_flat_feature_draft.py
tests/test_generation_boundary_regressions.py
tests/test_live_qa_regressions.py
tests/test_operation_contracts.py
tests/test_prd_item_policy.py
tests/test_priority_contract_repairs.py
tests/test_prompt_audit_regressions.py
tests/test_shared_generation_contracts.py
tests/test_storage_receipts.py
tests/test_storage_receipts_postgres.py
tests/test_workflow_contracts.py
```

작업 문서 15개(별도 docs 커밋 가능):
```
docs/additional-contract-changes-20261022.md
docs/agent-prompt-audit-20261023.md
docs/contract-storage-live-qa-20261023.md
docs/feature-spec-20-improvements-20261022.md
docs/five-contract-storage-fixes-20261023.md
docs/flat-contract-prd-policy-changes-20261024.md
docs/fresh-chat-careerlog-e2e-20261025.md
docs/live-e2e-verification-20261024-r22.md
docs/live-stabilization-20261024.md
docs/marketplace-qa-corrections-20261025.md
docs/phase2-phase3-code-review-20261022.md
docs/phase4-quality-stabilization-20261024.md
docs/priority-repair-changes-20261022.md
docs/prompt-audit-live-qa-20261023.md
docs/workflow-contract-fixes-20261024.md
```

제안 문서 커밋: `docs: record pipeline QA fixes and live verification`

## 기본 제외 대상

- 각 저장소 .env 및 실제 토큰·키가 있는 로컬 설정.
- pipeline의 test-runs/ 원시 JSON·SSE·응답 진단·로그·checkpoint. 이번 생성 코드 PR에는 전부 추가하지 않는다. 공유할 검증 결론은 docs/에 정리된 문서를 사용한다.
- test-runs/inspect_generated_outputs.py는 재사용 가능한 검사기 후보지만 현재 실행 산출물 디렉터리에 있으므로 기본 목록에서는 제외했다. 별도 정리 시 scripts/로 이동하고 이식성을 확인한다.
- .codex-finalizer/, .codex-finalizer-v2/, .codex_video_deps/, .poster-build/, .ppt-build/, .ppt-build-v2/, .video_edit_preview/.
- edited_videos/, poster-output/, poster_output/, poster_work/, presentation-output/ 등 발표/영상 작업 결과.
- backend의 중첩 저장소 및 run-all.bat. backend 루트에서 git add .로 일괄 추가하지 않는다.

## 권장 묶음과 배포 주의

1. pipeline 소스·테스트·공통 모듈·실행 스크립트 커밋.
2. pipeline 작업 문서 커밋(선택).
3. backend 저장 상태·복구 및 두 migration과 테스트를 한 묶음으로 커밋.
4. frontend 진행 상태·저장 확인 UI와 테스트를 한 묶음으로 커밋.

저장 확인 프로토콜은 pipeline storage 응답 → backend 상태 조회 → frontend 대기로 연결된다. 서로 다른 저장소라 PR은 나누되 관련 변경임을 설명한다. backend/frontend는 현재 develop 위 미커밋 변경이므로 실제 커밋 전에 각각 작업 브랜치를 만드는 것이 좋다.

878개 테스트 통과 및 실제 Chat→Phase4 저장 성공은 pipeline 검증 결과다. 이번 목록 점검 중 backend clean build나 frontend production build를 새로 실행한 것은 아니다. 세 저장소 전체 검증이 동일하게 완료됐다고 주장하지 않는다.
