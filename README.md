# timiroom-pipeline_py

초기 요구사항 수집, 한국어 RAG, PRD/DB/API 산출물 생성을 담당하는 FastAPI 서비스입니다.
문서 생성은 OpenAI GPT-5.4 Mini, 임베딩은 Upstage Solar,
검색 결과 재정렬은 Cohere Rerank를 사용합니다.
Phase 2 시장조사는 GPT-5.4 Mini의 Responses API Web Search를 사용하며 출처를 함께 생성합니다.

## 로컬 실행

```bash
python -m venv .venv
python -m pip install -r requirements-dev.txt
copy .env.example .env
uvicorn main:app --host 0.0.0.0 --port 8081
```

헬스체크는 `GET /actuator/health/readiness`, Swagger는 `/docs`입니다.
Cohere Rerank는 기본 의존성의 비동기 HTTP 클라이언트로 호출하므로 별도 모델 설치가 필요하지 않습니다.

## NAS 배포

`develop` 반영 시 Docker Hub에 `timiroom-pipeline-py:sha-*` 이미지를 올리고,
`timiroom-ops/apps/rag-pipeline`의 이미지와 SealedSecret을 갱신합니다.
NAS의 k3s에서는 `rag-pipeline-svc:8080` ClusterIP로만 노출되므로 별도 도메인이 필요하지 않습니다.

필수 GitHub production 환경 시크릿:

- `PIPELINE_DB_URL`: Python PostgreSQL URL. NAS에서는 호스트를 `postgres`로 지정
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`: OpenAI 호환 API의 기본 엔드포인트(예: `https://api.openai.com/v1`)
- `PIPELINE_UPSTAGE_API_KEY`
- `PIPELINE_COHERE_API_KEY`

Phase 2 기본 실행 제한은 LLM 동시 호출 8개, 호출당 120초, 전체 900초입니다.
`.env`의 `PHASE2_LLM_MAX_CONCURRENCY`, `OPENAI_REQUEST_TIMEOUT_SECONDS`,
`PHASE2_TIMEOUT_SECONDS`로 조정할 수 있습니다. Web Search를 끄려면
`PHASE2_WEB_SEARCH_ENABLED=false`로 설정할 수 있지만 시장조사의 최신성은 보장되지 않습니다.
Phase 3의 선택적 산출물 복구는 기본 300초로 별도 제한되며
`PHASE3_REPAIR_TIMEOUT_SECONDS`로 조정할 수 있습니다. 복구 타임아웃은
전체 Phase 2 생성 제한과 합산되지 않고 해당 보정 시도만 종료합니다.

저장소 시크릿 `DOCKER_USERNAME`, `DOCKER_PASSWORD`, `GH_PAT`도 필요합니다.

Solar Embedding 2는 1024차원 벡터를 반환합니다. 기존 `document_chunks`가 같은
모델로 생성된 `vector(1024)` 테이블이면 데이터를 보존한 채 그대로 사용합니다.
