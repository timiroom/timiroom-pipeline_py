# timiroom-pipeline_py

초기 요구사항 수집, 한국어 RAG, PRD/DB/API 산출물 생성을 담당하는 FastAPI 서비스입니다.
K-EXAONE은 Friendli 전용 엔드포인트로, 임베딩은 Upstage Solar로 호출합니다.

## 로컬 실행

```bash
python -m venv .venv
python -m pip install -r requirements-dev.txt
copy .env.example .env
uvicorn main:app --host 0.0.0.0 --port 8081
```

헬스체크는 `GET /actuator/health/readiness`, Swagger는 `/docs`입니다.
로컬 CrossEncoder 리랭커까지 사용할 경우 `requirements-reranker.txt`를 추가로 설치합니다.

## NAS 배포

`develop` 반영 시 Docker Hub에 `timiroom-pipeline-py:sha-*` 이미지를 올리고,
`timiroom-ops/apps/rag-pipeline`의 이미지와 SealedSecret을 갱신합니다.
NAS의 k3s에서는 `rag-pipeline-svc:8080` ClusterIP로만 노출되므로 별도 도메인이 필요하지 않습니다.

필수 GitHub production 환경 시크릿:

- `PIPELINE_DB_URL`: Python PostgreSQL URL. NAS에서는 호스트를 `postgres`로 지정
- `PIPELINE_EXAONE_API_KEY`
- `PIPELINE_EXAONE_ENDPOINT_ID`
- `PIPELINE_UPSTAGE_API_KEY`

저장소 시크릿 `DOCKER_USERNAME`, `DOCKER_PASSWORD`, `GH_PAT`도 필요합니다.

기존 Spring 파이프라인의 1024차원 벡터 데이터는 보존합니다. Python 서비스는
`RAG_DOCUMENT_TABLE=document_chunks_ko`를 사용하며 init container가 Solar용
`vector(1024)` 테이블을 멱등 생성합니다.
