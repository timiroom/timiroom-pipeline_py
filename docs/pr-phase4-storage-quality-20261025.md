# fix: 명세 계약·자동 보정 및 Phase4 저장 확인 안정화

## 변경 배경

기능명세·API·DB가 서로 다른 계약으로 생성되거나, QA 보정에서 기존 산출물이 사라져 전체 생성을 반복하는 문제를 보완합니다. Kafka 발행만으로 완료를 표시하던 흐름도 실제 DB 저장 확인 후 완료하도록 변경합니다.

## 주요 변경

- Feature Registry와 operation 계약을 공통 기준으로 사용합니다. 기능 ID·MVP 범위·우선순위·설명·요구사항을 유지하고 PRD와 Feature Spec의 핵심 기능을 동기화합니다.
- API 입력·응답 필드, 공개/본인/참여자 권한, 처리 테이블과 DB 변경 효과를 검사하고 OpenAPI 3.1로 전달합니다. 사진 변경 입력과 목록 페이지 계약도 보강합니다.
- DBA의 FK 대상, 컬럼 타입·길이·값 범위·상태 제약·인덱스를 결정론적으로 검사합니다. 잘못된 일반 인덱스와 비즈니스 의미가 있는 UNIQUE 오류를 구분합니다.
- PRD repair 시 기존 API/DB를 보존하고 변경된 기능·도메인만 targeted resync합니다. 전체 재생성 fallback을 차단하고 보정 예산·timeout·반복 blocker를 관리합니다.
- 실패 단계·완료 단계·미실행 단계를 SSE에 포함합니다. Phase4는 영속 receipt와 실제 chunk 개수로 저장을 확인하며, 이미 발행된 결과는 저장 대기와 생성 실패를 구분합니다.
- 최신 develop의 AI 사용량·오류 안내·일시 오류 재시도를 보존합니다. 재시도와 기능별 timeout을 함께 적용하고, 로그아웃만으로 세션 저장소를 강제하지 않되 명시된 bearer-session 저장 계약은 검증합니다.
- Kafka 비활성 실행은 저장 성공으로 표시하지 않습니다. 환경키·원시 실행 결과·응답 진단·발표/영상 산출물은 제외합니다.

## 검증

- 최신 develop 통합 후 `ruff check config/settings.py tests` 및 변경 모듈 F/I 검사 통과.
- PostgreSQL 저장 통합을 포함해 `pytest -q`: **910 passed**.
- 충돌 통합에 대한 별도 회귀: 개별 timeout과 gateway 재시도 공존, Kafka 비활성 시 잘못된 완료 표시 차단, AI 오류 원문 노출 방지, 명시적 세션 저장소 검증.
- Python compileall 통과.
- 통합 전 실제 새 Chat → Phase4 검증 기록: 전공물품 중고거래 주제, 445.459초, 기능 6개(P0 4/P1 2), API 20개, 테이블 8개, FK 8개, Kafka 비동기 저장 611/611 chunks. 계약 기반 고립 노드 및 FK/OpenAPI 오류 0. Feature Spec 전송 모드는 `json_object` 환경 변수 override였으며 코드 기본값은 `auto`입니다. 최신 develop 통합 뒤 라이브 E2E를 다시 실행했다는 의미는 아닙니다.

## 연동 및 배포

- Backend `develop`의 `9c775a6`(저장 상태·복구) 및 frontend `develop`의 `6e0fe4d`(저장 확인 UI)와 연결되는 변경입니다.
- 파이프라인 배포 시 `migrate_db.py`의 source_key 중복 방지·storage receipt 변경을 함께 적용해야 합니다. Backend에는 관련 Flyway migration 2개가 포함돼 있습니다.
- 시장 근거, 거래 재요청 정책, KPI의 실제 계측 시점 등 결과물의 남은 결정사항은 작업 QA 문서에 구분했습니다. 실제 애플리케이션 구현이나 모든 주제의 무조건 통과를 보장하는 변경은 아닙니다.
