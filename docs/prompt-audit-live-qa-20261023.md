# 프롬프트 정리 후 실제 실행 QA (2026-10-23)

## 검증 범위

반려식물 관리 서비스 '그린로그'를 새 Chat 8턴부터 실제 HTTP API로 실행했다. 실제 모델 호출, Phase1 검색, SSE 진행 상태, Phase4 Kafka 발행 및 PostgreSQL 저장 확인이 대상이다. 브라우저 로그인부터 Java 저장/UI까지의 전체 사용자 E2E를 실행했다는 의미는 아니다.

기반 변경 사항은 `docs/agent-prompt-audit-20261023.md`의 적용 기록에 있다. 검증을 통과시키기 위한 QA 완화, 잘린 JSON 부분 복구, 임의 계약 생성은 사용하지 않았다.

## 실행 1

- pipelineId: `04b81ae9-2e77-429e-8973-7dcdee233f7c`
- 결과 폴더: `test-runs/live-plant-care-20261023-022700`
- Chat 포함 221.361초, 실패 단계 Feature Spec.
- Chat → FormData → Phase1 → Search → PM → PRD까지 진행했다.
- Feature Spec의 두 응답은 각각 JSON offset 6655, 6607에서 구문 검증에 실패했다. 중첩 responseSchema를 닫기 전에 다음 endpoint가 시작됐다. finish_reason=stop이며 출력 토큰 부족이라고 단정하지 않았다.
- generation blocker를 불변 오류로 기록하고 Phase3의 같은 재생성 반복은 중단했다. DBA/API/Phase4는 실행하지 않았다. 실제 저장 청크는 0건이다.
- PM/PRD의 임시 입력 Registry 5개는 생성된 기능명세서 5개가 아니다. 실제 승인된 Feature Spec은 0개다.

## 실행 1 내용 검토

- PM은 물주기 누락·생육 변화·이상 징후·일정 안내를 목표와 범위에 연결했고, 판매/커뮤니티/IoT/확정 진단 등 범위 밖 항목을 구분했다.
- PRD는 다섯 핵심 사용자 흐름의 설명과 요구사항을 작성했다. 생육 사진에는 파일 형식·용량 실패 조건을, 이상 징후 안내에는 확정 진단이 아닌 참고 정보라는 조건을 포함했다.
- 외부 자료가 없다는 한계를 background에 명시했고 KPI는 내부 제품 목표로 구분했다. 확인되지 않은 통계가 검증된 근거라고 소개되지 않았다.
- 남은 내용 QA: mvpScope.excluded 배열은 비었지만 rationale/PM에는 제외 항목이 있다. 일부 페르소나의 연령·직업은 확인된 사용자 사실이 아니므로 가설 표시가 필요하다. auth 기술 설명의 소셜 로그인은 실제 최종 계약과 대조가 필요하다.
- KPI 1개 worker 검증 실패와 persona/KPI 개수 보강은 qualityMetadata.fallbacks에 기록돼 있다. 이 실행의 PRD를 조사·개발 검증까지 완료한 확정 명세라고 볼 수 없다.
- Feature Spec이 실패했으므로 API·DB 내용 품질과 지식그래프 연결성을 평가할 산출물은 없다. 빈 결과에서 FK 오류가 0개라는 것을 통과 근거로 사용하지 않았다.

## 추가 보강

auto 모드의 json_schema 응답이 JSON 구문 오류로 실패하면 기존 1회 재시도만 json_object로 전환한다. 로컬 스키마와 계약 검사는 그대로 유지한다. Feature Spec에는 PostgreSQL 필드 타입과 JSON Schema 타입을 구분하고 중복 선택 필드를 줄이도록 안내했다. 응답을 임의로 고쳐 통과시키지 않는다.

## 실행 2

- pipelineId: `a75c21ac-2094-4638-b969-f66baf335787`
- 결과 폴더: `test-runs/live-plant-care-20261023-023446`
- Chat 포함 211.157초, 실패 단계 Feature Spec. 실제 저장 청크 0건.
- 첫 json_schema 응답은 JSON 오류였고 기존 재시도의 json_object 응답은 정상 파싱됐다. 하지만 파싱 성공만으로 계약 통과시키지 않았다.
- 두 가지 실제 계약 문제가 남았다. 같은 plants.id/users.id의 default가 생성 기능에서는 gen_random_uuid(), 다른 기능에서는 null이었다. 두 GET의 fieldMappings.column에는 여러 컬럼을 쉼표로 묶은 값이 들어 있었다.
- raw 응답에는 기능 후보 7개가 있었으나 계약 검증에 실패했으므로 전달된 Feature Spec 기능은 0개다. DBA/API/Phase4를 실행하지 않았다.
- 보강: DB 필드는 전체 기능 집합에서 정의하되 참조 대상 필드를 기능마다 중복 정의하지 않아도 된다고 명시했다. 반복 선언 시 물리 타입·default·generated·PK는 동일하게 복사하도록 했다. 매핑은 스칼라 DTO 필드 하나와 실제 컬럼 하나만 허용하는 예시를 추가했다.
- 검사 코드에서도 같은 공유 컬럼의 최초 충돌 쌍만이 아니라 모든 선언 소유자를 수정 범위에 넣도록 고쳤다. 첫 두 개만 고친 뒤 다음 소유자에서 동일 blocker가 반복되는 것을 방지한다.
- 현재 호환 gateway에서 non-strict schema 응답의 JSON 오류가 반복 관찰되어 기본 출력 모드를 json_object로 정했다. 명시 auto/json_schema 설정은 유지하며 provider가 대신 계약을 보장한다고 설명하지 않는다. prompt/local schema 검증은 동일하다.

## 회귀 검증

- Python: 실제 PostgreSQL opt-in 포함 644건 통과.
- frontend: 33건 통과, lint/production build 성공.
- 주요 추가 검증: PM 계획의 PRD 전달, 실패 이후 실행 차단, provider/prompt/local schema 일치, cap보다 많은 계약 보존, 공유 route의 다중 featureId, 공개 인증/파라미터 보존, PRD transport와 format retry 분리, Search 예외의 근거 혼입 방지, JSON 재시도 예산 유지.

회귀 테스트 결과와 새 실제 실행 결과는 구분한다.

## 최종 실행 3

- pipelineId: `c2b01aaf-cadd-4421-be87-f977630f6659`
- 결과 폴더: `test-runs/live-plant-care-20261023-024422`
- Chat 포함 215.854초. Chat → FormData → Phase1 → Search → PM → PRD를 통과했지만 Feature Spec에서 실패했다.
- 두 호출 모두 json_object, finish_reason=stop이었다. 각각 35,089/30,586 bytes, completion tokens 8,839/7,534로 응답했으며 설정된 16,000 출력 토큰 한도에 도달하지 않았다. 이 결과를 토큰 부족 또는 timeout으로 설명하지 않는다.
- 첫 응답은 JSON offset 19856, 재시도는 18135에서 실패했다. 중첩 responseSchema의 객체를 닫기 전에 다음 endpoint가 이어지는 형태다. json_object 모드도 현재 경로에서 문법 유효성을 보장하지 못했다.
- 실제 전달 가능한 Feature Spec/API/DB는 없으며 Kafka 발행·Phase4 저장은 실행하지 않았다. PostgreSQL의 해당 pipelineId 문서 청크는 0건이다.

### 시간 분해

SSE 이벤트의 경과 시간 차이이며, Search/PM/PRD 값에는 단계 전환 오버헤드가 포함된다.

| 구간 | 초 |
| --- | ---: |
| Chat 시작 → 파이프라인 시작 | 44.84 |
| Phase1 | 0.48 |
| Search | 21.07 |
| PM | 11.88 |
| PRD | 46.96 |
| Feature Spec 및 실패 확정 | 90.62 |
| 전체 | 215.85 |

PDF 첨부 없는 실행이고 RAG 검색 결과도 0건이므로 Phase1은 FormData 컨텍스트를 사용했다. 이를 PDF ingestion 전체 성능으로 일반화하지 않는다.

### 최종 내용 QA

- 최종 PRD의 핵심 흐름 5개에는 설명과 요구사항이 있다. 생육 사진의 삭제, 이상 징후의 수정·삭제와 이력, 일정 알림의 활성화/비활성화도 요구사항에 포함됐다.
- 최종 PRD의 auth 기술 설명은 이메일 세션 인증이며, 실행 1에서 관찰했던 소셜 로그인 확장은 이 결과에는 없다.
- mvpScope.excluded의 빈 배열과 rationale/PM 제외 범위의 차이는 여전히 남아 있다. KPI worker 한 항목과 persona 개수 보강은 qualityMetadata에 기록돼 있다. 일부 페르소나 설명은 여전히 범용적이다.
- PRD에 남은 intent.*는 입력 추적용 임시 ID다. Feature Spec이 성공하지 않았기 때문에 최종 의미형 ID로 확정되지 않았다. 이를 생성된 feature_001 오류와 혼동하지 않는다.
- API/DB 산출물이 없으므로 endpoint/FK/지식그래프 정합성이 검증됐다는 결론을 내릴 수 없다.

## 판정과 남은 과제

프롬프트 점검 문서의 실행 순서·계약 우선·DTO 매핑·검증 형식·transport 분리 개선은 적용했고 회귀 검증을 통과했다. 하지만 실제 생성 성공과 Phase4 저장 성공은 달성하지 못했다.

현재 남은 주요 문제는 Feature Spec이 여러 기능의 깊은 requestSchema/responseSchema를 한 응답으로 직렬화하는 경로다. 프롬프트와 출력 모드 조정만으로 안정화됐다는 증거는 없다. 다음 구조 보강은 더 작은 typed 필드/DTO 계약을 생성하고 코드가 JSON Schema를 조립하는 방식, 또는 제한된 기능 단위 생성으로 출력 깊이와 크기를 줄이는 방식이 필요하다. 이 추가 구조 변경은 이번 작업에서 구현했다고 주장하지 않는다.

세 실행 모두 원문 일부 복구·검증 우회·무제한 재생성을 하지 않았다. 실패 결과를 확정 명세로 저장하지 않았다.

## 2026-10-24 후속 구현

위에서 남은 과제로 기록한 typed draft 및 코드 기반 JSON Schema 조립, 공유 테이블 정의, scoped repair, KPI/페르소나 공통 정책을 후속 구현했다. 내용은 `docs/flat-contract-prd-policy-changes-20261024.md`에 있다. 이 문서의 세 라이브 실행 결과는 수정 전 기록이며 후속 구현이 라이브에서 통과했다는 뜻은 아니다.
