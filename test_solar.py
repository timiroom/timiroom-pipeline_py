# pip install openai numpy
from openai import OpenAI
import numpy as np

client = OpenAI(
    api_key="up_tJ7QBw9w1cgSlEc7dL2mF5a2nYavO",  # 본인 키 입력
    base_url="https://api.upstage.ai/v1"
)

def get_embedding(text, model):
    response = client.embeddings.create(input=text, model=model)
    return np.array(response.data[0].embedding)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# -----------------------------------------------------------------------
# 테스트 데이터: 쿼리(query) - 정답 문서(passage) - 오답/방해 문서(distractor)들
# 실제 검색 시나리오처럼 "이 질문에 대해 정답 문서가 다른 문서들보다
# 더 높은 유사도로 상위에 뽑히는지"를 확인합니다.
# -----------------------------------------------------------------------
test_cases = [
    {
        "query": "파이썬에서 리스트를 정렬하는 방법",
        "correct": "파이썬의 sort() 메서드나 sorted() 함수를 사용하면 리스트를 오름차순 또는 내림차순으로 정렬할 수 있다.",
    },
    {
        "query": "감기에 좋은 음식이 뭐야?",
        "correct": "생강차, 배즙, 도라지즙은 기침과 목감기 증상 완화에 도움을 줄 수 있다.",
    },
    {
        "query": "서울에서 부산까지 가는 가장 빠른 방법",
        "correct": "KTX를 이용하면 서울에서 부산까지 약 2시간 30분이 소요된다.",
    },
    {
        "query": "강아지 산책은 하루에 몇 번이 적당해?",
        "correct": "성견 기준으로 하루 1~2회, 각 20~30분 정도의 산책이 적당하다고 알려져 있다.",
    },
    {
        "query": "주식 투자 초보자가 조심해야 할 점",
        "correct": "초보 투자자는 분산 투자 없이 한 종목에 몰빵하는 것을 특히 주의해야 한다.",
    },
]

# 모든 정답 문서를 모아서 "문서 풀(pool)"로 사용 (서로가 서로의 방해 문서 역할)
document_pool = [case["correct"] for case in test_cases]

# 추가 방해 문서(완전히 무관한 내용들)도 풀에 섞어 넣어 난이도를 높임
distractor_pool = [
    "오늘 회의는 오후 3시에 시작된다.",
    "고양이는 하루에 12~16시간을 잔다.",
    "아이폰 신제품이 다음 달 출시될 예정이다.",
    "된장찌개를 끓일 때는 된장을 체에 걸러 넣으면 깔끔하다.",
    "축구 국가대표팀이 다음 경기에서 우승 후보와 맞붙는다.",
]

all_documents = document_pool + distractor_pool


def run_retrieval_test(query_model="solar-embedding-2-query",
                        passage_model="solar-embedding-2-passage"):
    print("=" * 70)
    print(f"Top-k 검색 정확도 테스트 (query 모델: {query_model}, passage 모델: {passage_model})")
    print("=" * 70)

    # 문서 풀 전체를 passage 모델로 미리 임베딩 (실제 서비스에서 하는 방식과 동일)
    doc_embeddings = [get_embedding(doc, passage_model) for doc in all_documents]

    top1_correct = 0
    top3_correct = 0
    mrr_total = 0.0  # Mean Reciprocal Rank

    for case in test_cases:
        query_vec = get_embedding(case["query"], query_model)

        # 모든 문서와의 유사도 계산
        sims = [cosine_similarity(query_vec, doc_vec) for doc_vec in doc_embeddings]
        ranked_indices = np.argsort(sims)[::-1]  # 유사도 높은 순 정렬

        correct_idx = all_documents.index(case["correct"])
        rank = int(np.where(ranked_indices == correct_idx)[0][0]) + 1  # 1-based rank

        is_top1 = rank == 1
        is_top3 = rank <= 3
        top1_correct += is_top1
        top3_correct += is_top3
        mrr_total += 1.0 / rank

        print(f"\n질의: {case['query']}")
        print(f"  정답 문서 순위: {rank}위 (Top-1: {'✅' if is_top1 else '❌'}, Top-3: {'✅' if is_top3 else '❌'})")
        print(f"  상위 3개 검색 결과:")
        for i, idx in enumerate(ranked_indices[:3]):
            mark = "→ 정답" if idx == correct_idx else ""
            print(f"    {i+1}. [{sims[idx]:.4f}] {all_documents[idx]} {mark}")

    n = len(test_cases)
    print("\n" + "=" * 70)
    print("최종 결과")
    print("=" * 70)
    print(f"Top-1 정확도: {top1_correct}/{n} ({top1_correct/n*100:.1f}%)")
    print(f"Top-3 정확도: {top3_correct}/{n} ({top3_correct/n*100:.1f}%)")
    print(f"MRR (Mean Reciprocal Rank): {mrr_total/n:.4f}  (1.0에 가까울수록 좋음)")


if __name__ == "__main__":
    run_retrieval_test()