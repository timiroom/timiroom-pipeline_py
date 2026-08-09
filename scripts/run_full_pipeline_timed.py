"""로컬 서버의 채팅 시작부터 최종 문서 출력까지 시간과 결과를 기록한다."""

from __future__ import annotations

import io
import json
import os
import sys
import time

import httpx

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_URL = "http://127.0.0.1:8081"
DEFAULT_ANSWERS = [
    "냉장고에 뭐가 있는지 몰라서 장을 볼 때마다 뭘 사야 할지 헷갈리는 문제를 해결하는 서비스를 만들고 싶어요",
    "웹사이트(WEB)로 만들고 싶어요",
    "매번 냉장고를 열어서 눈으로 하나씩 확인해야 해서 번거로워요",
    "메모 앱에 그때그때 기록해 둬요",
    "냉장고 재고가 자동으로 파악되고 유통기한이 임박하면 알림이 오면 좋겠어요",
    "혼자 사는 20-30대 1인 가구",
    "재료 추가, 유통기한 알림, 소비 기록",
    "냉프래시",
]
ACADEMY_ANSWERS = [
    "소규모 학원에서 수업 예약과 출석을 전화나 메신저로 관리해서 중복 예약과 누락이 생기는 문제를 해결하고 싶어요",
    "웹사이트로 만들고 싶어요",
    "수강생이 예약 가능 시간을 물어볼 때마다 장부를 찾아보고 답하는 순간이 가장 번거로워요",
    "지금은 카카오톡과 엑셀 장부를 함께 사용하고 있어요",
    "수강생이 빈 수업 시간을 직접 확인해 예약하고 원장과 강사가 출석 현황을 바로 확인하면 좋겠어요",
    "소규모 학원 원장과 강사, 수강생",
    "강의 일정 등록, 수강 예약, 출석 기록",
    "클래스링크",
]
TOOL_RENTAL_ANSWERS = [
    "동네 공유 공구 대여소에서 종이 장부로 대여와 반납을 관리해 동일 공구가 중복 대여되고 파손 이력이 누락되는 문제를 해결하고 싶어요",
    "웹사이트로 만들고 싶어요",
    "회원별 대여 가능 여부와 반납 예정 시간을 확인하고 공구 상태까지 장부에서 찾는 과정이 가장 번거로워요",
    "현재는 엑셀과 종이 대여 장부를 같이 사용하고 있어요",
    "회원 로그인 후 공구의 대여 가능 시간과 반납 예정 시간을 확인해 중복 없이 대여하고 반납 점검 결과까지 연결되면 좋겠어요",
    "공유 공구를 빌리는 지역 주민과 대여소 관리자",
    "공구 등록, 공구 대여, 반납 점검",
    "툴셰어",
]
TODO_ANSWERS = [
    "개인과 소규모 팀이 해야 할 일을 메신저와 메모에 흩어 기록해서 담당 업무와 마감일을 놓치는 문제를 해결하고 싶어요",
    "웹사이트로 만들고 싶어요",
    "할 일이 여러 곳에 흩어져 있어 누가 언제까지 해야 하는지 확인하고 완료 여부를 다시 묻는 순간이 가장 번거로워요",
    "현재는 메신저, 메모 앱, 스프레드시트를 함께 사용하고 있어요",
    "사용자가 로그인한 뒤 할 일을 등록하고 담당자와 마감일을 지정하며 상태 변경 이력을 한곳에서 확인하면 좋겠어요",
    "개인 사용자와 3~10명 규모의 소규모 프로젝트 팀",
    "할 일 등록, 담당자 배정, 완료 상태 기록",
    "태스크플로우",
]
PET_VACCINE_ANSWERS = [
    "동물병원과 보호자가 반려동물 예방접종 기록과 다음 접종 일정을 종이 수첩과 문자로 관리해 접종 시기를 놓치는 문제를 해결하고 싶어요",
    "웹사이트로 만들고 싶어요",
    "보호자가 접종 이력을 문의할 때 진료 기록을 찾아 다음 접종일을 다시 계산하는 과정이 가장 번거로워요",
    "현재는 종이 접종 수첩과 병원 문자 알림을 함께 사용해요",
    "보호자가 반려동물별 접종 이력을 확인하고 병원이 다음 접종 일정을 등록하며 접종 완료 기록을 남기면 좋겠어요",
    "반려동물 보호자와 동물병원 수의사 및 직원",
    "반려동물 등록, 접종 일정 등록, 접종 완료 기록",
    "펫백신노트",
]
VOLUNTEER_ANSWERS = [
    "지역 봉사센터가 봉사활동 모집과 참여 신청 및 활동 확인을 여러 문서로 관리해 신청 누락이 생기는 문제를 해결하고 싶어요",
    "웹사이트로 만들고 싶어요",
    "담당자가 모집 정원과 신청자를 대조하고 실제 참여 여부를 다시 확인하는 과정이 가장 번거로워요",
    "현재는 온라인 설문과 스프레드시트 및 문자 메시지를 함께 사용해요",
    "담당자가 봉사활동과 정원을 등록하고 참여자가 신청하며 활동 후 담당자가 참여를 확인하면 좋겠어요",
    "지역 봉사센터 담당자와 봉사활동 참여자",
    "봉사활동 모집 등록, 참여 신청, 활동 확인",
    "함께온",
]
CONSIGNMENT_ANSWERS = [
    "중고 의류 위탁 매장이 상품별 위탁자와 판매 내역 및 정산 금액을 수기로 관리해 정산 오류가 생기는 문제를 해결하고 싶어요",
    "웹사이트로 만들고 싶어요",
    "상품이 판매된 뒤 위탁자를 찾고 수수료를 계산해 정산 여부를 확인하는 과정이 가장 번거로워요",
    "현재는 종이 상품표와 스프레드시트 및 계좌이체 내역을 함께 사용해요",
    "매장이 위탁자와 상품을 등록하고 판매 내역을 기록하면 정산 금액이 계산되고 위탁자가 정산 상태를 확인하면 좋겠어요",
    "중고 의류 위탁자와 위탁 매장 관리자",
    "위탁 상품 등록, 판매 기록, 정산 처리",
    "리세일파트너",
]
BOOK_CLUB_ANSWERS = [
    "독서모임 운영자가 모임 일정과 참여 신청 및 독서 기록을 여러 도구에 나눠 관리해 정보가 누락되는 문제를 해결하고 싶어요",
    "웹사이트로 만들고 싶어요",
    "운영자가 회차별 정원과 참석자를 확인하고 토론 기록을 다시 모으는 과정이 가장 번거로워요",
    "현재는 단체 대화방과 온라인 설문 및 문서 도구를 함께 사용해요",
    "운영자가 책과 모임 일정 및 정원을 등록하고 회원이 신청하며 참석 후 독서 기록을 공유하면 좋겠어요",
    "독서모임 운영자와 모임 회원",
    "모임 일정 등록, 참여 신청, 독서 기록 공유",
    "북서클",
]
STAGE_EQUIPMENT_ANSWERS = [
    "소규모 공연팀이 공동 장비의 예약과 대여 및 반납 상태를 대화방으로 관리해 일정 충돌과 파손 누락이 생기는 문제를 해결하고 싶어요",
    "웹사이트로 만들고 싶어요",
    "공연 준비 전에 사용 가능한 장비와 반납 예정일 및 기존 파손 상태를 확인하는 과정이 가장 번거로워요",
    "현재는 스프레드시트와 단체 대화방 및 종이 점검표를 함께 사용해요",
    "팀원이 장비를 예약하고 관리자가 대여를 승인하며 반납할 때 상태와 파손 여부를 점검하면 좋겠어요",
    "소규모 공연팀 구성원과 공연장 장비 관리자",
    "장비 등록, 장비 예약, 반납 점검",
    "스테이지기어",
]
SCENARIOS = {
    "default": DEFAULT_ANSWERS,
    "academy": ACADEMY_ANSWERS,
    "tool_rental": TOOL_RENTAL_ANSWERS,
    "todo": TODO_ANSWERS,
    "pet_vaccine": PET_VACCINE_ANSWERS,
    "volunteer": VOLUNTEER_ANSWERS,
    "consignment": CONSIGNMENT_ANSWERS,
    "book_club": BOOK_CLUB_ANSWERS,
    "stage_equipment": STAGE_EQUIPMENT_ANSWERS,
}
ANSWERS = json.loads(os.environ.get("PIPELINE_TEST_ANSWERS_JSON", "null")) or SCENARIOS.get(
    os.environ.get("PIPELINE_TEST_SCENARIO", "default"), DEFAULT_ANSWERS
)


def main() -> int:
    started = time.perf_counter()
    messages: list[dict[str, str]] = []
    form_data = None

    def stamp(label: str, detail: str = "") -> float:
        elapsed = time.perf_counter() - started
        print(f"[T+{elapsed:7.2f}s] {label} {detail}", flush=True)
        return elapsed

    with httpx.Client(timeout=httpx.Timeout(600.0, connect=10.0)) as client:
        for index, answer in enumerate(ANSWERS):
            messages.append({"role": "user", "content": answer})
            turn_started = time.perf_counter()
            response = client.post(f"{BASE_URL}/api/v1/chat/message", json={"messages": messages})
            turn_elapsed = time.perf_counter() - turn_started
            stamp(f"CHAT_{index}", f"status={response.status_code} duration={turn_elapsed:.2f}s")
            response.raise_for_status()
            data = response.json().get("data", {})
            print(
                "CHAT_RESPONSE "
                + json.dumps(
                    {
                        "message": data.get("message"),
                        "isComplete": data.get("isComplete"),
                        "stage": data.get("stage"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            messages.append({"role": "assistant", "content": json.dumps(data, ensure_ascii=False)})
            if data.get("isComplete") and data.get("formData"):
                form_data = data["formData"]
                break

        if form_data is None:
            stamp("FAILED", "formData가 생성되지 않음")
            return 1

        print("FORM_DATA=" + json.dumps(form_data, ensure_ascii=False), flush=True)
        request_started = time.perf_counter()
        response = client.post(
            f"{BASE_URL}/api/v1/orchestration/generate",
            data={"request": json.dumps(form_data, ensure_ascii=False), "skip_phase1": "false"},
        )
        stamp("GENERATE", f"status={response.status_code} duration={time.perf_counter() - request_started:.2f}s")
        response.raise_for_status()
        pipeline_id = response.json()["data"]["pipelineId"]
        print(f"PIPELINE_ID={pipeline_id}", flush=True)

        phase_times: dict[str, float] = {}
        last_step = "PIPELINE_START"
        last_at = time.perf_counter()
        with client.stream("GET", f"{BASE_URL}/api/v1/orchestration/progress/{pipeline_id}") as stream:
            event_name = ""
            for raw_line in stream.iter_lines():
                line = raw_line.strip()
                if line.startswith("event:"):
                    event_name = line[6:].strip()
                elif line.startswith("data:"):
                    now = time.perf_counter()
                    phase_times[last_step] = phase_times.get(last_step, 0.0) + now - last_at
                    payload = json.loads(line[5:].strip())
                    if event_name == "progress":
                        step = str(payload.get("step", "UNKNOWN"))
                        stamp(f"SSE_{step}", str(payload.get("message", "")))
                        last_step = step
                        last_at = now
                    elif event_name == "complete":
                        result = payload["result"]
                        stamp("COMPLETE")
                        print("PHASE_TIMES=" + json.dumps(phase_times, ensure_ascii=False), flush=True)
                        print("FINAL_RESULT=" + json.dumps(result, ensure_ascii=False), flush=True)
                        return 0
                    elif event_name == "error":
                        stamp("ERROR", json.dumps(payload, ensure_ascii=False))
                        return 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
