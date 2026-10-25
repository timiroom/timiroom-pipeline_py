"""Run a live chat-to-Phase4 scenario and persist every terminal artifact for audit."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

BASE_URL = "http://127.0.0.1:8081"
ANSWERS = [
    "초보 반려식물 사용자가 식물별 물주기와 생육 상태를 기록하고 이상 징후를 조기에 확인하기 어려운 문제를 해결하고 싶어요",
    "웹사이트로 만들고 싶어요",
    "여러 식물의 물주기 시기를 기억하지 못해 과습이나 건조가 발생하고, 잎 변색을 발견해도 원인과 조치를 판단하기 어려운 순간이 가장 불편해요",
    "현재는 휴대폰 알람과 메모장에 식물 이름과 물 준 날짜를 따로 기록하고 있어요",
    "사용자가 식물을 등록하고 물주기와 생육 사진을 기록하며, 식물별 관리 일정을 안내받고 이상 징후와 조치 방법을 확인하면 좋겠어요",
    "처음 식물을 키우는 1인 가구와 여러 반려식물을 관리하는 취미 재배자",
    "식물 등록, 물주기 기록, 생육 사진 기록, 관리 일정 알림, 이상 징후 기록과 조치 안내",
    "그린로그",
]
SCENARIOS = {
    "plant-care": ANSWERS,
    "campus-major-market": [
        "대학생들이 수업에서 사용한 전공 교재와 실습 도구를 같은 학교나 전공 학생에게 중고로 거래할 수 있는 서비스를 만들고 싶어요",
        "모바일에서도 편하게 사용할 수 있는 반응형 웹사이트로 만들고 싶어요",
        "일반 중고거래 앱에서는 내 학교와 전공에 맞는 물품을 찾기 어렵고, 단체 채팅방에서는 판매 여부와 구매 요청, 약속 내용을 따로 확인해야 해서 거래를 놓치기 쉬워요",
        "현재는 학과 단체 채팅방이나 일반 중고거래 앱에서 글을 찾아보고 판매자에게 개인 메시지를 보내 직거래를 약속해요",
        "학생이 학교와 전공 정보를 등록하고 물품 사진, 제목, 설명, 가격, 상태를 올리면 다른 학생이 학교·전공·가격으로 검색하고 구매를 요청할 수 있으면 좋겠어요. 판매자는 요청을 수락하고 양쪽이 거래별 메시지로 약속을 정한 뒤 거래를 완료하거나 취소하며 각자 내역을 확인해요. 같은 학생이 판매자와 구매자 모두 될 수 있고, 본인 물품에는 구매 요청을 할 수 없으며 한 물품의 거래는 동시에 하나만 수락할 수 있어야 해요. 초기에는 교내 직거래만 지원하고 결제, 배송, 외부 학교 인증 연동과 실시간 소켓 채팅은 제외해요",
        "전공 교재·실습 물품을 판매하거나 저렴하게 구매하려는 대학생이며 학교와 전공으로 물품을 탐색하고, 물품은 작성자만 수정하며 거래 메시지는 해당 판매자와 구매자만 볼 수 있어요",
        "전공 중고물품 등록·사진 첨부·수정·삭제 및 조회, 학교·전공·가격·판매상태별 물품 검색과 필터, 구매 요청과 판매자 수락·거절 및 거래 취소·완료, 거래 참여자 간 메시지와 내 판매·구매 내역 조회",
        "메이저마켓",
    ],
    "job-application-tracker": [
        "여러 회사에 지원하는 취업 준비생이 지원 마감일과 전형 진행 상태를 흩어진 메모에서 관리하다가 다음 할 일을 놓치는 문제를 해결하고 싶어요",
        "웹사이트로 만들고 싶어요",
        "회사마다 지원 마감일과 면접 일정이 달라 메모를 일일이 확인하고, 어떤 지원이 서류 심사 중인지 면접 대기인지 한눈에 파악하기 어려워요",
        "현재는 스프레드시트와 휴대폰 메모에 회사명, 공고 주소, 마감일, 진행 상태, 면접 일정과 준비 메모를 기록해요",
        "개인 사용자가 지원 건을 등록·조회·수정·삭제하고 전형 상태를 변경하며, 지원 건별 면접 일정과 준비 메모를 관리하고 임박한 마감일과 면접을 목록으로 확인하면 좋겠어요. 채용 사이트 크롤링, 파일 업로드, 이메일·푸시 알림, 기업 관리자와 사용자 간 공유는 초기 범위에서 제외해요",
        "여러 회사에 동시에 지원하는 대학생과 신입 취업 준비생이며 모든 데이터는 본인만 조회하고 수정해요",
        "지원 현황 관리, 전형 상태 변경 및 상태별 필터, 지원 건별 면접 일정과 준비 메모 관리, 임박한 일정 조회. 상태는 준비중·지원완료·서류통과·면접중·최종합격·불합격이며 잘못 입력한 상태는 수정할 수 있어요",
        "커리어로그",
    ],
    "workshop-booking": [
        "동네 공유 공방에서 이용자가 장비별 이용 가능 시간을 확인하지 못하고 운영자는 중복 예약을 수기로 조정하는 문제를 해결하고 싶어요",
        "웹사이트로 만들고 싶어요",
        "이용자가 전화로 장비와 시간대를 문의하고 운영자가 종이에 예약을 적어서 같은 장비에 예약이 겹치거나 취소 내용을 놓치는 것이 불편해요",
        "현재는 전화와 종이 장부로 장비 이용 가능 시간과 예약자를 확인하고 있어요",
        "운영자는 장비와 예약 가능 시간대를 등록하고 이용자는 빈 시간대를 골라 예약·변경·취소하며 자신의 예약 상태를 확인하면 좋겠어요. 결제, 채팅, 외부 달력 연동은 초기 범위에서 제외해요",
        "공방 장비를 예약하는 일반 이용자와 장비·운영 시간을 관리하는 공방 운영자",
        "운영자 장비 및 예약 가능 시간대 관리, 이용자 예약 신청·변경·취소, 개인 예약 목록과 처리 상태 조회",
        "메이커슬롯",
    ],
}


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCENARIOS, default="plant-care")
    args = parser.parse_args()
    started = time.perf_counter()
    output_dir = Path("test-runs") / f"live-{args.scenario}-{datetime.now(UTC):%Y%m%d-%H%M%SZ}"
    output_dir.mkdir(parents=True, exist_ok=False)
    messages: list[dict[str, str]] = []
    chat_trace: list[dict] = []
    form_data = None

    with httpx.Client(timeout=httpx.Timeout(1200.0, connect=10.0)) as client:
        for index, answer in enumerate(SCENARIOS[args.scenario]):
            messages.append({"role": "user", "content": answer})
            turn_started = time.perf_counter()
            response = client.post(
                f"{BASE_URL}/api/v1/chat/message", json={"messages": messages}
            )
            response.raise_for_status()
            data = response.json().get("data", {})
            chat_trace.append({
                "turn": index,
                "answer": answer,
                "durationSeconds": round(time.perf_counter() - turn_started, 3),
                "response": data,
            })
            _write_json(output_dir / "chat_trace.json", chat_trace)
            print(
                f"CHAT {index}: {data.get('stage')} complete={data.get('isComplete')}",
                flush=True,
            )
            messages.append({"role": "assistant", "content": json.dumps(data, ensure_ascii=False)})
            if data.get("isComplete") and data.get("formData"):
                form_data = data["formData"]
                break

        _write_json(output_dir / "chat_trace.json", chat_trace)
        if not form_data:
            raise RuntimeError("Chat did not produce formData")
        _write_json(output_dir / "form_data.json", form_data)

        response = client.post(
            f"{BASE_URL}/api/v1/orchestration/generate",
            data={"request": json.dumps(form_data, ensure_ascii=False), "skip_phase1": "false"},
        )
        response.raise_for_status()
        pipeline_id = response.json()["data"]["pipelineId"]
        print(f"PIPELINE_ID={pipeline_id}", flush=True)

        events: list[dict] = []
        with client.stream(
            "GET", f"{BASE_URL}/api/v1/orchestration/progress/{pipeline_id}"
        ) as stream:
            event_name = ""
            for raw_line in stream.iter_lines():
                line = raw_line.strip()
                if line.startswith("event:"):
                    event_name = line[6:].strip()
                    continue
                if not line.startswith("data:"):
                    continue
                payload = json.loads(line[5:].strip())
                events.append({
                    "event": event_name,
                    "elapsedSeconds": round(time.perf_counter() - started, 3),
                    "payload": payload,
                })
                _write_json(output_dir / "events.json", events)
                if event_name == "progress":
                    print(
                        f"{payload.get('percent')}% {payload.get('step')}: {payload.get('message')}",
                        flush=True,
                    )
                elif event_name == "complete":
                    _write_json(output_dir / "final_result.json", payload["result"])
                    _write_json(output_dir / "events.json", events)
                    print(f"COMPLETE output={output_dir.resolve()}", flush=True)
                    return 0
                elif event_name == "error":
                    _write_json(output_dir / "error.json", payload)
                    _write_json(output_dir / "events.json", events)
                    print(f"ERROR output={output_dir.resolve()}", flush=True)
                    return 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
