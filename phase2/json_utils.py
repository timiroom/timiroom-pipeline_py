"""EXAONE JSON 파싱 유틸리티 — 깨진 JSON 복구."""
import ast
import json
import logging
import re

logger = logging.getLogger(__name__)

# 한국어/영어 응답 맥락에서 나타날 수 없는 스크립트 — EXAONE이 드물게 토큰을 잘못
# 생성해 태국어·아랍/페르시아어·일본어 등이 섞여 나오는 오염을 감지한다.
_SUSPICIOUS_SCRIPT_RE = re.compile(
    r'[؀-ۿݐ-ݿ'  # 아랍/페르시아
    r'฀-๿'                # 태국
    r'ऀ-ॿ'                # 데바나가리
    r'֐-׿'                # 히브리
    r'Ѐ-ӿ'                # 키릴
    r'぀-ゟ'                # 히라가나
    r'゠-ヿ]'               # 가타카나
)


def has_suspicious_script(node) -> bool:
    """파싱된 JSON 트리 안에 예상치 못한 스크립트가 섞여 있으면 True."""
    if isinstance(node, str):
        return bool(_SUSPICIOUS_SCRIPT_RE.search(node))
    if isinstance(node, dict):
        return any(has_suspicious_script(v) for v in node.values())
    if isinstance(node, list):
        return any(has_suspicious_script(v) for v in node)
    return False


def try_parse_json(raw: str) -> dict | list | None:
    """마크다운·태그 제거 후 JSON 파싱. 깨진 경우 복구 시도."""
    raw = _truncate_at_repetition(raw)
    text = _strip_markdown(raw)
    text = _find_json_start(text)
    if not text:
        logger.debug("try_parse_json: JSON 시작 문자({[)를 찾지 못함. raw 앞 200자: %s", raw[:200])
        return None

    # 1) raw_decode: 첫 번째 완전한 JSON 객체만 파싱 (trailing text 무시)
    try:
        node, _ = json.JSONDecoder().raw_decode(text)
        return node
    except Exception:
        pass

    # 2) EXAONE 핵심 버그: 문자열 값 중간에 개행 후 다음 줄이 새 key로 시작
    #    "value\n"nextkey": → "value",\n"nextkey":
    text = _fix_broken_string_lines(text)

    # 3) 리터럴 개행 제거 후 재시도 (EXAONE 문자열 내 줄바꿈 버그)
    flat = text.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
    try:
        node, _ = json.JSONDecoder().raw_decode(flat)
        return node
    except Exception:
        pass

    # 4) trailing comma 제거 — 중첩 구조도 커버하기 위해 반복 적용
    no_trail = flat
    for _ in range(10):
        cleaned = re.sub(r',\s*([}\]])', r'\1', no_trail)
        if cleaned == no_trail:
            break
        no_trail = cleaned
    try:
        node, _ = json.JSONDecoder().raw_decode(no_trail)
        return node
    except Exception:
        pass

    # 4.5) 에러 위치 기반 누락 ] 삽입
    #   EXAONE 잘림 패턴: "indexes": ["val", "relationships": [...]
    #   → ] 누락으로 배열 안에서 "key": 를 만나 "Expecting ',' delimiter" 발생
    for _ in range(8):
        try:
            node, _ = json.JSONDecoder().raw_decode(no_trail)
            return node
        except json.JSONDecodeError as e:
            if e.pos and e.pos < len(no_trail) and no_trail[e.pos - 1] == '"':
                fixed = _insert_missing_bracket(no_trail, e.pos)
                if fixed != no_trail:
                    no_trail = fixed
                    continue
            break

    # 5) 누락된 쉼표 삽입: "value" "key" → "value", "key"
    add_comma = re.sub(r'("(?:[^"\\]|\\.)*"|\d+|true|false|null|\]|\})\s+("|\[|\{)', r'\1, \2', no_trail)
    try:
        node, _ = json.JSONDecoder().raw_decode(add_comma)
        return node
    except Exception:
        pass

    # 6) 열린 괄호 닫기
    repaired = _close_brackets(add_comma)
    try:
        return json.loads(repaired)
    except Exception as e6:
        try:
            pos = e6.pos
            ctx_start = max(0, pos - 120)
            ctx_end = min(len(repaired), pos + 80)
            logger.debug(
                "try_parse_json: 6단계 모두 실패. 오류=%s | 오류 위치 주변: ...%s[HERE]%s...",
                e6,
                repaired[ctx_start:pos],
                repaired[pos:ctx_end],
            )
        except Exception:
            logger.debug("try_parse_json: 6단계 모두 실패. 오류=%s | 앞 300자: %s", e6, repaired[:300])

    # 6.5) 마지막으로 파싱 가능한 요소 경계까지만 잘라내고 나머지 괄호를 닫는 구제 수단
    #      토큰 반복 루프나 응답 잘림으로 배열/객체 중간이 손상된 경우,
    #      이미 완성된 앞부분만이라도 살려서 반환한다 (전체 폐기보다 부분 성공이 낫다)
    salvaged = _salvage_last_valid_boundary(add_comma)
    if salvaged is not None:
        return salvaged

    # 7) ast.literal_eval 폴백
    #    EXAONE이 Python dict 스타일로 출력할 때:
    #    {'key': 'value'} / None 대신 null / True/False 대신 true/false
    try:
        py_text = repaired
        # JSON 예약어 → Python 예약어 (문자열 값 안의 단어는 건드리지 않도록 word boundary 사용)
        py_text = re.sub(r'\bnull\b', 'None', py_text)
        py_text = re.sub(r'\btrue\b', 'True', py_text)
        py_text = re.sub(r'\bfalse\b', 'False', py_text)
        node = ast.literal_eval(py_text)
        if isinstance(node, (dict, list, set, tuple, frozenset)):
            # ast.literal_eval은 set/tuple 반환 가능 — JSON 직렬화 가능한 형태로 변환
            node = _sanitize_for_json(node)
            if isinstance(node, (dict, list)):
                logger.debug("try_parse_json: ast.literal_eval 복구 성공")
                return node
    except Exception as e7:
        logger.debug("try_parse_json: ast.literal_eval 실패. %s | 앞 200자: %s", e7, repaired[:200])

    return None


def _truncate_at_repetition(text: str, min_len: int = 3, threshold: int = 20) -> str:
    """
    LLM 반복 루프 감지: 짧은 패턴(min_len~10자)이 threshold번 이상 연속 반복되면 해당 위치에서 잘라냄.
    예: 'GETGETGETGETGET...' → 첫 반복 시작 지점에서 truncate.
    """
    n = len(text)
    for pat_len in range(min_len, 11):
        i = 0
        while i < n - pat_len * threshold:
            pat = text[i:i + pat_len]
            if not pat.strip():
                i += 1
                continue
            repeat_end = i + pat_len
            while repeat_end + pat_len <= n and text[repeat_end:repeat_end + pat_len] == pat:
                repeat_end += pat_len
            count = (repeat_end - i) // pat_len
            if count >= threshold:
                logger.warning("_truncate_at_repetition: 반복 패턴 발견 '%s' ×%d @ pos %d — 잘라냄", pat[:20], count, i)
                return text[:i]
            i += 1
    return text


def _sanitize_for_json(obj):
    """ast.literal_eval 결과에 섞인 set/tuple/frozenset → list 변환 (JSON 직렬화 보장)."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


_KEY_LINE_RE = re.compile(r'^\s*"[^"\\]+":\s*')  # "key": 로 시작하는 줄


def _insert_missing_bracket(text: str, error_pos: int) -> str:
    """
    'Expecting ',' delimiter' 에러: 배열 안에서 "key": 패턴이 나타남 → ] 누락.
    EXAONE 잘림 버그: "indexes": ["val", "relationships": [...] (] 누락)
    error_pos는 ':' 위치. ':' 앞의 문자열 앞에 ], 를 삽입한다.
    """
    if error_pos <= 0 or error_pos >= len(text):
        return text

    # error_pos 위치의 문자가 ':' 인지 확인
    if text[error_pos] != ':':
        return text

    # text[error_pos - 1] 이 '"' (키 문자열의 닫는 따옴표)인지 확인
    if text[error_pos - 1] != '"':
        return text

    # 키 문자열의 시작 '"' 위치 탐색
    key_close = error_pos - 1  # 닫는 '"'
    pos = key_close - 1
    while pos >= 0:
        if text[pos] == '"':
            # 이스케이프 여부 확인
            num_backslash = 0
            check = pos - 1
            while check >= 0 and text[check] == '\\':
                num_backslash += 1
                check -= 1
            if num_backslash % 2 == 0:  # 이스케이프 아님
                break
        pos -= 1

    if pos < 0:
        return text

    key_open = pos  # 시작 '"'

    # key_open 이전 텍스트에서 trailing 쉼표/공백 제거 후 ], 삽입
    before = text[:key_open]
    stripped = before.rstrip()
    if stripped.endswith(','):
        stripped = stripped[:-1].rstrip()

    logger.debug("_insert_missing_bracket: pos=%d, key=%s → ] 삽입", error_pos, text[key_open:key_close + 1])
    return stripped + '], ' + text[key_open:]


def _fix_broken_string_lines(text: str) -> str:
    """
    EXAONE 핵심 버그 수정: 문자열 값 안에서 개행 후 다음 줄이 새 key-value로 시작.

    예:
      "feature": "강점이다
      "weakness": "약점"
    →
      "feature": "강점이다",
      "weakness": "약점"

    알고리즘: 줄 단위로 스캔하며 현재 줄이 닫히지 않은 문자열로 끝나는데
    다음 줄이 "key": 패턴이면 현재 줄 끝에 '", 를 추가해 문자열을 닫는다.
    """
    lines = text.split('\n')
    out = []
    for i, line in enumerate(lines):
        out.append(line)
        # 현재 줄에 홀수 개의 unescaped 따옴표 → 문자열이 열린 채로 끝남
        if _has_unclosed_string(line):
            # 다음 줄이 새로운 JSON key 로 시작하는지 확인
            next_stripped = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if _KEY_LINE_RE.match(next_stripped):
                # 현재 줄의 trailing 쉼표/공백 정리 후 문자열 닫기
                out[-1] = line.rstrip().rstrip(',') + '",'
    return '\n'.join(out)


def _has_unclosed_string(line: str) -> bool:
    """줄 안에서 unescaped " 개수가 홀수면 True (문자열이 열린 채로 끝남)."""
    count = 0
    esc = False
    for ch in line:
        if esc:
            esc = False
            continue
        if ch == '\\':
            esc = True
            continue
        if ch == '"':
            count += 1
    return count % 2 == 1


def _strip_markdown(text: str) -> str:
    text = text.strip()
    # EXAONE thinking 태그 제거 — enable_thinking=False여도 JSON 값 안에 유출되는 케이스 방어
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'</?think\b[^>]*>', '', text)
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"```\s*$", "", text).strip()
    # XML/tool 태그 안 JSON 추출
    m = re.search(r'<[^>]+>(.*?)</[^>]+>', text, re.DOTALL)
    if m:
        inner = m.group(1).strip()
        if inner.startswith('{') or inner.startswith('['):
            return inner
    return text


def _find_json_start(text: str) -> str:
    # { 우선: 모든 에이전트 응답은 JSON object({})이므로 [ 보다 { 선호
    brace = text.find('{')
    if brace != -1:
        return text[brace:]
    bracket = text.find('[')
    return text[bracket:] if bracket != -1 else ""


def _salvage_last_valid_boundary(text: str) -> dict | list | None:
    """토큰 반복 루프·응답 잘림으로 배열/객체 요소 중간이 손상된 경우,
    마지막으로 완결된 객체(`}`) 경계까지만 남기고 나머지 열린 괄호를 닫아 파싱을 시도한다.
    뒤에서부터 시도해 가장 많은 내용이 살아남는 경계를 우선 채택한다 —
    전체를 폐기하는 것보다 이미 완성된 앞부분이라도 살리는 편이 낫다.
    """
    positions = [i for i, ch in enumerate(text) if ch == '}']
    for pos in reversed(positions[-200:]):
        candidate = _close_brackets(text[:pos + 1])
        try:
            node = json.loads(candidate)
            logger.warning(
                "try_parse_json: 마지막 유효 객체 경계(%d/%d자)까지 잘라내어 부분 복구",
                pos + 1, len(text),
            )
            return node
        except Exception:
            continue
    return None


def _close_brackets(text: str) -> str:
    # 미완성 문자열 닫기
    in_str = False
    esc = False
    for ch in text:
        if esc:
            esc = False
            continue
        if ch == '\\':
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
    if in_str:
        text += '"'

    # 스택으로 열린 괄호 추적 → 역순으로 닫기
    stack = []
    closer = {'{': '}', '[': ']'}
    opener_of = {'}': '{', ']': '['}
    in_str = False
    esc = False
    for ch in text:
        if esc:
            esc = False
            continue
        if ch == '\\':
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if not in_str:
            if ch in ('{', '['):
                stack.append(ch)
            elif ch in ('}', ']'):
                if stack and stack[-1] == opener_of[ch]:
                    stack.pop()
    return text + ''.join(closer[c] for c in reversed(stack))
