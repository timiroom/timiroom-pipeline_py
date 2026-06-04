from .models import FormData, MoSCoW


class FormToQueryService:

    def synthesize(self, form: FormData) -> str:
        parts = []
        parts.append(f"{form.project_name} 서비스를 {form.platform.value} 플랫폼으로 개발합니다.")
        parts.append(form.project_description)
        if form.tech_stack:
            parts.append(f"기술 스택: {', '.join(form.tech_stack)}")

        pd = form.problem_definition
        parts.append("\n[문제 정의]")
        parts.append(f"핵심 문제: {pd.current_pain_point}")
        parts.append(f"현재 해결 방식: {pd.current_solution}")
        parts.append(f"이상적인 상태: {pd.ideal_state}")
        if pd.business_impact:
            parts.append(f"비즈니스 임팩트: {pd.business_impact}")
        if pd.motivation:
            parts.append(f"서비스 동기: {pd.motivation}")
        if pd.competitor_gap:
            parts.append(f"경쟁사 대비 차별점: {pd.competitor_gap}")

        parts.append("\n[타겟 유저]")
        for i, u in enumerate(form.target_users, 1):
            parts.append(f"유저{i}: {u.persona} / {u.usage_environment} / {u.biggest_pain_point}")

        fd = form.feature_definition
        must = self._get_must(fd)
        should = self._get_by_priority(fd, MoSCoW.SHOULD)
        could = self._get_by_priority(fd, MoSCoW.COULD)
        wont = self._get_by_priority(fd, MoSCoW.WONT)

        parts.append("\n[기능 정의]")
        if must:
            parts.append(f"MVP 필수(Must): {', '.join(must)}")
        if should:
            parts.append(f"우선순위 높음(Should): {', '.join(should)}")
        if could:
            parts.append(f"여유 시 포함(Could): {', '.join(could)}")
        if wont:
            parts.append(f"명시적 제외(Won't): {', '.join(wont)}")

        return "\n".join(parts).strip()

    def extract_must_features(self, form: FormData) -> list[str]:
        return self._get_must(form.feature_definition)

    def extract_excluded_features(self, form: FormData) -> list[str]:
        return self._get_by_priority(form.feature_definition, MoSCoW.WONT)

    def extract_all_included_features(self, form: FormData) -> list[str]:
        fd = form.feature_definition
        return (self._get_must(fd)
                + self._get_by_priority(fd, MoSCoW.SHOULD)
                + self._get_by_priority(fd, MoSCoW.COULD))

    def _get_must(self, fd) -> list[str]:
        result = []
        if fd.common_features:
            result += [f.feature_name for f in fd.common_features if f.selected]
        if fd.custom_features:
            result += [f.feature_name for f in fd.custom_features if f.priority == MoSCoW.MUST]
        return result

    def _get_by_priority(self, fd, priority: MoSCoW) -> list[str]:
        if not fd.custom_features:
            return []
        return [f.feature_name for f in fd.custom_features if f.priority == priority]
