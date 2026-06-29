import asyncio
import logging
from dataclasses import dataclass
from typing import ClassVar

import httpx
import numpy as np
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

RAW_BASE = "https://raw.githubusercontent.com/product-on-purpose/pm-skills/main/skills/{slug}/SKILL.md"

SKILL_SLUGS: list[str] = [
    "define-hypothesis", "define-jtbd-canvas", "define-opportunity-tree",
    "define-prioritization-framework", "define-problem-statement",
    "deliver-acceptance-criteria", "deliver-edge-cases", "deliver-launch-checklist",
    "deliver-prd", "deliver-release-notes", "deliver-user-stories",
    "develop-adr", "develop-design-rationale", "develop-solution-brief",
    "develop-spike-summary", "discover-competitive-analysis",
    "discover-interview-synthesis", "discover-journey-map", "discover-market-sizing",
    "discover-stakeholder-summary", "foundation-lean-canvas",
    "foundation-meeting-agenda", "foundation-meeting-brief", "foundation-meeting-recap",
    "foundation-meeting-synthesize", "foundation-okr-writer", "foundation-persona",
    "foundation-stakeholder-update", "iterate-lessons-log", "iterate-pivot-decision",
    "iterate-refinement-notes", "iterate-retrospective", "measure-dashboard-requirements",
    "measure-experiment-design", "measure-experiment-results",
    "measure-instrumentation-spec", "measure-okr-grader", "measure-survey-analysis",
    "tool-design-sprint-brief", "tool-design-sprint-decide-and-storyboard",
    "tool-design-sprint-map-and-target", "tool-design-sprint-prototype-plan",
    "tool-design-sprint-readiness", "tool-design-sprint-sketch",
    "tool-design-sprint-test-and-score", "tool-foundation-sprint-approach-options",
    "tool-foundation-sprint-basics", "tool-foundation-sprint-brief",
    "tool-foundation-sprint-differentiation", "tool-foundation-sprint-founding-hypothesis",
    "tool-foundation-sprint-magic-lenses", "tool-foundation-sprint-readiness",
    "tool-note-and-vote", "utility-mermaid-diagrams", "utility-pm-changelog-curator",
    "utility-pm-critic", "utility-pm-release-conductor", "utility-pm-skill-auditor",
    "utility-pm-skill-builder", "utility-pm-skill-iterate", "utility-pm-skill-validate",
    "utility-slideshow-creator", "utility-update-pm-skills",
]


@dataclass
class SkillEntry:
    slug: str
    content: str
    embedding: np.ndarray


class PmSkillsLoader:
    def __init__(self, openai_client: AsyncOpenAI):
        self._client = openai_client
        self._skills: list[SkillEntry] = []

    async def load(self) -> None:
        logger.info("PM 스킬 로드 시작 — 총 %d 개", len(SKILL_SLUGS))
        raw_skills = await self._fetch_all()
        if not raw_skills:
            logger.warning("PM 스킬을 하나도 로드하지 못했습니다.")
            return
        logger.info("PM 스킬 fetch 완료 — %d/%d 개 성공", len(raw_skills), len(SKILL_SLUGS))
        await self._embed_all(raw_skills)
        logger.info("PM 스킬 임베딩 완료 — %d 개 메모리에 저장", len(self._skills))

    def has_skills(self) -> bool:
        return bool(self._skills)

    async def find_relevant_skills(self, query: str, top_k: int = 5) -> str:
        if not self._skills:
            return ""

        try:
            resp = await self._client.embeddings.create(
                model="text-embedding-3-small", input=query
            )
            query_vec = np.array(resp.data[0].embedding, dtype=np.float32)
        except Exception as e:
            logger.warning("PM 스킬 쿼리 임베딩 실패 — 스킬 없이 진행: %s", e)
            return ""

        scored = sorted(
            self._skills,
            key=lambda s: self._cosine(query_vec, s.embedding),
            reverse=True,
        )[:top_k]

        parts = ["\n\n---\n## 관련 PM 방법론 (적용 필수)",
                 "아래는 이 프로젝트에 가장 관련된 PM 프레임워크입니다. 산출물 생성 시 실제로 적용하세요.\n"]
        for s in scored:
            parts.append(f"### {s.slug}\n{s.content}\n")
        return "\n".join(parts)

    async def _fetch_all(self) -> list[tuple[str, str]]:
        async def fetch_one(slug: str) -> tuple[str, str] | None:
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    r = await client.get(RAW_BASE.format(slug=slug))
                    if r.status_code == 200 and r.text.strip():
                        return (slug, r.text.strip())
            except Exception as e:
                logger.warning("PM 스킬 fetch 실패: %s — %s", slug, e)
            return None

        results = await asyncio.gather(*[fetch_one(s) for s in SKILL_SLUGS])
        return [r for r in results if r is not None]

    async def _embed_all(self, raw_skills: list[tuple[str, str]]) -> None:
        batch_size = 10
        for i in range(0, len(raw_skills), batch_size):
            batch = raw_skills[i:i + batch_size]
            texts = [content for _, content in batch]
            try:
                resp = await self._client.embeddings.create(
                    model="text-embedding-3-small", input=texts
                )
                for j, (slug, content) in enumerate(batch):
                    vec = np.array(resp.data[j].embedding, dtype=np.float32)
                    self._skills.append(SkillEntry(slug=slug, content=content, embedding=vec))
            except Exception as e:
                logger.warning("PM 스킬 임베딩 실패 (배치 %d): %s", i // batch_size, e)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom else 0.0
