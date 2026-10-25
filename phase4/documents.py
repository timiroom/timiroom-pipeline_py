"""One document manifest for consumer writes and producer-side verification."""

import json

from phase4.event import PipelineResultEvent


def event_documents(event: PipelineResultEvent):
    metadata = {"pipeline_id": event.pipeline_id, "project_name": event.project_name,
                "platform": event.platform, "tech_stack": event.tech_stack, "query": event.user_query}
    values = [("prd", event.prd_document), ("erd", event.db_schema), ("api", event.api_spec),
              ("features", "\n".join(event.feature_list))]
    for kind, content in (("openapi", event.openapi_spec), ("feature_spec", event.feature_spec_document),
                          ("feature_registry", json.dumps(event.feature_registry, ensure_ascii=False) if event.feature_registry else ""),
                          ("market_research", event.market_research)):
        if content.strip() and content.strip() != "{}":
            values.append((kind, content))
    return [(content, {**metadata, "type": kind}) for kind, content in values]
