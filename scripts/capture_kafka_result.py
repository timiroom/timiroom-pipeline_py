"""Print one pipeline result event from Kafka for local audit use."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aiokafka import AIOKafkaConsumer

from config.settings import settings


async def main(target_prefix: str) -> int:
    consumer = AIOKafkaConsumer(
        settings.kafka_topic_pipeline_result,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id=None,
        auto_offset_reset="earliest",
        value_deserializer=lambda value: json.loads(value.decode("utf-8")),
    )
    await consumer.start()
    try:
        await consumer.seek_to_beginning()
        while True:
            message = await asyncio.wait_for(consumer.getone(), timeout=15)
            event = message.value
            if str(event.get("pipelineId") or "").startswith(target_prefix):
                print(json.dumps(event, ensure_ascii=False, indent=2))
                return 0
    except asyncio.TimeoutError:
        print(f"result not found: {target_prefix}", file=sys.stderr)
        return 1
    finally:
        await consumer.stop()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv[1])))
