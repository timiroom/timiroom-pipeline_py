import asyncio
from types import SimpleNamespace

from phase1.embedding_service import EmbeddingService


def test_embedding_requests_are_batched_and_response_order_is_restored() -> None:
    class FakeEmbeddings:
        def __init__(self):
            self.calls = []

        async def create(self, model, input):
            self.calls.append((model, input))
            data = [
                SimpleNamespace(index=index, embedding=[float(value)])
                for index, value in enumerate(input)
            ]
            return SimpleNamespace(data=list(reversed(data)))

    async def run() -> None:
        service = EmbeddingService("test-key", batch_size=2)
        await service._client.close()
        fake_embeddings = FakeEmbeddings()
        service._client = SimpleNamespace(embeddings=fake_embeddings)

        vectors = await service.embed(["1", "2", "3", "4", "5"])

        assert vectors == [[1.0], [2.0], [3.0], [4.0], [5.0]]
        assert [len(call[1]) for call in fake_embeddings.calls] == [2, 2, 1]
        assert all(call[0] == "solar-embedding-2-passage" for call in fake_embeddings.calls)

    asyncio.run(run())
