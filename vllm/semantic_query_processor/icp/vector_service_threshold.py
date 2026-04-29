import argparse
from contextlib import asynccontextmanager
from typing import Any

import faiss
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


def _normalize_tuple(values: list[Any] | tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(values)


def _serialize_tuple(t: tuple[Any, ...]) -> str:
    return " ".join(map(str, t))


# -----------------------------
# Embedding Model
# -----------------------------
class SentenceTransformersRM:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )


# -----------------------------
# FAISS Vector Store (NO k)
# -----------------------------
class FaissVS:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: list[tuple[Any, ...]] = []

    def add(self, embeddings: np.ndarray, metadata: list[tuple[Any, ...]]) -> None:
        if len(embeddings) != len(metadata):
            raise ValueError("Embeddings and metadata must have the same length.")
        if len(embeddings) == 0:
            return

        self.index.add(embeddings.astype("float32"))
        self.metadata.extend(metadata)

    def search(
        self,
        query_embedding: np.ndarray,
        threshold: float,
    ) -> list[dict[str, Any]]:
        if self.index.ntotal == 0:
            return []

        query = np.array([query_embedding]).astype("float32")
        lims, D, I = self.index.range_search(query, threshold)

        results: list[dict[str, Any]] = []
        start, end = lims[0], lims[1]

        for idx in range(start, end):
            results.append({
                "metadata": list(self.metadata[I[idx]]),
                "score": float(D[idx]),
            })

        # sort for deterministic output
        results.sort(key=lambda x: x["score"], reverse=True)
        return results


# -----------------------------
# Vector DB Service
# -----------------------------
class VectorDBService:
    def __init__(self, rm: SentenceTransformersRM):
        self.rm = rm
        self.cache: dict[str, FaissVS] = {}
        self.table_cache: dict[str, list[tuple[Any, ...]]] = {}

    def build_index(
        self,
        cp_id: str,
        right_table: list[tuple[Any, ...]],
    ) -> FaissVS:
        cached_table = self.table_cache.get(cp_id)

        if cp_id in self.cache:
            if cached_table != right_table:
                raise ValueError(
                    f"Cached index for cp_id={cp_id!r} built with different table."
                )
            return self.cache[cp_id]

        if not right_table:
            raise ValueError("right_table must not be empty.")

        texts = [_serialize_tuple(row) for row in right_table]
        embeddings = self.rm.encode(texts)

        vs = FaissVS(embeddings.shape[1])
        vs.add(embeddings, right_table)

        self.cache[cp_id] = vs
        self.table_cache[cp_id] = list(right_table)
        return vs

    def query(
        self,
        cp_id: str,
        left_tuple: tuple[Any, ...],
        right_table: list[tuple[Any, ...]] | None,
        threshold: float=0.5,
    ) -> list[dict[str, Any]]:
        if right_table is not None:
            vs = self.build_index(cp_id, right_table)
        else:
            vs = self.cache.get(cp_id)
            if vs is None:
                raise ValueError(f"No index for cp_id={cp_id!r}")

        query_text = _serialize_tuple(left_tuple)
        query_vec = self.rm.encode([query_text])[0]

        return vs.search(query_vec, threshold)

    def clear_cp(self, cp_id: str) -> bool:
        removed = False
        if cp_id in self.cache:
            del self.cache[cp_id]
            removed = True
        if cp_id in self.table_cache:
            del self.table_cache[cp_id]
            removed = True
        return removed

    def cache_state(self) -> dict[str, int]:
        return {
            cp_id: vs.index.ntotal
            for cp_id, vs in self.cache.items()
        }


# -----------------------------
# API Models
# -----------------------------
class BuildIndexRequest(BaseModel):
    cp_id: str
    right_table: list[list[Any]]


class QueryRequest(BaseModel):
    cp_id: str
    left_tuple: list[Any]
    right_table: list[list[Any]] | None = None
    threshold: float


class ClearRequest(BaseModel):
    cp_id: str


# -----------------------------
# FastAPI App
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.vector_db = VectorDBService(
        SentenceTransformersRM(app.state.model_name)
    )
    yield


def create_app(model_name: str = "intfloat/e5-base-v2") -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.state.model_name = model_name

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/cache")
    async def cache():
        return {"cache": app.state.vector_db.cache_state()}

    @app.post("/build_index")
    async def build_index(request: BuildIndexRequest):
        try:
            right_table = [_normalize_tuple(r) for r in request.right_table]
            vs = app.state.vector_db.build_index(request.cp_id, right_table)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"rows_indexed": vs.index.ntotal}

    @app.post("/query")
    async def query(request: QueryRequest):
        try:
            results = app.state.vector_db.query(
                request.cp_id,
                _normalize_tuple(request.left_tuple),
                None if request.right_table is None else [
                    _normalize_tuple(r) for r in request.right_table
                ],
                request.threshold,
            )
        except ValueError as e:
            raise HTTPException(400, str(e))

        return {"results": results}

    @app.post("/clear")
    async def clear(request: ClearRequest):
        return {"cleared": app.state.vector_db.clear_cp(request.cp_id)}

    return app


# -----------------------------
# Run
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--model-name", default="intfloat/e5-base-v2")
    args = parser.parse_args()

    app = create_app(args.model_name)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()