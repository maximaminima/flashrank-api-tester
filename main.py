from fastapi import FastAPI, Request
from flashrank import Ranker, RerankRequest
from pydantic import BaseModel
from typing import List
from flashrank import Ranker, RerankRequest
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import json
import numpy as np

class Passage(BaseModel):
   id: int
   text: str
   meta: dict = {}



ranker = Ranker()

query = "How to speedup LLMs?"
passages = [
   {
      "id":1,
      "text":"Introduce *lookahead decoding*: - a parallel decoding algo to accelerate LLM inference - w/o the need for a draft model or a data store - linearly decreases # decoding steps relative to log(FLOPs) used per decoding step.",
      "meta": {"additional": "info1"}
   },
   {
      "id":2,
      "text":"LLM inference efficiency will be one of the most crucial topics for both industry and academia, simply because the more efficient you are, the more $$$ you will save. vllm project is a must-read for this direction, and now they have just released the paper",
      "meta": {"additional": "info2"}
   },
   {
      "id":3,
      "text":"There are many ways to increase LLM inference throughput (tokens/second) and decrease memory footprint, sometimes at the same time. Here are a few methods I’ve found effective when working with Llama 2. These methods are all well-integrated with Hugging Face. This list is far from exhaustive; some of these techniques can be used in combination with each other and there are plenty of others to try. - Bettertransformer (Optimum Library): Simply call `model.to_bettertransformer()` on your Hugging Face model for a modest improvement in tokens per second. - Fp4 Mixed-Precision (Bitsandbytes): Requires minimal configuration and dramatically reduces the model's memory footprint. - AutoGPTQ: Time-consuming but leads to a much smaller model and faster inference. The quantization is a one-time cost that pays off in the long run.",
      "meta": {"additional": "info3"}

   },
   {
      "id":4,
      "text":"Ever want to make your LLM inference go brrrrr but got stuck at implementing speculative decoding and finding the suitable draft model? No more pain! Thrilled to unveil Medusa, a simple framework that removes the annoying draft model while getting 2x speedup.",
      "meta": {"additional": "info4"}
   },
   {
      "id":5,
      "text":"vLLM is a fast and easy-to-use library for LLM inference and serving. vLLM is fast with: State-of-the-art serving throughput Efficient management of attention key and value memory with PagedAttention Continuous batching of incoming requests Optimized CUDA kernels",
      "meta": {"additional": "info5"}
   }
]


mypassage = [Passage(**passage) for passage in passages]

finalpassage = [passage.model_dump() for passage in mypassage]

app = FastAPI()

ranker = Ranker()





class Item(BaseModel):
    query: str

class Result(BaseModel):
    id: int
    text: str
    meta: dict
    score: float

@app.post("/rank")
async def rank(request: Request, response_model=List[Result]):
    data = await request.json()
    item = Item(**data)
    print(item)
    mypassage = [Passage(**passage) for passage in passages]
    finalpassage = [passage.model_dump() for passage in mypassage]
    rerankrequest = RerankRequest(query=item.dict()["query"], passages=finalpassage)
    result = ranker.rerank(rerankrequest)
    print(result)
    final_result = []
    for itemi in result:
        final_result.append({
            "id": itemi.get("id"),
            "text": itemi.get("text"),
            "meta": itemi.get("meta"),
            "score": np.float64(itemi.get("score"))
        }
        )
    print(final_result)
    return JSONResponse(content=final_result)
