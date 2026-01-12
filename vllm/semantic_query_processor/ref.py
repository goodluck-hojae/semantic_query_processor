

import asyncio
import aiohttp
import pandas as pd
import re
from typing import Iterator

import json
import numpy as np
import requests
from dataclasses import dataclass, field
from typing import List, Dict, Any
import sys, os
sys.path.insert(0, os.path.expanduser("~/project/vllm-test/vllm"))

from vllm import LLM, SamplingParams
import vllm

# =========================
# Utils
# =========================
def normalize(text: str) -> str:
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()



def load_resumes(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        encoding="utf-8",
        low_memory=False,
        nrows=40
        # nrows=41
    )

    df["data"] = (
        df["Resume_str"]
        .map(normalize)
    )

    return df





# def filter_map_conseq(resumes):

#     url = "http://localhost:8000/v1/completions"
#     headers = {
#         "Content-Type": "application/json",
#         "Authorization": "Bearer EMPTY"
#     } 

#     # Filter
#     prompts = []
#     for text in resumes["data"]: 
#         prompt = f"{text}\n\nIs the candidate capable of GPU programming?\n"
#         print(len(prompt), end= ' ')
#         prompts.append(prompt)

#     payload = {
#         "model": "meta-llama/Llama-3.1-8B-Instruct",
#         "prompt": prompts,
#         "max_tokens": 10,
#         "temperature": 0,
#     }

#     start = time.time()
#     r = requests.post(url, headers=headers, json=payload, timeout=6000)
#     r.raise_for_status()
#     print("Filter time:", time.time() - start)

#     for i, choice in enumerate(r.json()["choices"]):
#         if i % 20 == 0:
#             print(f"Prompt {i}:", choice["text"][:10])

#     # Filter
#     prompts = []
#     for text in resumes["data"]: 
#         prompt = f"{text}\n\nSummarize the candidate's experience in one paragraph.\n"
#         print(len(prompt), end= ' ')
#         prompts.append(prompt)

#     payload = {
#         "model": "meta-llama/Llama-3.1-8B-Instruct",
#         "prompt": prompts,
#         "max_tokens": 128,
#         "temperature": 0,
#     }

#     start = time.time()
#     r = requests.post(url, headers=headers, json=payload, timeout=6000)
#     r.raise_for_status()
#     print("Map time:", time.time() - start)

#     for i, choice in enumerate(r.json()["choices"]):
#         if i % 20 == 0:
#             print(f"Prompt {i}:", choice["text"][:10])


# if __name__ == "__main__":
#     # Load resumes (example)
#     import pandas as pd
#     import requests
#     import time

#     SERVER_URL = "http://localhost:8000/semantic_query"

#     RESUME_PATH = "/home/hojaeson_umass_edu/.cache/kagglehub/datasets/snehaanbhawal/resume-dataset/versions/1/Resume/Resume.csv"
#     resumes = load_resumes(RESUME_PATH)

#     start = time.time()
#     # filter_map(resumes)
#     filter_map_conseq(resumes)
#     print("\nTotal time:", time.time() - start)
#     start = time.time()
#     # filter_map(resumes)
#     filter_map(resumes)
#     print("\nPrefix Total time:", time.time() - start)
#     print(vllm.__version__)
#     print(vllm.__file__)
 

import asyncio
import aiohttp
import time

URL = "http://localhost:8000/v1/completions"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer EMPTY",
}
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
BATCH_SIZE = 10


async def post(session, payload):
    async with session.post(
        URL,
        json=payload,
        timeout=aiohttp.ClientTimeout(total=6000),
    ) as r:
        r.raise_for_status()
        return await r.json()



# def filter_map(resumes):

#     url = "http://localhost:8000/v1/completions"
#     headers = {
#         "Content-Type": "application/json",
#         "Authorization": "Bearer EMPTY"
#     } 

#     # Filter
#     batch_size = 20

#     for i in range(0, len(resumes), batch_size):
#         batch_prompts = []
#         for text in resumes["data"][i:i+batch_size]: 
#             prompt = f"{text}\n\nIs the candidate capable of GPU programming?\n"
#             print(len(prompt), end= ' ')
#             batch_prompts.append(prompt)

#         payload = {
#             "model": "meta-llama/Llama-3.1-8B-Instruct",
#             "prompt": batch_prompts,
#             "max_tokens": 10,
#             "temperature": 0,
#         }

#         start = time.time()
#         r = requests.post(url, headers=headers, json=payload, timeout=6000)
#         r.raise_for_status()
#         print("Filter time:", time.time() - start)

        
#         batch_prompts = []
#         for text in resumes["data"][i:i+batch_size]: 
#             prompt = f"{text}\n\nSummarize the candidate's experience in one paragraph.\n"
#             print(len(prompt), end= ' ')
#             batch_prompts.append(prompt)

#         payload = {
#             "model": "meta-llama/Llama-3.1-8B-Instruct",
#             "prompt": batch_prompts,
#             "max_tokens": 128,
#             "temperature": 0,
#         }

#         start = time.time()
#         r = requests.post(url, headers=headers, json=payload, timeout=6000)
#         r.raise_for_status()
#         print("Map time:", time.time() - start)



async def filter_map_with_delay(resumes, batch_size=10, delay_sec=2):
    async with aiohttp.ClientSession(headers=HEADERS) as session:
        for i in range(0, len(resumes), batch_size):
            texts = resumes["data"][i:i + batch_size]

            filter_payload = {
                "model": MODEL,
                "prompt": [
                    f"{t}\n\nIs the candidate capable of GPU programming?\n"
                    for t in texts
                ],
                "max_tokens": 10,
                "temperature": 0,
            }

            map_payload = {
                "model": MODEL,
                "prompt": [
                    f"{t}\n\nSummarize the candidate's experience in one paragraph.\n"
                    for t in texts
                ],
                "max_tokens": 128,
                "temperature": 0,
            }

            t0 = time.time()

            # 1. create filter task
            filter_task = asyncio.create_task(post(session, filter_payload))

            # 2. delay
            await asyncio.sleep(delay_sec)

            # 3. create map task
            map_task = asyncio.create_task(post(session, map_payload))

            # 4. wait for BOTH together
            filter_resp, map_resp = await asyncio.gather(
                filter_task,
                map_task,
            )

            print("Batch wall time:", time.time() - t0)



async def filter_map_async_one(resumes, batch_size=BATCH_SIZE):
    async with aiohttp.ClientSession(headers=HEADERS) as session:
        for i in range(0, len(resumes), batch_size):
            texts = resumes["data"][i : i + batch_size]

            # -------- Filter --------
            filter_prompts = [
                f"{text}\n\nIs the candidate capable of GPU programming? You should answer 'yes' or 'no'\n"
                for text in texts
            ]
 

            # -------- Map --------
            map_prompts = [
                f"{text}\n\nSummarize the candidate's experience in one paragraph.\n"
                for text in texts
            ]

            map_payload = {
                "model": MODEL,
                "prompt": filter_prompts+map_prompts,
                "max_tokens": 128,
                "temperature": 0,
            }

            t0 = time.time()
            resp = await post(session, map_payload)

            for i, choice in enumerate(resp["choices"]):
                print(f"Prompt {i}:", choice["text"][:10])
            print("Map time:", time.time() - t0)




if __name__ == "__main__":
    import time
    import vllm

    RESUME_PATH = "/home/hojaeson_umass_edu/.cache/kagglehub/datasets/snehaanbhawal/resume-dataset/versions/1/Resume/Resume.csv"
    resumes = load_resumes(RESUME_PATH)
    
    delay = 1.5
    # start = time.time()
    # batch_size = 40
    # print("\n batch size:", batch_size)
    # asyncio.run(filter_map_with_delay(resumes, batch_size, delay))
    # print("\n total time:", time.time() - start)

    # start = time.time()
    # batch_size = 40
    # print("\n batch size:", batch_size)
    # asyncio.run(filter_map_with_delay(resumes, batch_size, delay))
    # print("\n total time:", time.time() - start)

    start = time.time()
    batch_size = 25
    print("\n batch size:", batch_size)
    asyncio.run(filter_map_with_delay(resumes, batch_size, delay))
    print("\n total time:", time.time() - start)

    # start = time.time()
    # batch_size = 25
    # print("\n batch size:", batch_size)
    # asyncio.run(filter_map_with_delay(resumes, batch_size, delay))
    # print("\n total time:", time.time() - start)

    # start = time.time()
    # batch_size = 30
    # print("\n batch size:", batch_size)
    # asyncio.run(filter_map_with_delay(resumes, batch_size, delay))
    # print("\n total time:", time.time() - start)

    # start = time.time()
    # batch_size = 15
    # print("\n batch size:", batch_size)
    # asyncio.run(filter_map_with_delay(resumes, batch_size, delay))
    # print("\n total time:", time.time() - start)

