# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import asyncio
import requests
from typing import Optional, List, Optional
from functools import partial
import tqdm.asyncio
import requests
import os
import requests.exceptions
import mimetypes
import glob
import uuid
from typing import Optional


async def clip_search_images_by_text(
        text: str,
        service_url: str = "https://knn.laion.ai/knn-service",
        num_images: int = 40,
        indice_name: str = "laion5B-L-14",
        num_result_ids: int = 3000,
        use_mclip: bool = False,
        deduplicate: bool = True,
        use_safety_model: bool = True,
        use_violence_detector: bool = True,
        aesthetic_score: Optional[float] = None,
        aesthetic_weight: float = 0.5
    ):

    loop = asyncio.get_event_loop()

    response = await loop.run_in_executor(
        None,
        partial(
            requests.post,
            service_url,
            json={
                "text": text,
                "image": None,
                "image_url": None,
                "embedding_input": None,
                "modality": "image",
                "num_images": num_images,
                "indice_name": indice_name,
                "num_result_ids": num_result_ids,
                "use_mclip": use_mclip,
                "deduplicate": deduplicate,
                "use_safety_model": use_safety_model,
                "use_violence_detector": use_violence_detector,
                "aesthetic_score": "" if aesthetic_score is None else str(aesthetic_score),
                "aesthetic_weight": str(aesthetic_weight)
            }
        )
    )

    urls = [
        item['url'] for item in response.json() if 'url' in item
    ]

    return urls


async def clip_search_images_by_multi_text(
        texts: List[str],
        service_url: str = "https://knn.laion.ai/knn-service",
        num_images: int = 40,
        indice_name: str = "laion5B-L-14",
        num_result_ids: int = 3000,
        use_mclip: bool = False,
        deduplicate: bool = True,
        use_safety_model: bool = True,
        use_violence_detector: bool = True,
        aesthetic_score: Optional[float] = None,
        aesthetic_weight: float = 0.5,
        max_workers: int = 1
    ):

    # executor = ThreadPoolExecutor(max_workers=num_workers)
    semaphore = asyncio.Semaphore(max_workers)

    async def safe_coro(coro):
        async with semaphore:
            return await coro

    coros = []
    for text in texts:
        coros.append(
            safe_coro(
                clip_search_images_by_text(
                    text=text,
                    service_url=service_url,
                    num_images=num_images,
                    indice_name=indice_name,
                    num_result_ids=num_result_ids,
                    use_mclip=use_mclip,
                    deduplicate=deduplicate,
                    use_safety_model=use_safety_model,
                    use_violence_detector=use_violence_detector,
                    aesthetic_score=aesthetic_score,
                    aesthetic_weight=aesthetic_weight
                )
            )
        )

    results = await asyncio.gather(*coros)

    return sum(results, [])


def url_to_image_id(url: str):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, url))


def find_files_matching_image_id(folder: str, id: str):
    return glob.glob(os.path.join(folder, id + ".*"))


def image_with_id_exists(folder: str, id: str):
    return len(find_files_matching_image_id(folder, id)) > 0


def download_image(url: str, output_folder: str, timeout = None, verify=True):
    
    image_id = str(uuid.uuid5(uuid.NAMESPACE_URL, url))

    response = requests.get(url, timeout=timeout, verify=verify)

    content_type = response.headers['content-type']
    extension = mimetypes.guess_extension(content_type)

    if extension is None:
        raise RuntimeError("Extension type could not be determined.")

    if extension not in [".jpg", ".png"]:
        raise RuntimeError("Invalid image extension.")

    filename = image_id + extension
    
    full_path = os.path.join(output_folder, filename)

    with open(full_path, 'wb') as f:
        f.write(response.content)

    return filename
    

async def download_image_async(url: str, output_folder: str, timeout=None, verify=True):

    loop = asyncio.get_event_loop()

    result = await loop.run_in_executor(
        None,
        partial(
            download_image,
            url,
            output_folder,
            timeout,
            verify
        )
    )

    return result


async def download_many_images_async(
        urls: List[str], 
        output_folder: str, 
        max_workers: int = 1, 
        timeout=None, 
        verify=True
    ):

    skipped_urls = set()
    urls_to_download = set()
    for url in urls:
        if not image_with_id_exists(output_folder, url_to_image_id(url)):
            urls_to_download.add(url)
        else:
            skipped_urls.add(url)

    semaphore = asyncio.Semaphore(max_workers)

    async def safe_coro(coro, url):
        async with semaphore:
            try:
                result = await coro
                return True, result, url, None
            except BaseException as error:
                return False, "", url, error

    tasks = [
        asyncio.create_task(safe_coro(download_image_async(url, output_folder, timeout, verify), url))
        for url in urls_to_download
    ]

    for f in tqdm.asyncio.tqdm.as_completed(tasks):
        await f

    failed_urls = set()
    downloaded_urls = set()
    error_map_count = {}
    for task in tasks:
        success, filename, url, error = task.result()
        if success:
            downloaded_urls.add(url)
        else:
            if error.__class__ in error_map_count:
                error_map_count[error.__class__] += 1
            else:
                error_map_count[error.__class__] = 1
            failed_urls.add(url)

    return downloaded_urls, failed_urls, skipped_urls, error_map_count


def parse_multiple_input(values):
    all_values = []
    for val in values:
        for inner_val in val:
            all_values.append(inner_val)
    return all_values

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input_urls_file", type=str, help="Path to file containing image urls.")
    parser.add_argument("output_folder", type=str, help="Path to folder to store downloaded images.")
    parser.add_argument("--max_workers", type=int, default=16, help="Number of parallel workers.")
    parser.add_argument("--timeout", type=float, default=2., help="Timeout for each image.")
    parser.add_argument("--verify", type=bool, default=True, help="Enable / disable SSL verification.")
    args = parser.parse_args()

    with open(args.input_urls_file, 'r') as f:
        urls = f.readlines()
    urls = [u.strip("\n") for u in urls]
    
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    unique_urls = set()
    for url in urls:
        unique_urls.add(url)
    print(unique_urls)
    print(f"Found {len(unique_urls)} unique urls in {args.input_urls_file}")

    print(f"Downloading images to {args.output_folder}...")
    downloaded_urls, failed_urls, skipped_urls, error_map_count = asyncio.run(
        download_many_images_async(
            urls=list(unique_urls),
            output_folder=args.output_folder,
            max_workers=args.max_workers,
            timeout=args.timeout,
            verify=args.verify
        )
    )
    print(f"Skipped downloading of {len(skipped_urls)} urls. (Already existed.)")
    print(f"Successfully downloaded {len(downloaded_urls)} urls.")
    print(f"Failed to download {len(failed_urls)} urls. ({error_map_count})")
    