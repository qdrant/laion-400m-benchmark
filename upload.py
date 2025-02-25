# This script downloads LAION dataset data and uploads it to qdrant
import os

import pandas as pd
import numpy as np
import tqdm

from qdrant_client import QdrantClient, models


NUMBER_OF_PARTS = 409

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = "laion"


client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=True)

def get_img_emb_url(part: int) -> str:
    return f"https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings/img_emb/img_emb_{part}.npy"


def get_metadata_url(part: int) -> str:
    return f"https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings/metadata/metadata_{part}.parquet"

def local_img_emb_path(part: int) -> str:
    return os.path.join(DATA_DIR, "laion", f"img_emb_{part}.npy")

def local_metadata_path(part: int) -> str:
    return os.path.join(DATA_DIR, "laion", f"metadata_{part}.parquet")


def download_file(url: str, path: str):
    print(f"Downloading {url} to {path}")
    if not os.path.exists(path):
        os.system(f"wget {url} -O {path}")


def download(idx: int):
    # make dirs
    os.makedirs(os.path.join(DATA_DIR, "laion"), exist_ok=True)
    # download files
    download_file(get_img_emb_url(idx), local_img_emb_path(idx))
    download_file(get_metadata_url(idx), local_metadata_path(idx))

def clear_data(idx):
    os.remove(local_img_emb_path(idx))
    os.remove(local_metadata_path(idx))


def create_collection(force_recreate=False):
    if force_recreate:
        client.delete_collection(QDRANT_COLLECTION_NAME)
    
    if client.collection_exists(QDRANT_COLLECTION_NAME):
        return

    client.create_collection(
        QDRANT_COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=512,
            distance=models.Distance.COSINE,
            datatype=models.Datatype.FLOAT16,
            on_disk=True
        ),
        quantization_config=models.BinaryQuantization(
            binary=models.BinaryQuantizationConfig(
                always_ram=True,
            )
        ),
        optimizers_config=models.OptimizersConfigDiff(
            default_segment_number=2,
            # Bigger size of segments are desired for faster search
            # However it might be slower for indexing
            max_segment_size=5_000_000, 
        ),
        hnsw_config=models.HnswConfigDiff(
            m=6, # decrease M for lower memory usage
            on_disk=False
        ),
    )


def load_batch(offet: int, emb_path: str, metadata_path: str) -> int:
    emb = np.load(emb_path)

    df = pd.read_parquet(metadata_path)

    # Drop `exif` column
    df.drop(columns=["exif"], inplace=True)

    df.fillna(0, inplace=True)
    payloads = df.to_dict(orient="records")

    total_emb = emb.shape[0]

    uploaded = total_emb
    

    # import ipdb

    # # Intercepts any exception and opens the debugger

    # with ipdb.launch_ipdb_on_exception():

    client.upload_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors=emb,
        payload=payloads,
        ids=tqdm.tqdm(range(offet, total_emb + offet)),
        parallel=4,
    )

    return uploaded
    
    
def laod_all(limit):
    idx = 0
    uploaded = 0
    while idx <= limit:
        download(idx) 
        embedding_path = local_img_emb_path(idx)
        metadata_path = local_metadata_path(idx)
        uploaded += load_batch(uploaded, embedding_path, metadata_path)
        clear_data(idx)

        print(f"Uploaded {uploaded} vectors")

        idx += 1


def main():
    create_collection(force_recreate=True)
    laod_all(NUMBER_OF_PARTS)


if __name__ == "__main__":
    main()