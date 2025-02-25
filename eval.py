import argparse
import numpy as np
from qdrant_client import QdrantClient, models
import time
from expected import full_scan_result
import os


QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = "laion"


client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=True)


def query_point_id(query_id: int, rescore_limit = 1000, limit = 500):
    response = client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        query=query_id,
        limit=limit,
        search_params=models.SearchParams(
            quantization=models.QuantizationSearchParams(
                rescore=True,
            ),
        ),
        # Prefetch guarantees that we only re-score specified number of points independently of the number of segments
        prefetch=models.Prefetch( 
            query=query_id,
            limit=rescore_limit,
            params=models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    rescore=False,
                ),
            )
        )
    )

    return [point.id for point in response.points]


def eval(rescore_limit = 1000):
    precisions = []
    for idx, expected_result in enumerate(full_scan_result):

        time_start = time.time()
        search_result = query_point_id(idx, rescore_limit=rescore_limit)
        elapsed_time = time.time() - time_start
        intersection = len(set(search_result) & set(expected_result))

        precision = intersection/len(expected_result)
        precisions.append(precision)
        print(f"Intersection: {intersection}/{len(expected_result)}, which is {precision*100:.2f}%, elapsed time: {elapsed_time:.2f}s")
        average_precision = np.mean(precisions)

    print(f"Average precision: {average_precision*100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Evaluate the search quality')
    parser.add_argument('--rescore_limit', type=int, default=1000, help='The number of points to rescore')
    args = parser.parse_args()
    eval(args.rescore_limit)

if __name__ == "__main__":
    main()