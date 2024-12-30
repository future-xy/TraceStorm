import random
from typing import Any, Dict, List

from tracestorm.constants import DEFAULT_MESSAGES
from tracestorm.logger import init_logger

logger = init_logger(__name__)

def generate_request(
    model_name: str, nums: int, datasets: List = None, sort: str = "random", messages: str = DEFAULT_MESSAGES
) -> List[Dict[str, Any]]:
    requests = []
    
    # If no datasets, generate default requests
    if not datasets:
        for _ in range(nums):
            requests.append(
                {
                    "model": model_name,
                    "messages": [{"role": "user", "content": messages}],
                    "stream": True,
                }
            )
    else: # Add and sort requests from the provided datasets
        dataset_allocation = []
        requests_left = nums
        
        # Total ratio to calculate number of requests for each dataset
        total_ratio = sum(ratio for _, _, ratio, _ in datasets)
        
        for name, samples, ratio, num_samples in datasets:
            num_requests = int(round(nums * ratio / total_ratio))
            num_requests = min(num_requests, num_samples) # Consider available samples
            requests_left -= num_requests
            
            # Store prompts with indexing for round-robin
            # For example, if ratio of dataset1 is 5, we will append 5 requests for each idx
            for i, sample in enumerate(samples[:num_requests]):
                idx = i // ratio
                dataset_allocation.append((idx, sample))
                
            logger.info(f"Select {num_requests} requests from {name}, there are {requests_left} requests left") 
        
        # 1. Randomly sort the requests
        if sort == "random":    
            random.seed(88)
            random.shuffle(dataset_allocation)
        else: # 2. Sort by index for round-robin selection
            dataset_allocation.sort(key=lambda x: x[0])
            
        # Extract the prompts from the list
        requests = [
            {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
            }
            for _, prompt in dataset_allocation
        ]
        
        # Add default requests if requests_left > 0
        if requests_left > 0:
            for _ in range(requests_left): 
                requests.append(
                    {
                        "model": model_name,
                        "messages": [{"role": "user", "content": messages}],
                        "stream": True,
                    }
                )
                
    return requests
