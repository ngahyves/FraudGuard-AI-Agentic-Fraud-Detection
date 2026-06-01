import requests
import time
import numpy as np

# Configuration
URL = "http://localhost:8000/predict" # We test the core ML latency
PAYLOAD = {
    "transaction_id": 9999,
    "transaction": {
        "step": 1, "type": "TRANSFER", "amount": 5000.0, "category": "Finance",
        "hour": 12, "day_of_week": 1, "day_of_month": 1, "month": 1
    }
}

def run_benchmark(n_requests=100):
    print(f"Starting benchmark: {n_requests} requests to {URL}...")
    latencies = []

    # Warm-up (to avoid cold start bias in metrics)
    for _ in range(5):
        requests.post(URL, json=PAYLOAD)

    for i in range(n_requests):
        start_time = time.time()
        try:
            response = requests.post(URL, json=PAYLOAD)
            if response.status_code == 200:
                duration_ms = (time.time() - start_time) * 1000
                latencies.append(duration_ms)
        except Exception as e:
            print(f"Request {i} failed: {e}")

    if latencies:
        p50 = np.median(latencies)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        
        print("\n" + "="*30)
        print("LATENCY RESULTS (ms)")
        print("="*30)
        print(f"Median (p50): {p50:.2f} ms")
        print(f"95th Percentile (p95): {p95:.2f} ms")
        print(f"Peak (p99): {p99:.2f} ms")
        print("="*30)
        return p99
    return None

if __name__ == "__main__":
    run_benchmark()