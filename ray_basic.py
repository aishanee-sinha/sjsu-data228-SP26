import ray
import time

# 1. Ray 시작 (로컬 클러스터 생성)
ray.init(include_dashboard=True)

# 2. 병렬 실행할 함수
@ray.remote
def work(x):
    time.sleep(2)
    return x * x

# 3. 병렬 실행
start = time.time()

futures = [work.remote(i) for i in range(5)]
results = ray.get(futures)

end = time.time()

print("Results:", results)
print("Time taken:", round(end - start, 2), "seconds")

time.sleep(100)
