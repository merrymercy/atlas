import time
import ray

ray.init()

@ray.remote
def test_func(arg_1, arg_2):
    None

tb = time.time()
futures = [test_func.remote(None, None) for _ in range(32)]
values = ray.get(futures)
te = time.time()
print("T1: ", te - tb)

tb = time.time()
futures = [test_func.remote(None, None) for _ in range(32)]
values = ray.get(futures)
te = time.time()
print("T2: ", te - tb)

