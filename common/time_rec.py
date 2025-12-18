import time
from functools import wraps


import time
from collections import defaultdict
from statistics import mean, stdev
import functools


class AdvancedTimer:
    """高级计时器，支持多次测量和统计分析"""
    
    def __init__(self, name: str = "计时器"):
        self.name = name
        self.timings = []
        self.current_start = None
        self.history = []
        self.call_count = 0
    
    def start(self):
        """开始单次计时"""
        self.current_start = time.perf_counter()
        return self
    
    def stop(self) -> float:
        """停止单次计时并记录"""
        if self.current_start is None:
            raise RuntimeError("计时器未启动")
        
        elapsed = time.perf_counter() - self.current_start
        self.timings.append(elapsed)
        self.current_start = None
        self.call_count += 1
        self.history.append({
            'timestamp': time.time(),
            'duration': elapsed,
            'call_number': self.call_count
        })
        return elapsed
    
    def lap(self) -> float:
        """记录一个分段计时并继续计时"""
        if self.current_start is None:
            raise RuntimeError("计时器未启动")
        
        elapsed = time.perf_counter() - self.current_start
        self.timings.append(elapsed)
        self.call_count += 1
        # 重置开始时间，继续计时
        self.current_start = time.perf_counter()
        return elapsed
    
    def get_statistics(self):
        """获取统计信息"""
        if not self.timings:
            return None
        
        return {
            'name': self.name,
            'total_calls': len(self.timings),
            'total_time': sum(self.timings),
            'mean': mean(self.timings),
            'min': min(self.timings),
            'max': max(self.timings),
            'last': self.timings[-1] if self.timings else 0,
            'std_dev': stdev(self.timings) if len(self.timings) > 1 else 0
        }
    
    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_statistics()
        if stats:
            print(f"\n=== {stats['name']} 统计 ===")
            print(f"调用次数: {stats['total_calls']}")
            print(f"总耗时: {stats['total_time']:.6f} 秒")
            print(f"平均耗时: {stats['mean']:.6f} 秒")
            print(f"最短耗时: {stats['min']:.6f} 秒")
            print(f"最长耗时: {stats['max']:.6f} 秒")
            print(f"上次耗时: {stats['last']:.6f} 秒")
            print(f"标准差: {stats['std_dev']:.6f} 秒")
    
    def reset(self):
        """重置计时器"""
        self.timings.clear()
        self.history.clear()
        self.current_start = None
        self.call_count = 0
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def timer_decorator(name: str = None, verbose: bool = True):
    """装饰器版本的计时器"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            timer_name = name or func.__name__
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start_time
            
            if verbose:
                print(f"[{timer_name}] 耗时: {elapsed:.6f} 秒")
            
            return result
        return wrapper
    return decorator


# 使用示例
def example_advanced():
    # 创建高级计时器
    timer = AdvancedTimer("性能测试")
    
    # 多次测量
    for i in range(5):
        timer.start()
        time.sleep(0.1 * (i + 1))  # 模拟不同耗时
        elapsed = timer.stop()
        print(f"第{i+1}次执行耗时: {elapsed:.6f} 秒")
    
    # 打印统计信息
    timer.print_statistics()
    
    # 使用装饰器
    @timer_decorator("计算函数")
    def calculate_sum(n):
        return sum(range(n))
    
    result = calculate_sum(1000000)
    print(f"计算结果: {result}")


import time
from contextlib import ContextDecorator
from dataclasses import dataclass, field
from typing import Optional, Callable, Any
import functools


class Timer:
    """基础计时器类"""
    
    def __init__(self, name: str = "计时器", verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        if self.verbose:
            print(f"{self.name} - 耗时: {self.elapsed_time:.4f} 秒")
    
    def start(self):
        """开始计时"""
        self.start_time = time.perf_counter()
        return self
    
    def stop(self):
        """停止计时并返回耗时"""
        if self.start_time is None:
            raise RuntimeError("计时器未启动")
        
        self.end_time = time.perf_counter()
        self.elapsed_time = self.end_time - self.start_time
        return self.elapsed_time
    
    def reset(self):
        """重置计时器"""
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
    
    def get_elapsed_time(self):
        """获取当前耗时（不停止计时）"""
        if self.start_time is None:
            raise RuntimeError("计时器未启动")
        
        current_time = time.perf_counter()
        return current_time - self.start_time


# 使用示例
def example_basic():
    # 方法1：使用上下文管理器
    with Timer("测试代码块1"):
        time.sleep(0.5)
    
    # 方法2：手动控制
    timer = Timer("测试代码块2", verbose=True)
    timer.start()
    time.sleep(0.3)
    elapsed = timer.stop()
    print(f"手动计时: {elapsed:.4f} 秒")
    
    # 方法3：测量函数执行时间
    def expensive_function(n):
        result = 0
        for i in range(n):
            result += i * i
        return result
    
    with Timer("计算函数耗时"):
        result = expensive_function(1000000)
        print(f"计算结果: {result}")

def timeit(func=None, *, repeat: int = 1, verbose: bool = True):
    """
    计时装饰器
    
    Args:
        func: 被装饰的函数
        repeat: 重复执行次数
        verbose: 是否打印结果
    """
    if func is None:
        return lambda f: timeit(f, repeat=repeat, verbose=verbose)
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        timings = []
        
        for i in range(repeat):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            timings.append(end_time - start_time)
        
        avg_time = sum(timings) / len(timings)
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"函数: {func.__name__}")
            print(f"参数: args={args}, kwargs={kwargs}")
            print(f"重复次数: {repeat}")
            print(f"平均耗时: {avg_time:.6f} 秒")
            if repeat > 1:
                print(f"最快: {min(timings):.6f} 秒")
                print(f"最慢: {max(timings):.6f} 秒")
            print(f"{'='*50}")
        
        return result
    
    return wrapper


def measure_time(description: str = "代码块"):
    """测量代码块执行时间的上下文管理器"""
    class MeasureTime:
        def __init__(self, desc):
            self.desc = desc
        
        def __enter__(self):
            self.start = time.perf_counter()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.elapsed = time.perf_counter() - self.start
            print(f"{self.desc} 耗时: {self.elapsed:.6f} 秒")
    
    return MeasureTime(description)


# 使用示例
def example_lightweight():
    # 使用装饰器
    @timeit(repeat=3)
    def process_data(data_size):
        """模拟数据处理"""
        data = list(range(data_size))
        return sum(x * x for x in data)
    
    # 测试不同大小的数据
    for size in [1000, 10000, 100000]:
        process_data(size)
    
    # 使用上下文管理器
    with measure_time("复杂计算"):
        result = 0
        for i in range(1000000):
            result += i ** 0.5
        print(f"计算结果: {result:.2f}")


if __name__ == "__main__":
    # print("=== 基础计时器示例 ===")
    # example_basic()
    
    print("\n=== 高级计时器示例 ===")
    example_advanced()
    
    # print("\n=== 轻量级计时器示例 ===")
    # example_lightweight()