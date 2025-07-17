import numpy as np

def schwefel(x):
    """Schwefel 函數"""
    x = np.clip(x, -500, 500)  # 限制範圍
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

# 測試
D = 30  # 維度
x_optimal = [420.97] * D
print("Theoretical optimal value:", schwefel(x_optimal))  # 應接近 0
