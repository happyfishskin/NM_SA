import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
import time


class NelderMeadOptimizer:
    def __init__(self, max_iter=200000, tol=1e-9):
        """初始化 Nelder-Mead 優化參數"""
        self.max_iter = max_iter  # 最大迭代次數
        self.tol = tol  # 收斂容忍度
        self.history = []  # 用於記錄歷史點
        self.best_point = None  # 全局最佳解
        self.best_value = float('inf')  # 全局最佳值

    def eval_schwefel(self, x):
        """Schwefel 函數"""
        x = np.clip(x, -500, 500)  # 限制範圍在 [-500, 500]
        return 418.9829 * len(x) - sum(x_i * np.sin(np.sqrt(abs(x_i))) for x_i in x)

    def callback(self, xk):
        """記錄每次迭代的解，並更新全局最佳值"""
        xk = np.clip(xk, -500, 500)  # 確保變數範圍
        self.history.append(np.copy(xk))  # 記錄歷史點
        # 同步更新全局最佳值
        current_value = self.eval_schwefel(xk)
        if current_value < self.best_value:
            self.best_point = np.copy(xk)
            self.best_value = current_value

    def optimize(self, initial_point):
        """執行 Nelder-Mead 優化"""
        self.history = []  # 清空歷史記錄
        result = minimize(
            self.eval_schwefel,  # 目標函數
            initial_point,  # 初始點
            method='Nelder-Mead',  # 使用 Nelder-Mead 方法
            callback=self.callback,  # 設置回調函數
            options={'maxiter': self.max_iter, 'xatol': self.tol, 'fatol': self.tol, 'disp': True}
        )
        return result

    def plot(self, result):
        """繪製 Schwefel 函數的 3D 表面圖和搜索歷史"""
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 創建曲面的網格數據
        x = np.linspace(-500, 500, 100)
        y = np.linspace(-500, 500, 100)
        X, Y = np.meshgrid(x, y)
        Z = 418.9829 * 2 - (X * np.sin(np.sqrt(abs(X))) + Y * np.sin(np.sqrt(abs(Y))))  # Schwefel 函數

        # 繪製 Schwefel 函數曲面
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)

        # 繪製歷史點
        x_vals = [point[0] for point in self.history]
        y_vals = [point[1] for point in self.history]
        z_vals = [self.eval_schwefel([x, y]) for x, y in zip(x_vals, y_vals)]
        ax.scatter(x_vals, y_vals, z_vals, c='blue', marker='o', s=50, label='Search History')

        # 繪製最佳解
        best_x, best_y = result.x[0], result.x[1]
        best_z = result.fun
        ax.scatter(best_x, best_y, best_z, c='red', marker='^', s=150, label='Best Solution')

        # 添加最佳解的標註
        ax.text(best_x, best_y, best_z + 100, f"({best_x:.2f}, {best_y:.2f}, {best_z:.2f})", color='red')

        # 設置標題與軸標籤
        ax.set_title('3D Surface Plot of Schwefel Function with Search History')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        plt.legend()
        plt.show()


# 主程式
if __name__ == "__main__":
    optimizer = NelderMeadOptimizer(max_iter=200000, tol=1e-9)

    # 初始化隨機起點
    np.random.seed(int(time.time()))
    initial_point = np.random.uniform(-500, 500, 10)  # 30 維變數

    # 執行 Nelder-Mead 優化
    result = optimizer.optimize(initial_point)

    # 找出最接近0的變數及其索引
    closest_to_zero = min(result.x, key=abs)
    closest_index = np.argmin(np.abs(result.x))

    # 輸出結果
    print(f"最佳解: {result.x}")
    print(f"最佳值（越接近0越好）: {result.fun}")
    print(f"最佳解中每個變數的範圍：最小值 = {np.min(result.x)}, 最大值 = {np.max(result.x)}")
    print(f"最接近0的變數：{closest_to_zero}（索引：{closest_index}）")

    # 繪製結果
    optimizer.plot(result)
