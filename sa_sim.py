import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


class SimulatedAnnealingOptimizer:
    def __init__(self, max_iter=10000, initial_temp=1000, cooling_rate=0.99):
        """初始化模擬退火參數"""
        self.max_iter = max_iter  # 最大迭代次數
        self.initial_temp = initial_temp  # 初始溫度
        self.cooling_rate = cooling_rate  # 降溫速率
        self.history = []  # 用於記錄所有搜索點

    def eval_sphere(self, x):
        """Sphere 函數"""
        return sum(x_i**2 for x_i in x)

    def optimize(self, initial_point):
        """執行模擬退火優化"""
        current_point = np.copy(initial_point)  # 初始化當前點
        current_value = self.eval_sphere(current_point)  # 當前點的目標值
        best_point = np.copy(current_point)  # 初始化最佳點
        best_value = current_value  # 初始化最佳值
        temperature = self.initial_temp  # 設置初始溫度

        self.history = []  # 清空歷史記錄
        self.history.append((np.copy(current_point), current_value))  # 記錄初始點

        for iteration in range(self.max_iter):
            # 生成鄰域點
            neighbor = current_point + np.random.uniform(-10, 10, len(current_point))
            neighbor_value = self.eval_sphere(neighbor)

            # 接受新點的概率計算
            delta = neighbor_value - current_value
            if delta < 0 or np.random.rand() < np.exp(-delta / temperature):
                current_point = neighbor
                current_value = neighbor_value
                self.history.append((np.copy(current_point), current_value))  # 記錄新點

            # 更新最佳解
            if current_value < best_value:
                best_point = np.copy(current_point)
                best_value = current_value

            # 降低溫度
            temperature *= self.cooling_rate

            # 停止條件
            if temperature < 1e-8:
                break

            # 每 100 次輸出進度
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Current Best Value = {best_value}")

        return best_point, best_value

    def plot(self, best_point, best_value):
        """繪製 Sphere 函數的 3D 表面圖和搜索歷史"""
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 創建曲面的網格數據
        x = np.linspace(-100, 100, 100)
        y = np.linspace(-100, 100, 100)
        X, Y = np.meshgrid(x, y)
        Z = X**2 + Y**2  # Sphere 函數

        # 繪製 Sphere 函數曲面
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)

        # 繪製所有歷史點
        x_vals = [point[0][0] for point in self.history]
        y_vals = [point[0][1] for point in self.history]
        z_vals = [point[1] for point in self.history]
        ax.scatter(x_vals, y_vals, z_vals, c='blue', marker='o', s=50, label='Search History')

        # 繪製最佳解
        ax.scatter(best_point[0], best_point[1], best_value, c='red', marker='^', s=150, label='Best Solution')

        # 添加最佳解的標註
        ax.text(best_point[0], best_point[1], best_value + 100,
                f"({best_point[0]:.2f}, {best_point[1]:.2f}, {best_value:.2f})", color='red')

        # 設置標題與軸標籤
        ax.set_title('3D Surface Plot of Sphere Function with Search History')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        plt.legend()
        plt.show()


# 主程式
if __name__ == "__main__":
    optimizer = SimulatedAnnealingOptimizer(max_iter=10000, initial_temp=1000, cooling_rate=0.99)

    # 初始化隨機起點
    np.random.seed(int(time.time()))
    initial_point = np.random.uniform(-100, 100, 10)  # 30 維變數

    # 執行模擬退火優化
    best_point, best_value = optimizer.optimize(initial_point)

    # 找出最接近 0 的變數及其索引
    closest_to_zero = min(best_point, key=abs)  # 找到最接近 0 的值
    closest_index = np.argmin(np.abs(best_point))  # 找到最接近 0 的索引

    # 輸出結果
    print("\n優化完成。")
    print(f"最佳解: {best_point}")
    print(f"最佳值（越接近0越好）: {best_value}")
    print(f"最佳解中每個變數的範圍：最小值 = {np.min(best_point)}, 最大值 = {np.max(best_point)}")
    print(f"最接近0的變數：{closest_to_zero}（索引：{closest_index}）")

    # 繪製結果
    optimizer.plot(best_point, best_value)
