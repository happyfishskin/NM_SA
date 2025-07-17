import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


class SimulatedAnnealingOptimizer:
    def __init__(self, max_iter=200000, initial_temp=5000, cooling_rate=0.9):
        """初始化模擬退火演算法參數"""
        self.max_iter = max_iter  # 最大迭代次數
        self.initial_temp = initial_temp  # 初始溫度
        self.cooling_rate = cooling_rate  # 降溫速率
        self.history = []  # 用於記錄所有點的歷史

    def eval_schwefel(self, x):
        """Schwefel函數的評估函數，限制範圍在[-500, 500]"""
        x = np.clip(x, -500, 500)  # 限制每個變數的範圍
        return 418.9829 * len(x) - sum(x_i * np.sin(np.sqrt(abs(x_i))) for x_i in x)

    def optimize(self, initial_point):
        """執行模擬退火優化"""
        current_point = np.clip(initial_point, -500, 500)  # 初始點
        current_value = self.eval_schwefel(current_point)  # 評估初始點
        best_point = np.copy(current_point)
        best_value = current_value
        temperature = self.initial_temp

        self.history = []  # 清空歷史記錄
        self.history.append(np.copy(current_point))

        for iteration in range(self.max_iter):
            # 生成鄰域點
            neighbor = current_point + np.random.uniform(-10, 10, len(current_point))
            neighbor = np.clip(neighbor, -500, 500)
            neighbor_value = self.eval_schwefel(neighbor)

            # 接受新點的概率計算
            delta = neighbor_value - current_value
            if delta < 0 or np.random.rand() < np.exp(-delta / temperature):
                current_point = neighbor
                current_value = neighbor_value
                self.history.append(np.copy(current_point))

            # 更新最佳解
            if current_value < best_value:
                best_point = np.copy(current_point)
                best_value = current_value

            # 降低溫度
            temperature *= self.cooling_rate

            # 停止條件
            if temperature < 1e-8:
                break

        return best_point, best_value

    def plot(self, best_point, best_value, filename="sa_3d_plot.png"):
        """繪製Schwefel函數的3D表面圖以及所有點和最佳解"""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        self.plot_schwefel_surface(ax)

        # 繪製所有點
        x_vals = [point[0] for point in self.history]
        y_vals = [point[1] for point in self.history]
        z_vals = [self.eval_schwefel(point) for point in self.history]
        ax.scatter(x_vals, y_vals, z_vals, c='green', marker='o', s=50, label='All Points')

        # 繪製最佳解位置
        best_x, best_y = best_point[0], best_point[1]
        best_z = best_value
        ax.scatter(best_x, best_y, best_z, c='red', marker='^', s=150, label='Best Solution')

        # 添加最佳解的標註
        ax.text(best_x, best_y, best_z + 200,  # 提高Z坐標，避免重疊
                f"({best_x:.2f}, {best_y:.2f}, {best_z:.2f})",
                color='blue', fontsize=10, weight='bold')

        # 設置視角
        ax.view_init(elev=30, azim=45)

        # 圖像配置
        ax.set_box_aspect([1, 1, 0.5])
        ax.set_title("Simulated Annealing Optimization of Schwefel Function", fontsize=12)
        ax.set_facecolor('white')
        ax.grid(color='gray', linestyle='--', linewidth=0.5)
        plt.legend(loc='upper left')
        plt.show()

    def plot_schwefel_surface(self, ax):
        """繪製Schwefel函數的3D表面"""
        u = np.linspace(-500, 500, 100)
        x, y = np.meshgrid(u, u)
        z = 418.9829 * 2 - (x * np.sin(np.sqrt(abs(x))) + y * np.sin(np.sqrt(abs(y))))
        ax.plot_surface(x, y, z, cmap='viridis', alpha=0.6)
        ax.set_zlim(-10000, 10000)  # 根據Schwefel函數的範圍調整


# 主程式
if __name__ == "__main__":
    optimizer = SimulatedAnnealingOptimizer(max_iter=200000, initial_temp=5000, cooling_rate=0.9)

    # 使用時間作為隨機種子
    np.random.seed(int(time.time()))
    initial_point = np.random.uniform(-500, 500, 10)

    # 執行模擬退火優化
    best_point, best_value = optimizer.optimize(initial_point)

    # 找出最接近0的變數及其索引
    closest_to_zero = min(best_point, key=abs)
    closest_index = np.argmin(np.abs(best_point))

    # 輸出結果
    print(f"最佳解: {best_point}")
    print(f"最佳值（越接近0越好）: {best_value}")
    print(f"最佳解中每個變數的範圍：最小值 = {np.min(best_point)}, 最大值 = {np.max(best_point)}")
    print(f"最接近0的變數：{closest_to_zero}（索引：{closest_index}）")

    # 繪製結果
    optimizer.plot(best_point, best_value)
