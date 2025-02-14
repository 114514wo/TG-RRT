import numpy as np
import cv2
import os
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox  # 添加这行导入
import random
from math import sqrt

class MapGenerator:
    def __init__(self):
        self.width = 224
        self.height = 224

    def generate_point(self, is_start=True):
        """生成起点或终点"""
        size = 5
        if is_start:
            x = random.randint(size, self.width // 2 - size)
        else:
            x = random.randint(self.width // 2 + size, self.width - size)
        y = random.randint(size, self.height - size)
        return (x, y)

    def check_distance(self, p1, p2):
        """检查两点间距离"""
        return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) >= 150

    def draw_point(self, img, center, color):
        """绘制5x5的起点或终点"""
        x, y = center
        cv2.rectangle(img, (x - 2, y - 2), (x + 2, y + 2), color, -1)

    def generate_rectangle(self, img, density, start=None, end=None, retry_count=0):
        """生成两个长矩形和2-3处离散方块区域"""
        MAX_RETRIES = 10
        if retry_count >= MAX_RETRIES or start is None or end is None:
            return False

        img_temp = np.ones_like(img) * 255

        params = {
            'rect_width': 20,  # 保持长矩形的宽度
            'rect_length': 120,  # 显著增加长矩形的长度
            'square_size': 25,  # 进一步增加散点方块的大小
            'margin': 25,  # 边界距离
            'safe_radius': 25,  # 安全距离
            'min_distance': 30  # 障碍物间最小距离
        }

        def check_path_exists(start_point, end_point):
            """检查是否存在可行路径"""
            visited = np.zeros_like(img_temp[:, :, 0], dtype=bool)
            queue = [(start_point[0], start_point[1])]
            visited[start_point[1], start_point[0]] = True

            directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                          (1, 1), (1, -1), (-1, 1), (-1, -1)]

            while queue:
                cx, cy = queue.pop(0)
                if (cx, cy) == (end_point[0], end_point[1]):
                    return True

                for dx, dy in directions:
                    nx, ny = cx + dx, cy + dy
                    if (0 <= nx < self.width and
                            0 <= ny < self.height and
                            not visited[ny, nx] and
                            (img_temp[ny, nx] == 255).all()):
                        queue.append((nx, ny))
                        visited[ny, nx] = True
            return False

        def generate_rectangles():
            """生成三个长矩形的位置"""
            width_third = self.width // 3
            height_third = self.height // 3

            # 第一个长矩形（左上区域）
            rect1 = {
                'x': params['margin'],
                'y': params['margin'],
                'width': params['rect_length'],
                'height': params['rect_width']
            }

            # 第二个长矩形（中间区域）
            rect2 = {
                'x': width_third + random.randint(-10, 10),
                'y': height_third + random.randint(-10, 10),
                'width': params['rect_length'],
                'height': params['rect_width']
            }

            # 第三个长矩形（右下区域）
            rect3 = {
                'x': self.width - params['margin'] - params['rect_length'],
                'y': self.height - params['margin'] - params['rect_width'] - random.randint(-10, 10),
                'width': params['rect_length'],
                'height': params['rect_width']
            }

            return [rect1, rect2, rect3]

        def generate_scattered_squares():
            """生成2-3处离散方块区域，每处至少3个方块"""
            squares = []
            num_clusters = random.randint(2, 3)

            # 调整可能的区域，确保方块群不会与长矩形重叠
            regions = [
                (self.width // 4, self.height * 2 // 3),  # 左下区域
                (self.width * 3 // 4, self.height // 3),  # 右上区域
                (self.width // 2, self.height * 3 // 4)  # 中下区域
            ]

            selected_regions = random.sample(regions, num_clusters)

            for center_x, center_y in selected_regions:
                # 每个区域固定生成3-4个大方块
                num_squares = random.randint(3, 4)  # 确保至少3个方块

                # 生成方块的相对位置模式
                patterns = [
                    [(0, 0), (1, 0), (0, 1)],  # L形
                    [(0, 0), (1, 0), (1, 1)],  # L形变体
                    [(0, 0), (1, 0), (2, 0)],  # 水平线
                    [(0, 0), (0, 1), (0, 2)]  # 垂直线
                ]

                # 随机选择一个基础模式
                base_pattern = random.choice(patterns)

                # 应用基础模式生成前三个方块
                for dx, dy in base_pattern:
                    square = {
                        'x': center_x + dx * (params['square_size'] + 5) + random.randint(-3, 3),
                        'y': center_y + dy * (params['square_size'] + 5) + random.randint(-3, 3),
                        'size': params['square_size']
                    }
                    squares.append(square)

                # 如果需要第四个方块，在基础模式附近随机添加
                if num_squares == 4:
                    extra_dx = random.randint(0, 2)
                    extra_dy = random.randint(0, 2)
                    square = {
                        'x': center_x + extra_dx * (params['square_size'] + 5) + random.randint(-3, 3),
                        'y': center_y + extra_dy * (params['square_size'] + 5) + random.randint(-3, 3),
                        'size': params['square_size']
                    }
                    squares.append(square)

            return squares

        def draw_obstacles(rectangles, squares):
            """绘制所有障碍物"""
            # 绘制长矩形
            for rect in rectangles:
                cv2.rectangle(img_temp,
                              (rect['x'], rect['y']),
                              (rect['x'] + rect['width'], rect['y'] + rect['height']),
                              (0, 0, 0), -1)

            # 绘制散点方块
            for square in squares:
                cv2.rectangle(img_temp,
                              (square['x'], square['y']),
                              (square['x'] + square['size'], square['y'] + square['size']),
                              (0, 0, 0), -1)

        def check_valid_position(obstacles):
            """检查所有障碍物位置是否有效"""
            for obs in obstacles:
                x = obs.get('x', 0)
                y = obs.get('y', 0)
                width = obs.get('width', obs.get('size', 0))
                height = obs.get('height', obs.get('size', 0))

                if not (0 <= x < self.width - width and
                        0 <= y < self.height - height):
                    return False
            return True

        def check_safe_distance(obstacles):
            """检查障碍物与起点终点的安全距离"""
            for obs in obstacles:
                x = obs.get('x', 0)
                y = obs.get('y', 0)
                width = obs.get('width', obs.get('size', 0))
                height = obs.get('height', obs.get('size', 0))

                corners = [
                    (x, y), (x + width, y),
                    (x, y + height), (x + width, y + height)
                ]

                for corner in corners:
                    dist_start = sqrt((corner[0] - start[0]) ** 2 + (corner[1] - start[1]) ** 2)
                    dist_end = sqrt((corner[0] - end[0]) ** 2 + (corner[1] - end[1]) ** 2)
                    if dist_start < params['safe_radius'] or dist_end < params['safe_radius']:
                        return False
            return True

        # 尝试生成障碍物
        for attempt in range(3):
            img_temp = np.ones_like(img) * 255

            # 生成长矩形
            rectangles = generate_rectangles()
            # 生成散点方块
            squares = generate_scattered_squares()

            # 合并所有障碍物
            all_obstacles = rectangles + squares

            # 检查位置是否有效
            if not check_valid_position(all_obstacles):
                continue

            # 检查与起点终点的安全距离
            if not check_safe_distance(all_obstacles):
                continue

            # 绘制障碍物
            draw_obstacles(rectangles, squares)

            # 检查是否存在可行路径
            if check_path_exists(start, end):
                img[:] = img_temp
                return True

        return self.generate_rectangle(img, density, start, end, retry_count + 1)

    def generate_circles(self, img, density):
        """生成圆形障碍物"""
        # 同样将密度值调小
        num_circles = int((self.width * self.height) * density / 2000)
        for _ in range(num_circles):
            radius = random.randint(5, 15)
            x = random.randint(radius, self.width - radius)
            y = random.randint(radius, self.height - radius)
            cv2.circle(img, (x, y), radius, (0, 0, 0), -1)

    def generate_map(self, obstacle_type, density, max_attempts=50):
        """生成单张有效的地图"""
        for attempt in range(max_attempts):
            img = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255

            # 生成起点和终点
            while True:
                start = self.generate_point(True)
                end = self.generate_point(False)
                if self.check_distance(start, end):
                    break

            # 生成障碍物
            success = True
            if obstacle_type == "圆形":
                self.generate_circles(img, density)
            elif obstacle_type == "矩形":
                if not self.generate_rectangle(img, density, start, end):
                    success = False
            else:  # 混合
                self.generate_circles(img, density / 2)
                if not self.generate_rectangle(img, density / 2, start, end):
                    success = False

            if not success:
                continue

            # 检查是否存在有效的障碍物且不与起点终点重合
            if not self.check_obstacles_exist(img, start, end):
                continue

            # 检查起点终点是否在白色区域
            if not ((img[start[1], start[0]] == 255).all() and
                    (img[end[1], end[0]] == 255).all()):
                continue

            # 绘制起点和终点
            self.draw_point(img, start, (0, 255, 0))
            self.draw_point(img, end, (255, 0, 0))

            return img

        # 如果达到最大尝试次数仍未生成有效地图，递归重试
        return self.generate_map(obstacle_type, density)

    def check_obstacles_exist(self, img, start, end):
        """检查地图中是否存在有效的障碍物，且不与起点终点重合"""
        # 计算黑色像素(障碍物)的数量
        black_pixels = np.sum(img[:, :, 0] == 0)
        total_pixels = self.width * self.height

        # 设置障碍物比例阈值
        min_obstacle_ratio = 0.05  # 最小障碍物比例
        max_obstacle_ratio = 0.3  # 最大障碍物比例

        obstacle_ratio = black_pixels / total_pixels

        # 检查障碍物比例是否在合理范围内
        if obstacle_ratio < min_obstacle_ratio or obstacle_ratio > max_obstacle_ratio:
            return False

        # 检查起点终点周围是否有障碍物
        safe_radius = 10  # 安全距离半径

        def check_point_safety(point):
            x, y = point
            for dx in range(-safe_radius, safe_radius + 1):
                for dy in range(-safe_radius, safe_radius + 1):
                    check_x = x + dx
                    check_y = y + dy
                    if (0 <= check_x < self.width and
                            0 <= check_y < self.height and
                            img[check_y, check_x, 0] == 0):  # 检查是否有黑色像素
                        return False
            return True

        # 检查起点和终点周围是否安全
        if not check_point_safety(start) or not check_point_safety(end):
            return False

        # 使用连通区域分析检查障碍物
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 检查是否存在足够大的连通区域
        min_contour_area = 100  # 最小连通区域面积
        valid_obstacles = False

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_contour_area:
                # 检查障碍物是否与起点终点重合
                mask = np.zeros_like(gray)
                cv2.drawContours(mask, [contour], -1, 255, -1)

                # 检查起点终点是否在障碍物区域内
                if mask[start[1], start[0]] == 0 and mask[end[1], end[0]] == 0:
                    valid_obstacles = True
                    break

        return valid_obstacles


class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("地图生成器")
        self.setup_gui()
        self.map_generator = MapGenerator()

    def setup_gui(self):
        # 数量选择
        tk.Label(self.root, text="生成数量:").grid(row=0, column=0)
        self.num_maps = tk.Entry(self.root)
        self.num_maps.insert(0, "1")
        self.num_maps.grid(row=0, column=1)

        # 密度选择
        tk.Label(self.root, text="障碍物密度(1-10):").grid(row=1, column=0)
        self.density = tk.Scale(self.root, from_=1, to=10, orient=tk.HORIZONTAL)
        self.density.grid(row=1, column=1)

        # 障碍物类型选择
        tk.Label(self.root, text="障碍物类型:").grid(row=2, column=0)
        self.obstacle_type = ttk.Combobox(self.root, values=["圆形", "矩形", "混合"])
        self.obstacle_type.set("圆形")
        self.obstacle_type.grid(row=2, column=1)

        # 生成按钮
        tk.Button(self.root, text="生成地图", command=self.generate).grid(row=3, column=0, columnspan=2)

    def generate(self):
        try:
            num = int(self.num_maps.get())
            density = self.density.get()
            obs_type = self.obstacle_type.get()

            # 创建保存目录
            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            save_dir = os.path.join(desktop, "Map")
            os.makedirs(save_dir, exist_ok=True)

            # 记录成功生成的地图数量
            success_count = 0
            max_attempts = num * 3  # 最大尝试次数是请求数量的3倍

            for i in range(max_attempts):
                if success_count >= num:
                    break

                img = self.map_generator.generate_map(obs_type, density)
                if img is not None:
                    cv2.imwrite(os.path.join(save_dir, f"map_{success_count + 1}.png"), img)
                    success_count += 1

            if success_count < num:
                messagebox.showwarning("警告",
                                       f"只能生成{success_count}张有效地图，少于请求的{num}张")
            else:
                messagebox.showinfo("成功",
                                    f"已生成{success_count}张地图并保存至桌面Map文件夹")

        except Exception as e:
            messagebox.showerror("错误", str(e))

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    gui = GUI()
    gui.run()