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
        """生成三个U型障碍物，如果生成失败则重试"""
        # 最大重试次数限制
        MAX_RETRIES = 10
        if retry_count >= MAX_RETRIES:
            return False

        if start is None or end is None:
            return False

        # 清空图像为白色
        img_temp = np.ones_like(img) * 255

        # 调整U型障碍物参数 - 根据图片调整尺寸
        params = {
            'wall_thickness': 5,  # U形墙壁厚度
            'u_shape_width': 50,  # U形的宽度
            'middle_length': 120,  # U形的长度
            'margin': 25,  # 边界距离
            'safe_radius': 20,  # 与起点终点的安全距离
            'min_distance': 30  # 障碍物间最小距离
        }

        def generate_obstacle_positions():
            """生成三个U型障碍物的位置配置"""
            # 将地图划分为3x3的网格
            width_third = self.width // 3
            height_third = self.height // 3

            # 在不同区域生成障碍物
            configs = [
                # 左上区域
                (
                    params['margin'] + random.randint(0, 10),
                    params['margin'] + random.randint(0, 10),
                    params['u_shape_width'],
                    params['middle_length'],
                    random.choice([0, 1])  # 上开或右开
                ),
                # 中间区域
                (
                    width_third + random.randint(-10, 10),
                    height_third + random.randint(-10, 10),
                    params['u_shape_width'],
                    params['middle_length'],
                    random.choice([0, 2])  # 上开或下开
                ),
                # 右下区域
                (
                    self.width - params['margin'] - params['u_shape_width'] - random.randint(0, 10),
                    self.height - params['margin'] - params['middle_length'] - random.randint(0, 10),
                    params['u_shape_width'],
                    params['middle_length'],
                    random.choice([2, 3])  # 下开或左开
                )
            ]
            return configs

        def draw_u_shape(x, y, width, length, direction):
            """绘制U形障碍物"""
            t = params['wall_thickness']

            if direction == 0:  # 上开
                cv2.rectangle(img_temp, (x, y + t), (x + t, y + length), (0, 0, 0), -1)
                cv2.rectangle(img_temp, (x + width - t, y + t), (x + width, y + length), (0, 0, 0), -1)
                cv2.rectangle(img_temp, (x, y + length - t), (x + width, y + length), (0, 0, 0), -1)
                return (x, y, width, length)

            elif direction == 1:  # 右开
                cv2.rectangle(img_temp, (x, y), (x + length - t, y + t), (0, 0, 0), -1)
                cv2.rectangle(img_temp, (x, y + width - t), (x + length - t, y + width), (0, 0, 0), -1)
                cv2.rectangle(img_temp, (x, y), (x + t, y + width), (0, 0, 0), -1)
                return (x, y, length, width)

            elif direction == 2:  # 下开
                cv2.rectangle(img_temp, (x, y), (x + t, y + length - t), (0, 0, 0), -1)
                cv2.rectangle(img_temp, (x + width - t, y), (x + width, y + length - t), (0, 0, 0), -1)
                cv2.rectangle(img_temp, (x, y), (x + width, y + t), (0, 0, 0), -1)
                return (x, y, width, length)

            else:  # 左开
                cv2.rectangle(img_temp, (x + t, y), (x + length, y + t), (0, 0, 0), -1)
                cv2.rectangle(img_temp, (x + t, y + width - t), (x + length, y + width), (0, 0, 0), -1)
                cv2.rectangle(img_temp, (x + length - t, y), (x + length, y + width), (0, 0, 0), -1)
                return (x, y, length, width)

        def check_valid_position(x, y, width, height):
            """检查位置是否在有效范围内"""
            return (x >= params['margin'] and
                    y >= params['margin'] and
                    x + width <= self.width - params['margin'] and
                    y + height <= self.height - params['margin'])

        def check_safe_distance(x, y, width, height, start, end):
            """检查与起点终点的安全距离"""

            def distance(p1, p2):
                return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

            corners = [
                (x, y), (x + width, y),
                (x, y + height), (x + width, y + height)
            ]

            for corner in corners:
                if (distance(corner, start) < params['safe_radius'] or
                        distance(corner, end) < params['safe_radius']):
                    return False
            return True

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

        # 尝试生成障碍物
        for attempt in range(3):
            obstacle_boxes = []
            img_temp = np.ones_like(img) * 255

            # 获取障碍物配置
            configs = generate_obstacle_positions()
            all_valid = True

            # 尝试放置每个障碍物
            for x, y, width, length, direction in configs:
                if not check_valid_position(x, y, width, length):
                    all_valid = False
                    break

                if not check_safe_distance(x, y, width, length, start, end):
                    all_valid = False
                    break

                box = draw_u_shape(x, y, width, length, direction)
                if box:
                    obstacle_boxes.append(box)
                else:
                    all_valid = False
                    break

            # 验证生成结果
            if all_valid and len(obstacle_boxes) == 3:
                if check_path_exists(start, end):
                    img[:] = img_temp
                    return True

        # 如果当前尝试失败，递归重试
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
            save_dir = os.path.join(desktop, "4.4")
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