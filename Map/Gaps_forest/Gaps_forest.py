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

    def generate_rectangle(self, img, density, start=None, end=None):
        """生成贯穿画布的长矩形和周围的离散矩形块"""
        img_temp = np.ones_like(img) * 255

        params = {
            'rect_width': 25,  # 长矩形的宽度
            'rect_gap': 30,  # 空缺的宽度
            'small_rect_size': (10, 15),  # 小矩形的大小范围
            'margin': 20,  # 边界距离
            'safe_radius': 25,  # 与起点终点的安全距离
        }

        def generate_main_rectangles():
            """生成两个贯穿画布的长矩形，每个只有一个空缺"""
            rectangles = []

            # 第一个长矩形（1/4位置）
            x1 = int(self.width * 0.25)
            gap_y1 = random.randint(int(self.height * 0.3),
                                    int(self.height * 0.7))  # 空缺位置

            # 第二个长矩形（5/8位置）
            x2 = int(self.width * 0.625)
            gap_y2 = random.randint(int(self.height * 0.3),
                                    int(self.height * 0.7))  # 空缺位置

            # 添加第一个长矩形（分成上下两部分）
            rectangles.append({
                'x': x1,
                'y': 0,  # 从顶部开始
                'width': params['rect_width'],
                'height': gap_y1  # 到空缺位置
            })
            rectangles.append({
                'x': x1,
                'y': gap_y1 + params['rect_gap'],
                'width': params['rect_width'],
                'height': self.height - (gap_y1 + params['rect_gap'])  # 到底部
            })

            # 添加第二个长矩形（分成上下两部分）
            rectangles.append({
                'x': x2,
                'y': 0,
                'width': params['rect_width'],
                'height': gap_y2
            })
            rectangles.append({
                'x': x2,
                'y': gap_y2 + params['rect_gap'],
                'width': params['rect_width'],
                'height': self.height - (gap_y2 + params['rect_gap'])
            })

            return rectangles, [(x1, gap_y1), (x2, gap_y2)]  # 返回矩形列表和空缺位置

        def generate_scattered_rectangles(main_rects_pos):
            """在长矩形周围生成离散的小矩形"""
            scattered = []
            num_scattered = random.randint(5, 12)

            for x_center, gap_y in main_rects_pos:
                # 在每个长矩形的空缺周围生成小矩形
                for _ in range(num_scattered // 2):
                    # 随机选择小矩形的大小
                    width = random.randint(8, 15)
                    height = random.randint(8, 15)

                    # 随机选择位置（左右两侧和空缺周围）
                    side = random.choice([-1, 1])  # -1表示左侧，1表示右侧
                    x_offset = random.randint(30, 50) * side

                    # 在空缺周围生成，范围更集中
                    y_offset = random.randint(-40, 40)

                    x = x_center + x_offset
                    y = gap_y + y_offset

                    # 确保在边界内
                    if (0 <= x <= self.width - width and
                            0 <= y <= self.height - height):
                        scattered.append({
                            'x': x,
                            'y': y,
                            'width': width,
                            'height': height
                        })

            return scattered

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

        # 生成主要的长矩形和散点矩形
        for attempt in range(10):
            img_temp = np.ones_like(img) * 255

            # 生成主要的长矩形
            main_rectangles, gap_positions = generate_main_rectangles()

            # 生成散点矩形
            scattered_rectangles = generate_scattered_rectangles(gap_positions)

            # 绘制所有矩形
            for rect in main_rectangles + scattered_rectangles:
                cv2.rectangle(img_temp,
                              (rect['x'], rect['y']),
                              (rect['x'] + rect['width'],
                               rect['y'] + rect['height']),
                              (0, 0, 0), -1)

            # 检查起点终点
            if start and end:
                if not ((img_temp[start[1], start[0]] == 255).all() and
                        (img_temp[end[1], end[0]] == 255).all()):
                    continue

            # 检查是否存在可行路径
            if start and end and not check_path_exists(start, end):
                continue

            # 所有检查都通过，复制到原图像
            img[:] = img_temp
            return True

        return False

    def generate_circles(self, img, density):
        """生成5-8个大小不同的圆形障碍物"""
        # 圆形参数设置
        params = {
            'min_radius': 15,  # 最小半径
            'max_radius': 30,  # 最大半径
            'min_distance': 20,  # 圆形之间的最小距离
            'safe_radius': 25,  # 与起点终点的安全距离
        }

        def check_circle_overlap(x, y, r, circles):
            """检查圆形是否与其他圆形重叠"""
            for cx, cy, cr in circles:
                if sqrt((x - cx) ** 2 + (y - cy) ** 2) < (r + cr + params['min_distance']):
                    return True
            return False

        def check_circle_valid(x, y, r, start=None, end=None):
            """检查圆形位置是否有效"""
            # 检查是否在边界内
            if not (r <= x <= self.width - r and r <= y <= self.height - r):
                return False

            # 检查与起点终点的距离
            if start and end:
                dist_start = sqrt((x - start[0]) ** 2 + (y - start[1]) ** 2)
                dist_end = sqrt((x - end[0]) ** 2 + (y - end[1]) ** 2)
                if (dist_start < r + params['safe_radius'] or
                        dist_end < r + params['safe_radius']):
                    return False

            return True

        # 随机生成5-8个圆
        num_circles = random.randint(5, 8)
        circles = []  # 存储已生成的圆 (x, y, radius)

        max_attempts = 100  # 最大尝试次数
        for _ in range(num_circles):
            attempts = 0
            while attempts < max_attempts:
                # 生成随机半径
                radius = random.randint(params['min_radius'], params['max_radius'])
                # 生成随机位置
                x = random.randint(radius + 10, self.width - radius - 10)
                y = random.randint(radius + 10, self.height - radius - 10)

                # 检查位置是否有效
                if not check_circle_valid(x, y, radius):
                    attempts += 1
                    continue

                # 检查是否与其他圆重叠
                if check_circle_overlap(x, y, radius, circles):
                    attempts += 1
                    continue

                # 位置有效，添加圆形
                circles.append((x, y, radius))
                # 在图像上绘制圆形
                cv2.circle(img, (x, y), radius, (0, 0, 0), -1)
                break

            if attempts >= max_attempts:
                print(f"Warning: Could not place circle {len(circles) + 1}")

        return len(circles) >= 5  # 返回是否成功生成至少5个圆

    def generate_map(self, obstacle_type, density):
        """生成单张地图"""
        max_attempts = 10  # 最大重试次数

        for attempt in range(max_attempts):
            # 创建白色背景
            img = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255

            # 生成起点和终点
            while True:
                start = self.generate_point(True)
                end = self.generate_point(False)
                if self.check_distance(start, end):
                    break

            # 在起点和终点位置创建安全区域
            safety_radius = 10
            cv2.circle(img, start, safety_radius, (255, 255, 255), -1)
            cv2.circle(img, end, safety_radius, (255, 255, 255), -1)

            # 生成障碍物
            success = True
            if obstacle_type == "圆形":
                if not self.generate_circles(img, density):
                    success = False
            elif obstacle_type == "矩形":
                if not self.generate_rectangle(img, density, start, end):
                    success = False
            else:  # 混合
                # 先生成矩形
                if not self.generate_rectangle(img, density / 2, start, end):
                    success = False
                # 再生成圆形
                if not self.generate_circles(img, density / 2):
                    success = False

            if not success:
                continue

            # 确保起点和终点区域没有障碍物
            start_area = img[start[1] - safety_radius:start[1] + safety_radius + 1,
                         start[0] - safety_radius:start[0] + safety_radius + 1]
            end_area = img[end[1] - safety_radius:end[1] + safety_radius + 1,
                       end[0] - safety_radius:end[0] + safety_radius + 1]

            if (np.any(start_area != 255) or np.any(end_area != 255)):
                continue

            # 检查是否存在可行路径
            if not self.check_path_exists(img, start, end):
                continue

            # 绘制起点和终点
            self.draw_point(img, start, (0, 255, 0))  # 绿色起点
            self.draw_point(img, end, (255, 0, 0))  # 蓝色终点

            return img

        # 如果所有尝试都失败，递归重试
        return self.generate_map(obstacle_type, density)

    def check_path_exists(self, img, start, end):
        """检查起点到终点是否存在可行路径"""
        # 创建访问标记数组
        visited = np.zeros((self.height, self.width), dtype=bool)

        # 定义搜索方向（8个方向）
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                      (1, 1), (1, -1), (-1, 1), (-1, -1)]

        # 使用BFS搜索路径
        queue = [(start[0], start[1])]
        visited[start[1], start[0]] = True

        while queue:
            x, y = queue.pop(0)
            if (x, y) == (end[0], end[1]):
                return True

            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                if (0 <= new_x < self.width and
                        0 <= new_y < self.height and
                        not visited[new_y, new_x] and
                        (img[new_y, new_x] == 255).all()):
                    queue.append((new_x, new_y))
                    visited[new_y, new_x] = True

        return False

    def check_obstacles_exist(self, img):
        """检查地图中是否存在有效的障碍物"""
        # 计算黑色像素(障碍物)的数量
        black_pixels = np.sum(img[:, :, 0] == 0)
        total_pixels = self.width * self.height

        # 设置障碍物比例阈值
        min_obstacle_ratio = 0.05  # 最小障碍物比例
        max_obstacle_ratio = 0.4  # 最大障碍物比例

        obstacle_ratio = black_pixels / total_pixels

        # 检查障碍物比例是否在合理范围内
        if obstacle_ratio < min_obstacle_ratio:
            return False
        if obstacle_ratio > max_obstacle_ratio:
            return False

        # 使用连通区域分析检查是否有完整的障碍物结构
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 检查是否存在足够大的连通区域
        min_contour_area = 100  # 最小连通区域面积
        large_obstacles = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_contour_area:
                large_obstacles += 1

        # 至少要有两个较大的障碍物
        return large_obstacles >= 2


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
                    # 再次检查确保地图有效
                    if self.map_generator.check_obstacles_exist(img):
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
        """运行GUI程序"""
        self.root.mainloop()


if __name__ == "__main__":
    gui = GUI()
    gui.run()