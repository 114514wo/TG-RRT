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
        """生成随机位置和方向的U形障碍物，增加连接部分长度"""
        # 清空图像为白色
        img_temp = np.ones_like(img) * 255

        if start is None or end is None:
            return False

        def is_safe_distance(x, y, width, height, start, end, safe_radius=20):
            """检查障碍物是否与起点终点保持安全距离"""
            obstacle_points = []
            for i in range(x, x + width + 1):
                obstacle_points.extend([(i, y), (i, y + height)])
            for i in range(y, y + height + 1):
                obstacle_points.extend([(x, i), (x + width, i)])

            for px, py in obstacle_points:
                if np.sqrt((px - start[0]) ** 2 + (py - start[1]) ** 2) < safe_radius:
                    return False
                if np.sqrt((px - end[0]) ** 2 + (py - end[1]) ** 2) < safe_radius:
                    return False
            return True

        def draw_u_shape(img, x, y, width, middle_length, thickness, direction):
            """绘制U形障碍物，width为两端的长度（更长），middle_length为中间部分长度"""
            if direction == 0:  # 上开
                # 绘制左竖线
                cv2.rectangle(img, (x, y + thickness),
                              (x + thickness, y + middle_length),
                              (0, 0, 0), -1)
                # 绘制右竖线
                cv2.rectangle(img, (x + width - thickness, y + thickness),
                              (x + width, y + middle_length),
                              (0, 0, 0), -1)
                # 绘制底横线
                cv2.rectangle(img, (x, y + middle_length - thickness),
                              (x + width, y + middle_length),
                              (0, 0, 0), -1)
                return (x, y, width, middle_length)

            elif direction == 1:  # 右开
                # 绘制上横线
                cv2.rectangle(img, (x, y),
                              (x + middle_length - thickness, y + thickness),
                              (0, 0, 0), -1)
                # 绘制下横线
                cv2.rectangle(img, (x, y + width - thickness),
                              (x + middle_length - thickness, y + width),
                              (0, 0, 0), -1)
                # 绘制左竖线
                cv2.rectangle(img, (x, y),
                              (x + thickness, y + width),
                              (0, 0, 0), -1)
                return (x, y, middle_length, width)

            elif direction == 2:  # 下开
                # 绘制左竖线
                cv2.rectangle(img, (x, y),
                              (x + thickness, y + middle_length - thickness),
                              (0, 0, 0), -1)
                # 绘制右竖线
                cv2.rectangle(img, (x + width - thickness, y),
                              (x + width, y + middle_length - thickness),
                              (0, 0, 0), -1)
                # 绘制上横线
                cv2.rectangle(img, (x, y),
                              (x + width, y + thickness),
                              (0, 0, 0), -1)
                return (x, y, width, middle_length)

            else:  # 左开
                # 绘制上横线
                cv2.rectangle(img, (x + thickness, y),
                              (x + middle_length, y + thickness),
                              (0, 0, 0), -1)
                # 绘制下横线
                cv2.rectangle(img, (x + thickness, y + width - thickness),
                              (x + middle_length, y + width),
                              (0, 0, 0), -1)
                # 绘制右竖线
                cv2.rectangle(img, (x + middle_length - thickness, y),
                              (x + middle_length, y + width),
                              (0, 0, 0), -1)
                return (x, y, middle_length, width)

        def check_path_exists(img, start_point, end_point):
            """检查是否存在可通行路径"""
            visited = np.zeros_like(img[:, :, 0], dtype=bool)
            queue = [(start_point[0], start_point[1])]
            visited[start_point[1], start_point[0]] = True

            directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                          (1, 1), (1, -1), (-1, 1), (-1, -1)]

            while queue:
                current = queue.pop(0)
                if current == (end_point[0], end_point[1]):
                    return True

                for dx, dy in directions:
                    new_x = current[0] + dx
                    new_y = current[1] + dy

                    if (0 <= new_x < img.shape[1] and
                            0 <= new_y < img.shape[0] and
                            not visited[new_y, new_x] and
                            (img[new_y, new_x] == 255).all()):
                        queue.append((new_x, new_y))
                        visited[new_y, new_x] = True
            return False

        # 调整障碍物参数
        wall_thickness = 20  # 墙壁厚度
        u_shape_width = 160  # U形两端的长度（进一步增加）
        middle_length = 100  # 中间部分长度
        margin = 30  # 边界margin
        safe_radius = 25  # 与起点终点的安全距离

        max_attempts = 50
        for attempt in range(max_attempts):
            img_temp = np.ones_like(img) * 255

            direction = random.randint(0, 3)

            if direction in [0, 2]:  # 上开或下开
                max_x = img.shape[1] - u_shape_width - margin
                max_y = img.shape[0] - middle_length - margin
            else:  # 左开或右开
                max_x = img.shape[1] - middle_length - margin
                max_y = img.shape[0] - u_shape_width - margin

            obstacle_x = random.randint(margin, max_x)
            obstacle_y = random.randint(margin, max_y)

            if direction in [0, 2]:
                if not is_safe_distance(obstacle_x, obstacle_y,
                                        u_shape_width, middle_length,
                                        start, end, safe_radius):
                    continue
            else:
                if not is_safe_distance(obstacle_x, obstacle_y,
                                        middle_length, u_shape_width,
                                        start, end, safe_radius):
                    continue

            draw_u_shape(img_temp, obstacle_x, obstacle_y,
                         u_shape_width, middle_length,
                         wall_thickness, direction)

            if check_path_exists(img_temp, start, end):
                img[:] = img_temp
                return True

        # 如果所有尝试都失败，生成一个安全的默认U形
        img_temp = np.ones_like(img) * 255
        safe_x = margin
        safe_y = margin
        while not is_safe_distance(safe_x, safe_y, u_shape_width, middle_length,
                                   start, end, safe_radius):
            safe_x += 20
            safe_y += 20
            if safe_x >= img.shape[1] - u_shape_width or safe_y >= img.shape[0] - middle_length:
                safe_x = margin
                safe_y = margin
                break

        draw_u_shape(img_temp, safe_x, safe_y,
                     u_shape_width, middle_length,
                     wall_thickness, 0)
        img[:] = img_temp
        return True

    def generate_circles(self, img, density):
        """生成圆形障碍物"""
        # 同样将密度值调小
        num_circles = int((self.width * self.height) * density / 2000)
        for _ in range(num_circles):
            radius = random.randint(5, 15)
            x = random.randint(radius, self.width - radius)
            y = random.randint(radius, self.height - radius)
            cv2.circle(img, (x, y), radius, (0, 0, 0), -1)

    def generate_map(self, obstacle_type, density):
        """生成单张地图"""
        img = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255

        # 先生成起点和终点
        while True:
            start = self.generate_point(True)
            end = self.generate_point(False)
            if self.check_distance(start, end):
                break

        # 生成障碍物
        if obstacle_type == "圆形":
            self.generate_circles(img, density)
            # 检查起点终点是否与障碍物重合
            if not ((img[start[1], start[0]] == 255).all() and
                    (img[end[1], end[0]] == 255).all()):
                return self.generate_map(obstacle_type, density)
        elif obstacle_type == "矩形":
            self.generate_rectangle(img, density, start, end)
        else:  # 混合
            self.generate_circles(img, density / 2)
            self.generate_rectangle(img, density / 2, start, end)
            # 检查起点终点是否与障碍物重合
            if not ((img[start[1], start[0]] == 255).all() and
                    (img[end[1], end[0]] == 255).all()):
                return self.generate_map(obstacle_type, density)

        # 绘制起点和终点
        self.draw_point(img, start, (0, 255, 0))  # 绿色起点
        self.draw_point(img, end, (255, 0, 0))  # 蓝色终点

        return img


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
            save_dir = os.path.join(desktop, "5")
            os.makedirs(save_dir, exist_ok=True)

            # 生成并保存地图
            for i in range(num):
                img = self.map_generator.generate_map(obs_type, density)
                cv2.imwrite(os.path.join(save_dir, f"map_{i + 1}.png"), img)

            messagebox.showinfo("成功", f"已生成{num}张地图并保存至桌面Map文件夹")  # 修改这行
        except Exception as e:
            messagebox.showerror("错误", str(e))  # 修改这行

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    gui = GUI()
    gui.run()