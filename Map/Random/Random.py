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
        """生成连续的矩形障碍物"""
        # 清空图像为白色
        img_temp = np.ones_like(img) * 255

        # 生成主要的长方形通道
        num_long_rects = int(density * 2)  # 根据密度确定长方形数量
        min_length = 80
        max_length = 160

        # 保存起点和终点的区域
        start_region = None
        end_region = None
        if start and end:
            start_region = (
                max(0, start[0] - 10),
                min(self.width, start[0] + 10),
                max(0, start[1] - 10),
                min(self.height, start[1] + 10)
            )
            end_region = (
                max(0, end[0] - 10),
                min(self.width, end[0] + 10),
                max(0, end[1] - 10),
                min(self.height, end[1] + 10)
            )

        def is_safe_region(x, y, w, h):
            """检查是否与起点终点区域重叠"""
            if not (start_region and end_region):
                return True
            rect = (x, x + w, y, y + h)
            # 检查与起点区域的重叠
            if not (rect[0] >= start_region[1] or rect[1] <= start_region[0] or
                    rect[2] >= start_region[3] or rect[3] <= start_region[2]):
                return False
            # 检查与终点区域的重叠
            if not (rect[0] >= end_region[1] or rect[1] <= end_region[0] or
                    rect[2] >= end_region[3] or rect[3] <= end_region[2]):
                return False
            return True

        # 生成水平长方形
        for _ in range(num_long_rects):
            width = random.randint(min_length, max_length)
            height = random.randint(15, 25)
            attempts = 0
            while attempts < 10:
                x = random.randint(0, self.width - width)
                y = random.randint(0, self.height - height)
                if is_safe_region(x, y, width, height):
                    cv2.rectangle(img_temp, (x, y), (x + width, y + height), (0, 0, 0), -1)
                    break
                attempts += 1

        # 生成垂直长方形
        for _ in range(num_long_rects):
            width = random.randint(15, 25)
            height = random.randint(min_length, max_length)
            attempts = 0
            while attempts < 10:
                x = random.randint(0, self.width - width)
                y = random.randint(0, self.height - height)
                if is_safe_region(x, y, width, height):
                    cv2.rectangle(img_temp, (x, y), (x + width, y + height), (0, 0, 0), -1)
                    break
                attempts += 1

        # 生成一些连接通道
        num_connectors = int(density * 3)
        for _ in range(num_connectors):
            width = random.randint(20, 40)
            height = random.randint(20, 40)
            x = random.randint(0, self.width - width)
            y = random.randint(0, self.height - height)
            if is_safe_region(x, y, width, height):
                cv2.rectangle(img_temp, (x, y), (x + width, y + height), (0, 0, 0), -1)

        # 确保起点和终点区域是空白的
        if start and end:
            # 清除起点区域的障碍物
            cv2.rectangle(img_temp,
                          (start_region[0], start_region[2]),
                          (start_region[1], start_region[3]),
                          (255, 255, 255), -1)
            # 清除终点区域的障碍物
            cv2.rectangle(img_temp,
                          (end_region[0], end_region[2]),
                          (end_region[1], end_region[3]),
                          (255, 255, 255), -1)

        # 将临时图像复制到原图像
        img[:] = img_temp

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
            save_dir = os.path.join(desktop, "Maps")
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