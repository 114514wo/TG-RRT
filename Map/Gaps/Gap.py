import numpy as np
import cv2
import os
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox  
import random
from math import sqrt
 
class MapGenerator:
    def __init__(self):
        self.width = 224
        self.height = 224

    def generate_point(self, is_start=True):
     
        size = 5
        if is_start:
            x = random.randint(size, self.width // 2 - size)
        else:
            x = random.randint(self.width // 2 + size, self.width - size)
        y = random.randint(size, self.height - size)
        return (x, y)

    def check_distance(self, p1, p2):

        return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) >= 150

    def draw_point(self, img, center, color):

        x, y = center
        cv2.rectangle(img, (x - 2, y - 2), (x + 2, y + 2), color, -1)

    def generate_rectangle(self, img, density, start=None, end=None):

        img_temp = np.ones_like(img) * 255

        if start is None or end is None:
            return False

        def draw_gapped_rectangle(img, x, y, width, height, gap_y):

            cv2.rectangle(img, (x, y), (x + width, gap_y), (0, 0, 0), -1)

            cv2.rectangle(img, (x, gap_y + gap_size),
                          (x + width, y + height), (0, 0, 0), -1)

        def check_path_exists(img, start_point, end_point):

            if start_point is None or end_point is None:
                return False

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


        wall_thickness = 30  
        gap_size = 40  

        mid_x = (start[0] + end[0]) // 2

        if start[0] < end[0]:
            obstacle_x = mid_x - wall_thickness // 2
        else:
            obstacle_x = mid_x - wall_thickness // 2

        obstacle_height = 180
        obstacle_y = 20

        if start[1] < end[1]:
   
            gap_y = obstacle_y + obstacle_height // 2
        else:
 
            gap_y = obstacle_y + obstacle_height // 3

        draw_gapped_rectangle(img_temp, obstacle_x, obstacle_y,
                              wall_thickness, obstacle_height, gap_y)

        if check_path_exists(img_temp, start, end):
            img[:] = img_temp
            return True
        else:
 
            gap_y = obstacle_y + obstacle_height // 2
            img_temp = np.ones_like(img) * 255
            draw_gapped_rectangle(img_temp, obstacle_x, obstacle_y,
                                  wall_thickness, obstacle_height, gap_y)
            img[:] = img_temp
            return True

        return False

    def generate_circles(self, img, density):

        num_circles = int((self.width * self.height) * density / 2000)
        for _ in range(num_circles):
            radius = random.randint(5, 15)
            x = random.randint(radius, self.width - radius)
            y = random.randint(radius, self.height - radius)
            cv2.circle(img, (x, y), radius, (0, 0, 0), -1)

    def generate_map(self, obstacle_type, density):

        img = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255

        while True:
            start = self.generate_point(True)
            end = self.generate_point(False)
            if self.check_distance(start, end):
                break

        if obstacle_type == "圆形":
            self.generate_circles(img, density)

            if not ((img[start[1], start[0]] == 255).all() and
                    (img[end[1], end[0]] == 255).all()):
                return self.generate_map(obstacle_type, density)
        elif obstacle_type == "矩形":
            self.generate_rectangle(img, density, start, end)
        else:  
            self.generate_circles(img, density / 2)
            self.generate_rectangle(img, density / 2, start, end)

            if not ((img[start[1], start[0]] == 255).all() and
                    (img[end[1], end[0]] == 255).all()):
                return self.generate_map(obstacle_type, density)

        self.draw_point(img, start, (0, 255, 0))  
        self.draw_point(img, end, (255, 0, 0))  

        return img


class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("地图生成器")
        self.setup_gui()
        self.map_generator = MapGenerator()

    def setup_gui(self):

        tk.Label(self.root, text="生成数量:").grid(row=0, column=0)
        self.num_maps = tk.Entry(self.root)
        self.num_maps.insert(0, "1")
        self.num_maps.grid(row=0, column=1)

        tk.Label(self.root, text="障碍物密度(1-10):").grid(row=1, column=0)
        self.density = tk.Scale(self.root, from_=1, to=10, orient=tk.HORIZONTAL)
        self.density.grid(row=1, column=1)

        tk.Label(self.root, text="障碍物类型:").grid(row=2, column=0)
        self.obstacle_type = ttk.Combobox(self.root, values=["圆形", "矩形", "混合"])
        self.obstacle_type.set("圆形")
        self.obstacle_type.grid(row=2, column=1)

        tk.Button(self.root, text="生成地图", command=self.generate).grid(row=3, column=0, columnspan=2)

    def generate(self):
        try:
            num = int(self.num_maps.get())
            density = self.density.get()
            obs_type = self.obstacle_type.get()

            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            save_dir = os.path.join(desktop, "1.1")
            os.makedirs(save_dir, exist_ok=True)

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
