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

    def generate_rectangle(self, img, density, start=None, end=None, retry_count=0):
        MAX_RETRIES = 10
        if retry_count >= MAX_RETRIES:
            return False

        if start is None or end is None:
            return False

        img_temp = np.ones_like(img) * 255

        params = {
            'wall_thickness': 5,
            'u_shape_width': 50,
            'middle_length': 120,
            'margin': 25,
            'safe_radius': 20,
            'min_distance': 30
        }

        def generate_obstacle_positions():
            width_third = self.width // 3
            height_third = self.height // 3

            configs = [
                (
                    params['margin'] + random.randint(0, 10),
                    params['margin'] + random.randint(0, 10),
                    params['u_shape_width'],
                    params['middle_length'],
                    random.choice([0, 1])
                ),
                (
                    width_third + random.randint(-10, 10),
                    height_third + random.randint(-10, 10),
                    params['u_shape_width'],
                    params['middle_length'],
                    random.choice([0, 2])
                ),
                (
                    self.width - params['margin'] - params['u_shape_width'] - random.randint(0, 10),
                    self.height - params['margin'] - params['middle_length'] - random.randint(0, 10),
                    params['u_shape_width'],
                    params['middle_length'],
                    random.choice([2, 3])
                )
            ]
            return configs

        def draw_u_shape(x, y, width, length, direction):
            t = params['wall_thickness']

            if direction == 0:
                cv2.rectangle(img_temp, (x, y + t), (x + t, y + length), (0, 0, 0), -1)
                cv2.rectangle(img_temp, (x + width - t, y + t), (x + width, y + length), (0, 0, 0), -1)
                cv2.rectangle(img_temp, (x, y + length - t), (x + width, y + length), (0, 0, 0), -1)
                return (x, y, width, length)

            elif direction == 1:
                cv2.rectangle(img_temp, (x, y), (x + length - t, y + t), (0, 0, 0), -1)
                cv2.rectangle(img_temp, (x, y + width - t), (x + length - t, y + width), (0, 0, 0), -1)
                cv2.rectangle(img_temp, (x, y), (x + t, y + width), (0, 0, 0), -1)
                return (x, y, length, width)

            elif direction == 2:
                cv2.rectangle(img_temp, (x, y), (x + t, y + length - t), (0, 0, 0), -1)
                cv2.rectangle(img_temp, (x + width - t, y), (x + width, y + length - t), (0, 0, 0), -1)
                cv2.rectangle(img_temp, (x, y), (x + width, y + t), (0, 0, 0), -1)
                return (x, y, width, length)

            else:
                cv2.rectangle(img_temp, (x + t, y), (x + length, y + t), (0, 0, 0), -1)
                cv2.rectangle(img_temp, (x + t, y + width - t), (x + length, y + width), (0, 0, 0), -1)
                cv2.rectangle(img_temp, (x + length - t, y), (x + length, y + width), (0, 0, 0), -1)
                return (x, y, length, width)

        def check_valid_position(x, y, width, height):
            return (x >= params['margin'] and
                    y >= params['margin'] and
                    x + width <= self.width - params['margin'] and
                    y + height <= self.height - params['margin'])

        def check_safe_distance(x, y, width, height, start, end):
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

        for attempt in range(3):
            obstacle_boxes = []
            img_temp = np.ones_like(img) * 255

            configs = generate_obstacle_positions()
            all_valid = True

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

            if all_valid and len(obstacle_boxes) == 3:
                if check_path_exists(start, end):
                    img[:] = img_temp
                    return True

        return self.generate_rectangle(img, density, start, end, retry_count + 1)

    def generate_circles(self, img, density):
        num_circles = int((self.width * self.height) * density / 2000)
        for _ in range(num_circles):
            radius = random.randint(5, 15)
            x = random.randint(radius, self.width - radius)
            y = random.randint(radius, self.height - radius)
            cv2.circle(img, (x, y), radius, (0, 0, 0), -1)

    def generate_map(self, obstacle_type, density, max_attempts=50):
        for attempt in range(max_attempts):
            img = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255

            while True:
                start = self.generate_point(True)
                end = self.generate_point(False)
                if self.check_distance(start, end):
                    break

            success = True
            if obstacle_type == "Circle":
                self.generate_circles(img, density)
            elif obstacle_type == "Rectangle":
                if not self.generate_rectangle(img, density, start, end):
                    success = False
            else:
                self.generate_circles(img, density / 2)
                if not self.generate_rectangle(img, density / 2, start, end):
                    success = False

            if not success:
                continue

            if not self.check_obstacles_exist(img, start, end):
                continue

            if not ((img[start[1], start[0]] == 255).all() and
                    (img[end[1], end[0]] == 255).all()):
                continue

            self.draw_point(img, start, (0, 255, 0))
            self.draw_point(img, end, (255, 0, 0))

            return img

        return self.generate_map(obstacle_type, density)

    def check_obstacles_exist(self, img, start, end):
        black_pixels = np.sum(img[:, :, 0] == 0)
        total_pixels = self.width * self.height

        min_obstacle_ratio = 0.05
        max_obstacle_ratio = 0.3

        obstacle_ratio = black_pixels / total_pixels

        if obstacle_ratio < min_obstacle_ratio or obstacle_ratio > max_obstacle_ratio:
            return False

        safe_radius = 10

        def check_point_safety(point):
            x, y = point
            for dx in range(-safe_radius, safe_radius + 1):
                for dy in range(-safe_radius, safe_radius + 1):
                    check_x = x + dx
                    check_y = y + dy
                    if (0 <= check_x < self.width and
                            0 <= check_y < self.height and
                            img[check_y, check_x, 0] == 0):
                        return False
            return True

        if not check_point_safety(start) or not check_point_safety(end):
            return False

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_contour_area = 100
        valid_obstacles = False

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_contour_area:
                mask = np.zeros_like(gray)
                cv2.drawContours(mask, [contour], -1, 255, -1)

                if mask[start[1], start[0]] == 0 and mask[end[1], end[0]] == 0:
                    valid_obstacles = True
                    break

        return valid_obstacles


class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Map Generator")
        self.setup_gui()
        self.map_generator = MapGenerator()

    def setup_gui(self):
        tk.Label(self.root, text="Number of maps:").grid(row=0, column=0)
        self.num_maps = tk.Entry(self.root)
        self.num_maps.insert(0, "1")
        self.num_maps.grid(row=0, column=1)

        tk.Label(self.root, text="Obstacle density (1-10):").grid(row=1, column=0)
        self.density = tk.Scale(self.root, from_=1, to=10, orient=tk.HORIZONTAL)
        self.density.grid(row=1, column=1)

        tk.Label(self.root, text="Obstacle type:").grid(row=2, column=0)
        self.obstacle_type = ttk.Combobox(self.root, values=["Circle", "Rectangle", "Mixed"])
        self.obstacle_type.set("Circle")
        self.obstacle_type.grid(row=2, column=1)

        tk.Button(self.root, text="Generate Maps", command=self.generate).grid(row=3, column=0, columnspan=2)

    def generate(self):
        try:
            num = int(self.num_maps.get())
            density = self.density.get()
            obs_type = self.obstacle_type.get()
            
            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            save_dir = os.path.join(desktop, "4.4")
            os.makedirs(save_dir, exist_ok=True)
            
            success_count = 0
            max_attempts = num * 3

            for i in range(max_attempts):
                if success_count >= num:
                    break

                img = self.map_generator.generate_map(obs_type, density)
                if img is not None:
                    cv2.imwrite(os.path.join(save_dir, f"map_{success_count + 1}.png"), img)
                    success_count += 1

            if success_count < num:
                messagebox.showwarning("Warning",
                                       f"Only {success_count} valid maps could be generated, less than the requested {num}")
            else:
                messagebox.showinfo("Success",
                                    f"Generated {success_count} maps and saved to Desktop/Maps folder")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    gui = GUI()
    gui.run()
