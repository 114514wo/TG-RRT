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

        params = {
            'rect_width': 25,  
            'rect_gap': 30,  
            'small_rect_size': (10, 15),  
            'margin': 20,  
            'safe_radius': 25,  
        }

        def generate_main_rectangles():
            rectangles = []
  
            x1 = int(self.width * 0.25)
            gap_y1 = random.randint(int(self.height * 0.3),
                                    int(self.height * 0.7))  

            x2 = int(self.width * 0.625)
            gap_y2 = random.randint(int(self.height * 0.3),
                                    int(self.height * 0.7))  

            rectangles.append({
                'x': x1,
                'y': 0,  
                'width': params['rect_width'],
                'height': gap_y1  
            })
            rectangles.append({
                'x': x1,
                'y': gap_y1 + params['rect_gap'],
                'width': params['rect_width'],
                'height': self.height - (gap_y1 + params['rect_gap'])  
            })

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

            return rectangles, [(x1, gap_y1), (x2, gap_y2)]  

        def generate_scattered_rectangles(main_rects_pos):
            scattered = []
            num_scattered = random.randint(5, 12)

            for x_center, gap_y in main_rects_pos:
                for _ in range(num_scattered // 2):
                    width = random.randint(8, 15)
                    height = random.randint(8, 15)

                    side = random.choice([-1, 1])  
                    x_offset = random.randint(30, 50) * side

                    y_offset = random.randint(-40, 40)

                    x = x_center + x_offset
                    y = gap_y + y_offset

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

        for attempt in range(10):
            img_temp = np.ones_like(img) * 255

            main_rectangles, gap_positions = generate_main_rectangles()

            scattered_rectangles = generate_scattered_rectangles(gap_positions)

            for rect in main_rectangles + scattered_rectangles:
                cv2.rectangle(img_temp,
                              (rect['x'], rect['y']),
                              (rect['x'] + rect['width'],
                               rect['y'] + rect['height']),
                              (0, 0, 0), -1)

            if start and end:
                if not ((img_temp[start[1], start[0]] == 255).all() and
                        (img_temp[end[1], end[0]] == 255).all()):
                    continue

            if start and end and not check_path_exists(start, end):
                continue

            img[:] = img_temp
            return True

        return False

    def generate_circles(self, img, density):
        params = {
            'min_radius': 15,  
            'max_radius': 30,  
            'min_distance': 20, 
            'safe_radius': 25,  
        }

        def check_circle_overlap(x, y, r, circles):
            for cx, cy, cr in circles:
                if sqrt((x - cx) ** 2 + (y - cy) ** 2) < (r + cr + params['min_distance']):
                    return True
            return False

        def check_circle_valid(x, y, r, start=None, end=None):
            if not (r <= x <= self.width - r and r <= y <= self.height - r):
                return False

            if start and end:
                dist_start = sqrt((x - start[0]) ** 2 + (y - start[1]) ** 2)
                dist_end = sqrt((x - end[0]) ** 2 + (y - end[1]) ** 2)
                if (dist_start < r + params['safe_radius'] or
                        dist_end < r + params['safe_radius']):
                    return False

            return True

        num_circles = random.randint(5, 8)
        circles = []  

        max_attempts = 100  
        for _ in range(num_circles):
            attempts = 0
            while attempts < max_attempts:
                radius = random.randint(params['min_radius'], params['max_radius'])

                x = random.randint(radius + 10, self.width - radius - 10)
                y = random.randint(radius + 10, self.height - radius - 10)

                if not check_circle_valid(x, y, radius):
                    attempts += 1
                    continue

                if check_circle_overlap(x, y, radius, circles):
                    attempts += 1
                    continue

                circles.append((x, y, radius))
    
                cv2.circle(img, (x, y), radius, (0, 0, 0), -1)
                break

            if attempts >= max_attempts:
                print(f"Warning: Could not place circle {len(circles) + 1}")

        return len(circles) >= 5  

    def generate_map(self, obstacle_type, density):
        max_attempts = 10  

        for attempt in range(max_attempts):
            img = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255

            while True:
                start = self.generate_point(True)
                end = self.generate_point(False)
                if self.check_distance(start, end):
                    break

            safety_radius = 10
            cv2.circle(img, start, safety_radius, (255, 255, 255), -1)
            cv2.circle(img, end, safety_radius, (255, 255, 255), -1)

            success = True
            if obstacle_type == "Circle":
                if not self.generate_circles(img, density):
                    success = False
            elif obstacle_type == "Rectangle":
                if not self.generate_rectangle(img, density, start, end):
                    success = False
            else:  
                if not self.generate_rectangle(img, density / 2, start, end):
                    success = False

                if not self.generate_circles(img, density / 2):
                    success = False

            if not success:
                continue

            start_area = img[start[1] - safety_radius:start[1] + safety_radius + 1,
                         start[0] - safety_radius:start[0] + safety_radius + 1]
            end_area = img[end[1] - safety_radius:end[1] + safety_radius + 1,
                       end[0] - safety_radius:end[0] + safety_radius + 1]

            if (np.any(start_area != 255) or np.any(end_area != 255)):
                continue

            if not self.check_path_exists(img, start, end):
                continue

            self.draw_point(img, start, (0, 255, 0))  # Green start point
            self.draw_point(img, end, (255, 0, 0))  # Blue end point

            return img

        return self.generate_map(obstacle_type, density)

    def check_path_exists(self, img, start, end):
        visited = np.zeros((self.height, self.width), dtype=bool)

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                      (1, 1), (1, -1), (-1, 1), (-1, -1)]

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
        black_pixels = np.sum(img[:, :, 0] == 0)
        total_pixels = self.width * self.height

        min_obstacle_ratio = 0.05 
        max_obstacle_ratio = 0.4 

        obstacle_ratio = black_pixels / total_pixels

        if obstacle_ratio < min_obstacle_ratio:
            return False
        if obstacle_ratio > max_obstacle_ratio:
            return False

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_contour_area = 100  
        large_obstacles = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_contour_area:
                large_obstacles += 1

        return large_obstacles >= 2


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

        tk.Button(self.root, text="Generate Map", command=self.generate).grid(row=3, column=0, columnspan=2)

    def generate(self):
        try:
            num = int(self.num_maps.get())
            density = self.density.get()
            obs_type = self.obstacle_type.get()

            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            save_dir = os.path.join(desktop, "Map")
            os.makedirs(save_dir, exist_ok=True)

            success_count = 0
            max_attempts = num * 3 

            for i in range(max_attempts):
                if success_count >= num:
                    break

                img = self.map_generator.generate_map(obs_type, density)
                if img is not None:
                    if self.map_generator.check_obstacles_exist(img):
                        cv2.imwrite(os.path.join(save_dir, f"map_{success_count + 1}.png"), img)
                        success_count += 1

            if success_count < num:
                messagebox.showwarning("Warning",
                                     f"Only generated {success_count} valid maps, less than the requested {num}")
            else:
                messagebox.showinfo("Success",
                                  f"Generated {success_count} maps and saved to the Map folder on Desktop")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    gui = GUI()
    gui.run()
