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
        if retry_count >= MAX_RETRIES or start is None or end is None:
            return False

        img_temp = np.ones_like(img) * 255

        params = {
            'rect_width': 20,  
            'rect_length': 120, 
            'square_size': 25,  
            'margin': 25,  
            'safe_radius': 25, 
            'min_distance': 30 
        }

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

        def generate_rectangles():
       
            width_third = self.width // 3
            height_third = self.height // 3

            rect1 = {
                'x': params['margin'],
                'y': params['margin'],
                'width': params['rect_length'],
                'height': params['rect_width']
            }

     
            rect2 = {
                'x': width_third + random.randint(-10, 10),
                'y': height_third + random.randint(-10, 10),
                'width': params['rect_length'],
                'height': params['rect_width']
            }

            rect3 = {
                'x': self.width - params['margin'] - params['rect_length'],
                'y': self.height - params['margin'] - params['rect_width'] - random.randint(-10, 10),
                'width': params['rect_length'],
                'height': params['rect_width']
            }

            return [rect1, rect2, rect3]

        def generate_scattered_squares():
 
            squares = []
            num_clusters = random.randint(2, 3)

  
            regions = [
                (self.width // 4, self.height * 2 // 3), 
                (self.width * 3 // 4, self.height // 3),  
                (self.width // 2, self.height * 3 // 4)  
            ]

            selected_regions = random.sample(regions, num_clusters)

            for center_x, center_y in selected_regions:
      
                num_squares = random.randint(3, 4) 

                patterns = [
                    [(0, 0), (1, 0), (0, 1)], 
                    [(0, 0), (1, 0), (1, 1)],  
                    [(0, 0), (1, 0), (2, 0)],  
                    [(0, 0), (0, 1), (0, 2)]  
                ]

           
                base_pattern = random.choice(patterns)

   
                for dx, dy in base_pattern:
                    square = {
                        'x': center_x + dx * (params['square_size'] + 5) + random.randint(-3, 3),
                        'y': center_y + dy * (params['square_size'] + 5) + random.randint(-3, 3),
                        'size': params['square_size']
                    }
                    squares.append(square)

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

            for rect in rectangles:
                cv2.rectangle(img_temp,
                              (rect['x'], rect['y']),
                              (rect['x'] + rect['width'], rect['y'] + rect['height']),
                              (0, 0, 0), -1)


            for square in squares:
                cv2.rectangle(img_temp,
                              (square['x'], square['y']),
                              (square['x'] + square['size'], square['y'] + square['size']),
                              (0, 0, 0), -1)

        def check_valid_position(obstacles):
     
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

        for attempt in range(3):
            img_temp = np.ones_like(img) * 255

            rectangles = generate_rectangles()

            squares = generate_scattered_squares()


            all_obstacles = rectangles + squares

            if not check_valid_position(all_obstacles):
                continue


            if not check_safe_distance(all_obstacles):
                continue


            draw_obstacles(rectangles, squares)

  
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
            if obstacle_type == "Circular":
                self.generate_circles(img, density)
            elif obstacle_type == "Rectangular":
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
        self.obstacle_type = ttk.Combobox(self.root, values=["Circular", "Rectangular", "Mixed"])
        self.obstacle_type.set("Circular")
        self.obstacle_type.grid(row=2, column=1)

        tk.Button(self.root, text="Generate Maps", command=self.generate).grid(row=3, column=0, columnspan=2)

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
                    cv2.imwrite(os.path.join(save_dir, f"map_{success_count + 1}.png"), img)
                    success_count += 1

            if success_count < num:
                messagebox.showwarning("Warning",
                                       f"Only {success_count} valid maps could be generated, fewer than the requested {num}")
            else:
                messagebox.showinfo("Success",
                                    f"Successfully generated {success_count} maps and saved to the Desktop Map folder")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    gui = GUI()
    gui.run()
