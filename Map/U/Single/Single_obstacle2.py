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
        """Generate start or end point"""
        size = 5
        if is_start:
            x = random.randint(size, self.width // 2 - size)
        else:
            x = random.randint(self.width // 2 + size, self.width - size)
        y = random.randint(size, self.height - size)
        return (x, y)

    def check_distance(self, p1, p2):
        """Check distance between two points"""
        return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) >= 150

    def draw_point(self, img, center, color):
        """Draw a 5x5 start or end point"""
        x, y = center
        cv2.rectangle(img, (x - 2, y - 2), (x + 2, y + 2), color, -1)

    def generate_rectangle(self, img, density, start=None, end=None):
        """Generate U-shaped obstacle at random position and direction, with increased connection length"""
        # Clear image to white
        img_temp = np.ones_like(img) * 255

        if start is None or end is None:
            return False

        def is_safe_distance(x, y, width, height, start, end, safe_radius=20):
            """Check if obstacle maintains safe distance from start/end points"""
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
            """Draw U-shaped obstacle, width for end sections (longer), middle_length for middle section"""
            if direction == 0:  # Open at top
                # Draw left vertical line
                cv2.rectangle(img, (x, y + thickness),
                              (x + thickness, y + middle_length),
                              (0, 0, 0), -1)
                # Draw right vertical line
                cv2.rectangle(img, (x + width - thickness, y + thickness),
                              (x + width, y + middle_length),
                              (0, 0, 0), -1)
                # Draw bottom horizontal line
                cv2.rectangle(img, (x, y + middle_length - thickness),
                              (x + width, y + middle_length),
                              (0, 0, 0), -1)
                return (x, y, width, middle_length)

            elif direction == 1:  # Open at right
                # Draw top horizontal line
                cv2.rectangle(img, (x, y),
                              (x + middle_length - thickness, y + thickness),
                              (0, 0, 0), -1)
                # Draw bottom horizontal line
                cv2.rectangle(img, (x, y + width - thickness),
                              (x + middle_length - thickness, y + width),
                              (0, 0, 0), -1)
                # Draw left vertical line
                cv2.rectangle(img, (x, y),
                              (x + thickness, y + width),
                              (0, 0, 0), -1)
                return (x, y, middle_length, width)

            elif direction == 2:  # Open at bottom
                # Draw left vertical line
                cv2.rectangle(img, (x, y),
                              (x + thickness, y + middle_length - thickness),
                              (0, 0, 0), -1)
                # Draw right vertical line
                cv2.rectangle(img, (x + width - thickness, y),
                              (x + width, y + middle_length - thickness),
                              (0, 0, 0), -1)
                # Draw top horizontal line
                cv2.rectangle(img, (x, y),
                              (x + width, y + thickness),
                              (0, 0, 0), -1)
                return (x, y, width, middle_length)

            else:  # Open at left
                # Draw top horizontal line
                cv2.rectangle(img, (x + thickness, y),
                              (x + middle_length, y + thickness),
                              (0, 0, 0), -1)
                # Draw bottom horizontal line
                cv2.rectangle(img, (x + thickness, y + width - thickness),
                              (x + middle_length, y + width),
                              (0, 0, 0), -1)
                # Draw right vertical line
                cv2.rectangle(img, (x + middle_length - thickness, y),
                              (x + middle_length, y + width),
                              (0, 0, 0), -1)
                return (x, y, middle_length, width)

        def check_path_exists(img, start_point, end_point):
            """Check if a traversable path exists"""
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

        # Adjust obstacle parameters
        wall_thickness = 20  # Wall thickness
        u_shape_width = 160  # U-shape end sections length (further increased)
        middle_length = 100  # Middle section length
        margin = 30  # Boundary margin
        safe_radius = 25  # Safe distance from start/end points

        max_attempts = 50
        for attempt in range(max_attempts):
            img_temp = np.ones_like(img) * 255

            direction = random.randint(0, 3)

            if direction in [0, 2]:  # Open at top or bottom
                max_x = img.shape[1] - u_shape_width - margin
                max_y = img.shape[0] - middle_length - margin
            else:  # Open at left or right
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

        # If all attempts fail, generate a safe default U-shape
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
        """Generate circular obstacles"""
        # Reduce density value
        num_circles = int((self.width * self.height) * density / 2000)
        for _ in range(num_circles):
            radius = random.randint(5, 15)
            x = random.randint(radius, self.width - radius)
            y = random.randint(radius, self.height - radius)
            cv2.circle(img, (x, y), radius, (0, 0, 0), -1)

    def generate_map(self, obstacle_type, density):
        """Generate a single map"""
        img = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255

        # First generate start and end points
        while True:
            start = self.generate_point(True)
            end = self.generate_point(False)
            if self.check_distance(start, end):
                break

        # Generate obstacles
        if obstacle_type == "Circle":
            self.generate_circles(img, density)
            # Check if start/end points overlap with obstacles
            if not ((img[start[1], start[0]] == 255).all() and
                    (img[end[1], end[0]] == 255).all()):
                return self.generate_map(obstacle_type, density)
        elif obstacle_type == "Rectangle":
            self.generate_rectangle(img, density, start, end)
        else:  # Mixed
            self.generate_circles(img, density / 2)
            self.generate_rectangle(img, density / 2, start, end)
            # Check if start/end points overlap with obstacles
            if not ((img[start[1], start[0]] == 255).all() and
                    (img[end[1], end[0]] == 255).all()):
                return self.generate_map(obstacle_type, density)

        # Draw start and end points
        self.draw_point(img, start, (0, 255, 0))  # Green start point
        self.draw_point(img, end, (255, 0, 0))  # Blue end point

        return img


class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Map Generator")
        self.setup_gui()
        self.map_generator = MapGenerator()

    def setup_gui(self):
        # Quantity selection
        tk.Label(self.root, text="Number of maps:").grid(row=0, column=0)
        self.num_maps = tk.Entry(self.root)
        self.num_maps.insert(0, "1")
        self.num_maps.grid(row=0, column=1)

        # Density selection
        tk.Label(self.root, text="Obstacle density (1-10):").grid(row=1, column=0)
        self.density = tk.Scale(self.root, from_=1, to=10, orient=tk.HORIZONTAL)
        self.density.grid(row=1, column=1)

        # Obstacle type selection
        tk.Label(self.root, text="Obstacle type:").grid(row=2, column=0)
        self.obstacle_type = ttk.Combobox(self.root, values=["Circle", "Rectangle", "Mixed"])
        self.obstacle_type.set("Circle")
        self.obstacle_type.grid(row=2, column=1)

        # Generate button
        tk.Button(self.root, text="Generate Maps", command=self.generate).grid(row=3, column=0, columnspan=2)

    def generate(self):
        try:
            num = int(self.num_maps.get())
            density = self.density.get()
            obs_type = self.obstacle_type.get()

            # Create save directory
            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            save_dir = os.path.join(desktop, "5")
            os.makedirs(save_dir, exist_ok=True)

            # Generate and save maps
            for i in range(num):
                img = self.map_generator.generate_map(obs_type, density)
                cv2.imwrite(os.path.join(save_dir, f"map_{i + 1}.png"), img)

            messagebox.showinfo("Success", f"Generated {num} maps and saved to Desktop/Map folder")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    gui = GUI()
    gui.run()
