import cv2
import numpy as np
import os
import random
from math import sqrt, inf, pi, cos, sin
from math import sqrt, inf, pi, cos, sin, log  
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool

class RRTStarPathPlanner:
    def __init__(self):
        self.desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        self.map_dir = os.path.join(self.desktop, "4")
        self.enlarged_dir = os.path.join(self.desktop, "d")
        os.makedirs(self.enlarged_dir, exist_ok=True)
        
        self.step_size = 10  
        self.max_iterations = 5000  
        self.search_radius = 30  
        self.goal_sample_rate = 0.3  
        self.safety_distance = 3  
        self.min_distance_to_goal = 15  
        
        self.num_processes = os.cpu_count() or 4  
        self.min_paths = 15  
        self.max_paths = 35  
        
        self.collision_cache = {}
        self.distance_cache = {}

    def find_colored_point(self, img, color):
        if color == 'green':
            lower_bgr = np.array([0, 240, 0])
            upper_bgr = np.array([50, 255, 50])
            
            lower_hsv = np.array([35, 100, 100])
            upper_hsv = np.array([85, 255, 255])
        else:  
            lower_bgr = np.array([240, 0, 0])
            upper_bgr = np.array([255, 50, 50])
            
            lower_hsv = np.array([100, 100, 100])
            upper_hsv = np.array([130, 255, 255])
        
        mask_bgr = cv2.inRange(img, lower_bgr, upper_bgr)
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
        
        mask = cv2.bitwise_or(mask_bgr, mask_hsv)
        
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.erode(mask, kernel, iterations=1)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)

            if M["m00"] != 0:
                x = int(M["m10"] / M["m00"])
                y = int(M["m01"] / M["m00"])
                return (x, y)
        
        points = cv2.findNonZero(mask)
        if points is not None and len(points) > 0:
            x = int(np.mean(points[:, 0, 0]))
            y = int(np.mean(points[:, 0, 1]))
            return (x, y)
        
        print(f"Cannot find {color} point")
        cv2.imwrite(f"debug_{color}_mask.png", mask)
        cv2.imwrite(f"debug_{color}_original.png", img)
        return None

    def distance(self, p1, p2):
        return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def is_collision_free(self, p1, p2, img):
        if not (0 <= p1[0] < img.shape[1] and 0 <= p1[1] < img.shape[0] and
                0 <= p2[0] < img.shape[1] and 0 <= p2[1] < img.shape[0]):
            return False

        x1, y1 = p1
        x2, y2 = p2
        
        if abs(x1 - x2) <= 1 and abs(y1 - y2) <= 1:
            p1_free = int(img[y1, x1]) >= 200
            p2_free = int(img[y2, x2]) >= 200
            return p1_free and p2_free
        
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        sx = 1 if x2 > x1 else -1
        sy = 1 if y2 > y1 else -1

        if dx > dy:
            err = dx / 2.0
            while x != x2:
                points.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y2:
                points.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        points.append((x2, y2))
        
        safety_distance = self.safety_distance
        for x, y in points:
            x_min = max(0, x - safety_distance)
            x_max = min(img.shape[1], x + safety_distance + 1)
            y_min = max(0, y - safety_distance)
            y_max = min(img.shape[0], y + safety_distance + 1)
            
            region = img[y_min:y_max, x_min:x_max]
            if np.any(region < 200):  
                return False
        return True

    def get_line_points(self, x1, y1, x2, y2):
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        sx = 1 if x2 > x1 else -1
        sy = 1 if y2 > y1 else -1

        if dx > dy:
            err = dx / 2.0
            while x != x2:
                points.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y2:
                points.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy

        points.append((x2, y2))
        return points

    def find_nearest(self, nodes, point):
        distances = [self.distance(point, n) for n in nodes]
        return nodes[np.argmin(distances)]

    def steer(self, from_point, to_point, img):
        dist = self.distance(from_point, to_point)
        if dist <= self.step_size:
            return to_point

        theta = np.arctan2(to_point[1] - from_point[1],
                           to_point[0] - from_point[0])
       
        new_x = int(round(from_point[0] + self.step_size * np.cos(theta)))
        new_y = int(round(from_point[1] + self.step_size * np.sin(theta)))
        
        new_x = max(0, min(new_x, img.shape[1] - 1))
        new_y = max(0, min(new_y, img.shape[0] - 1))
        return (new_x, new_y)

    def find_neighbors(self, nodes, point):
        neighbors = []
        max_neighbors = 50  
        for node in nodes:
            if len(neighbors) >= max_neighbors:
                break
            if self.distance(node, point) <= self.search_radius:
                neighbors.append(node)
        return neighbors

    def rrt_star(self, img, start, goal):
        nodes = [start]
        parent = {start: None}
        cost = {start: 0}
        valid_nodes = {start}
        best_cost = float('inf')
        best_path = None
      
        dist_to_goal = self.distance(start, goal)
        max_nodes = min(3000, int(dist_to_goal * 5)) 
        search_radius = min(self.search_radius, dist_to_goal * 0.3)

        def is_valid_node(node):
            return (node in valid_nodes and
                    node in cost and
                    0 <= node[0] < img.shape[1] and
                    0 <= node[1] < img.shape[0])

        for i in range(self.max_iterations):
            if len(nodes) >= max_nodes:
                break
          
            if random.random() < self.goal_sample_rate:
                random_point = goal
            else:
                margin = 50
                x_min = max(0, min(start[0], goal[0]) - margin)
                x_max = min(img.shape[1] - 1, max(start[0], goal[0]) + margin)
                y_min = max(0, min(start[1], goal[1]) - margin)
                y_max = min(img.shape[0] - 1, max(start[1], goal[1]) + margin)
                random_point = (
                    random.randint(x_min, x_max),
                    random.randint(y_min, y_max)
                )
         
            valid_distances = [(n, self.distance(random_point, n))
                               for n in valid_nodes]
            if not valid_distances:
                continue

            nearest_node = min(valid_distances, key=lambda x: x[1])[0]
            new_node = self.steer(nearest_node, random_point, img)
           
            if (not (0 <= new_node[0] < img.shape[1] and
                     0 <= new_node[1] < img.shape[0]) or
                    new_node in valid_nodes or
                    not self.is_collision_free(nearest_node, new_node, img)):
                continue
           
            neighbors = []
            for node in valid_nodes:
                if len(neighbors) >= 30:  
                    break
                if self.distance(node, new_node) <= search_radius:
                    neighbors.append(node)
            
            min_cost = float('inf')
            best_parent = None

            for neighbor in neighbors:
                if not is_valid_node(neighbor):
                    continue
                potential_cost = cost[neighbor] + self.distance(neighbor, new_node)
                if (potential_cost < min_cost and
                        self.is_collision_free(neighbor, new_node, img)):
                    min_cost = potential_cost
                    best_parent = neighbor

            if best_parent is None:
                if is_valid_node(nearest_node):
                    best_parent = nearest_node
                    min_cost = cost[nearest_node] + self.distance(nearest_node, new_node)
                else:
                    continue
            
            nodes.append(new_node)
            valid_nodes.add(new_node)
            parent[new_node] = best_parent
            cost[new_node] = min_cost
           
            for neighbor in neighbors:
                if not is_valid_node(neighbor) or neighbor == best_parent:
                    continue
                potential_cost = cost[new_node] + self.distance(new_node, neighbor)
                if (potential_cost < cost[neighbor] and
                        self.is_collision_free(new_node, neighbor, img)):
                    parent[neighbor] = new_node
                    cost[neighbor] = potential_cost
           
            if self.distance(new_node, goal) < self.min_distance_to_goal:
                if self.is_collision_free(new_node, goal, img):
                    final_node = self.steer(new_node, goal, img)
                    path_nodes = []
                    current = new_node
                    path_length = 0
                    
                    while current is not None and path_length < 1000:
                        if not is_valid_node(current) and current != goal:
                            path_nodes = []
                            break
                        path_nodes.append(current)
                        current = parent.get(current)
                        path_length += 1

                    if path_nodes and path_length < 1000:
                        path_nodes.reverse()
                        if path_nodes[-1] != goal:
                            path_nodes.append(goal)
                        current_cost = sum(self.distance(path_nodes[j], path_nodes[j + 1])
                                           for j in range(len(path_nodes) - 1))
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_path = path_nodes.copy()
                            search_radius = min(self.search_radius, current_cost * 0.2)
            
            if i % 100 == 0 and best_path is not None:
                keep_nodes = set(best_path)
                for node in best_path:
                    for n in list(valid_nodes)[-200:]:
                        if self.distance(node, n) <= search_radius:
                            keep_nodes.add(n)
                
                valid_nodes = {n for n in keep_nodes if is_valid_node(n)}
                nodes = list(valid_nodes)
                parent = {k: v for k, v in parent.items() if k in valid_nodes}
                cost = {k: v for k, v in cost.items() if k in valid_nodes}

        return best_path

    def extract_path(self, parent, node):
        path = []
        current = node
        max_path_length = 1000  
        while current is not None and len(path) < max_path_length:
            path.append(current)
            current = parent.get(current)
        if len(path) >= max_path_length:
            print("Warning: Path length exceeds limit")
            return None
        return path[::-1]

    def optimize_path(self, path, img):
        if len(path) < 3:
            return path
        optimized_path = [path[0]]
        current_point = 0
        while current_point < len(path) - 1:
            for i in range(len(path) - 1, current_point, -1):
                if self.is_collision_free(path[current_point], path[i], img):
                    optimized_path.append(path[i])
                    current_point = i
                    break
            else:
                current_point += 1
                if current_point < len(path):
                    optimized_path.append(path[current_point])
        
        smoothed_path = self.smooth_path(optimized_path, img)
        return smoothed_path

    def process_map(self, map_file):
        try:
            img_path = os.path.join(self.map_dir, map_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Cannot read map: {map_file}")
                return
           
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced = cv2.merge((cl, a, b))
            img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            start = self.find_colored_point(img, 'green')
            goal = self.find_colored_point(img, 'blue')

            if start is None or goal is None:
                print(f"Cannot find start or goal point: {map_file}")
                return

            print(f"Found start: {start}, goal: {goal}")
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
           
            kernel = np.ones((5, 5), np.uint8)
            binary = cv2.erode(binary, kernel, iterations=1)
            binary = cv2.dilate(binary, kernel, iterations=1)
            
            cv2.circle(binary, start, 5, 255, -1)
            cv2.circle(binary, goal, 5, 255, -1)
          
            PATH_COLOR = (0, 255, 0) 
            
            all_paths = []
            path_costs = []
            
            MIN_PATHS = 15  
            MAX_PATHS = 35 
            MAX_SEARCH_PATHS = 60  
            num_attempts = 150  
            
            display_img = img.copy()
           
            dist_to_goal = self.distance(start, goal)
            max_nodes = min(3000, int(dist_to_goal * 5)) 
            search_radius = min(self.search_radius, dist_to_goal * 0.3)

            def is_valid_node(node):
                return (node in valid_nodes and
                        node in cost and
                        0 <= node[0] < img.shape[1] and
                        0 <= node[1] < img.shape[0])

            def quick_feasibility_check():
                if binary[start[1], start[0]] < 200 or binary[goal[1], goal[0]] < 200:
                    return False, "The start or goal point is blocked"
         
                radius = 20
                start_region = binary[
                    max(0, start[1] - radius):min(binary.shape[0], start[1] + radius),
                    max(0, start[0] - radius):min(binary.shape[1], start[0] + radius)
                ]
                goal_region = binary[
                    max(0, goal[1] - radius):min(binary.shape[0], goal[1] + radius),
                    max(0, goal[0] - radius):min(binary.shape[1], goal[0] + radius)
                ]

                if np.mean(start_region) < 180 or np.mean(goal_region) < 180:
                    return False, "The start or goal point is blocked"

                return True, "Initial check passed"

            feasible, reason = quick_feasibility_check()
            if not feasible:
                print(f"Map {map_file} quick check failed: {reason}")
                return False 
          
            num_attempts = 20
            args_list = [(start, goal, binary, i) for i in range(num_attempts)]

            all_paths = []
            path_costs = []
            failed_attempts = 0
            max_failed_attempts = 3  
            
            with Pool(processes=self.num_processes) as pool:
                try:
                    pending_tasks = []
                    
                    for i in range(min(self.num_processes, num_attempts)):
                        task = pool.apply_async(self.find_path_attempt, (args_list[i],))
                        pending_tasks.append((i, task))

                    next_task_idx = self.num_processes

                    while pending_tasks:
                        for task_idx, task in pending_tasks[:]:
                            if task.ready():
                                try:
                                    pending_tasks.remove((task_idx, task))
                                    path, cost = task.get(timeout=0.1)

                                    if path is not None:
                                        failed_attempts = 0 
                                        all_paths.append(path)
                                        path_costs.append(cost)
                                        print(f"Found path {len(all_paths)}")

                                        if len(all_paths) >= self.min_paths:
                                            good_paths = [p for p, c in zip(all_paths, path_costs)
                                                          if c <= min(path_costs) * 1.2]
                                            if len(good_paths) >= self.min_paths:
                                                return self.save_successful_paths(img, start, goal, all_paths,
                                                                                  path_costs, map_file)
                                    else:
                                        failed_attempts += 1
                                        print(f"Consecutive failure #{failed_attempts}")
                                        if failed_attempts >= max_failed_attempts:
                                            print(f"Failed {max_failed_attempts} times in a row, skipping this map")
                                            return False  
                                except Exception as e:
                                    print(f"Error processing task {task_idx}: {str(e)}")
                                    failed_attempts += 1
                                
                                if next_task_idx < len(args_list):
                                    new_task = pool.apply_async(self.find_path_attempt, (args_list[next_task_idx],))
                                    pending_tasks.append((next_task_idx, new_task))
                                    next_task_idx += 1
                    time.sleep(0.05)
                except Exception as e:
                    print(f"Process pool execution error: {str(e)}")
                    return False
                finally:
                    pool.terminate()
                    pool.join()
           
            if not all_paths:
                print(f"Could not generate any valid paths for map {map_file}")
                return False  
            return False
        except Exception as e:
            print(f"Error processing map {map_file}: {str(e)}")
            return False

    def paths_are_similar(self, path1, path2, threshold=10):
        if abs(len(path1) - len(path2)) > len(path1) * 0.3:  
            return False
       
        sample_points = min(10, min(len(path1), len(path2)))
        indices = np.linspace(0, min(len(path1), len(path2)) - 1, sample_points, dtype=int)

        for i in indices:
            if self.distance(path1[i], path2[i]) > threshold:
                return False
        return True

    def process_all_maps(self):
        if not os.path.exists(self.map_dir):
            print("Map folder does not exist")
            return
    
        existing_files = [f for f in os.listdir(self.enlarged_dir)
                          if f.startswith('path_map_') and f.endswith(('.png', '.jpg', '.jpeg'))]

        max_existing_number = 0
        for file in existing_files:
            try:
                number = int(''.join(filter(str.isdigit, file)))
                max_existing_number = max(max_existing_number, number)
            except ValueError:
                continue

        print(f"Maximum image number in target folder: {max_existing_number}")
        
        map_files = []
        for map_file in os.listdir(self.map_dir):
            if map_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    number = int(''.join(filter(str.isdigit, map_file)))
                    if number > max_existing_number:  
                        map_files.append((number, map_file))
                except ValueError:
                    continue
        
        map_files.sort(key=lambda x: x[0])

        if not map_files:
            print(f"No new maps to process (with number greater than {max_existing_number})")
            return

        print(f"Will process maps from {map_files[0][0]} to {map_files[-1][0]}")
        
        for _, map_file in map_files:
            print(f"\nStarting to process map: {map_file}")
            self.process_map(map_file)

    def smooth_path(self, path, img):
        if len(path) < 3:
            return path

        smoothed = path.copy()
        change = True
        while change:
            change = False
            for i in range(1, len(smoothed) - 1):
                original = smoothed[i]
                for t in np.linspace(0.2, 0.8, 7):
                    new_x = int(smoothed[i - 1][0] * (1 - t) + smoothed[i + 1][0] * t)
                    new_y = int(smoothed[i - 1][1] * (1 - t) + smoothed[i + 1][1] * t)
                    new_point = (new_x, new_y)

                    if (self.is_collision_free(smoothed[i - 1], new_point, img) and
                            self.is_collision_free(new_point, smoothed[i + 1], img)):
                        if new_point != original:
                            smoothed[i] = new_point
                            change = True
                            break
        return smoothed

    def update_children_cost(self, node, cost, parent, nodes):
        children = [n for n in nodes if parent.get(n) == node]
        for child in children:
            cost[child] = cost[node] + self.distance(node, child)
            self.update_children_cost(child, cost, parent, nodes)

    def path_cost(self, path):
        if not path:
            return float('inf')
        cost = 0
        for i in range(len(path) - 1):
            cost += self.distance(path[i], path[i + 1])
        return cost

    def draw_tree(self, node):
        if node.parent:
            plt.plot([node.x, node.parent.x],
                     [node.y, node.parent.y],
                     '-g', linewidth=1)
            plt.plot(node.x, node.y, 'go', markersize=3)

    def visualize(self):
        plt.clf()  
        plt.plot(self.start.x, self.start.y, 'bo', label='Start')
        plt.plot(self.goal.x, self.goal.y, 'ro', label='Goal')
        self.plot_obstacles()
        plt.grid(True)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('RRT Path Planning')
        plt.axis([self.min_rand, self.max_rand, self.min_rand, self.max_rand])

    def plan(self):
        for i in range(self.max_iter):
            self.visualize()
            rnd = self.get_random_point()
            nearest_node = self.get_nearest_node(self.node_list, rnd)
            new_node = self.steer(nearest_node, rnd, img)
           
            if not self.check_collision(new_node):
                self.node_list.append(new_node)
                self.draw_tree(new_node)
                
                if self.is_near_goal(new_node):
                    final_node = self.steer(new_node, self.goal, img)
                    if not self.check_collision(final_node):
                        self.generate_final_course(final_node)
                        break
            plt.pause(0.01)  

    def find_path_attempt(self, args):
        start, goal, binary, attempt_number = args
        try:
            random.seed(os.getpid() + attempt_number)
            nodes = [start]
            parent = {start: None}
            cost = {start: 0}
            valid_nodes = {start}
            best_cost = float('inf')
            best_path = None
           
            dist_to_goal = self.distance(start, goal)
            max_nodes = min(2000, int(dist_to_goal * 4))
            search_radius = min(self.search_radius, dist_to_goal * 0.25)

            def is_valid_node(node):
                return (node in valid_nodes and
                        node in cost and
                        0 <= node[0] < binary.shape[1] and
                        0 <= node[1] < binary.shape[0])

            for i in range(self.max_iterations):
                if len(nodes) >= max_nodes:
                    break

                if random.random() < self.goal_sample_rate:
                    random_point = goal
                else:
                    margin = 50
                    x_min = max(0, min(start[0], goal[0]) - margin)
                    x_max = min(binary.shape[1] - 1, max(start[0], goal[0]) + margin)
                    y_min = max(0, min(start[1], goal[1]) - margin)
                    y_max = min(binary.shape[0] - 1, max(start[1], goal[1]) + margin)

                    random_point = (
                        random.randint(x_min, x_max),
                        random.randint(y_min, y_max)
                    )
                
                nearest_node = min(valid_nodes, key=lambda n: self.distance(n, random_point))
                new_node = self.steer(nearest_node, random_point, binary)
                
                if (not (0 <= new_node[0] < binary.shape[1] and
                         0 <= new_node[1] < binary.shape[0]) or
                        new_node in valid_nodes or
                        not self.is_collision_free(nearest_node, new_node, binary)):
                    continue
               
                neighbors = [n for n in valid_nodes
                             if self.distance(n, new_node) <= search_radius][:30]
                
                min_cost = float('inf')
                best_parent = None

                for neighbor in neighbors:
                    if not is_valid_node(neighbor):
                        continue

                    potential_cost = cost[neighbor] + self.distance(neighbor, new_node)
                    if (potential_cost < min_cost and
                            self.is_collision_free(neighbor, new_node, binary)):
                        min_cost = potential_cost
                        best_parent = neighbor

                if best_parent is None:
                    if is_valid_node(nearest_node):
                        best_parent = nearest_node
                        min_cost = cost[nearest_node] + self.distance(nearest_node, new_node)
                    else:
                        continue
                
                nodes.append(new_node)
                valid_nodes.add(new_node)
                parent[new_node] = best_parent
                cost[new_node] = min_cost
                
                if self.distance(new_node, goal) < self.min_distance_to_goal:
                    if self.is_collision_free(new_node, goal, binary):
                        path = []
                        current = new_node
                        while current is not None:
                            path.append(current)
                            current = parent.get(current)
                        path.reverse()
                        path.append(goal)

                        path_cost = sum(self.distance(path[i], path[i + 1])
                                        for i in range(len(path) - 1))

                        if path_cost < best_cost:
                            best_cost = path_cost
                            best_path = path

            return best_path, best_cost if best_path else (None, float('inf'))
        except Exception as e:
            print(f"Attempt {attempt_number} failed: {str(e)}")
            return None, float('inf')

    def save_failed_image(self, img, start, goal, map_file, reason):
        try:
            gray_img = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.circle(gray_img, start, 5, 255, -1)
            cv2.circle(gray_img, goal, 5, 255, -1)
           
            output_filename = f"failed_{reason}_{map_file}"
            cv2.imwrite(os.path.join(self.enlarged_dir, output_filename), gray_img)
            print(f"Failed image saved: {output_filename}")
            return True
        except Exception as e:
            print(f"Error saving failed image: {str(e)}")
            return False

    def save_successful_paths(self, img, start, goal, all_paths, path_costs, map_file):
        try:
            path_pairs = list(zip(all_paths, path_costs))
            path_pairs.sort(key=lambda x: x[1])  # Sort by cost
            path_pairs = path_pairs[:self.max_paths]
            gray_img = np.zeros(img.shape[:2], dtype=np.uint8)
            
            for path, cost in path_pairs:
                path_array = np.array(path)
                for i in range(len(path_array) - 1):
                    cv2.line(gray_img,
                             tuple(path_array[i]),
                             tuple(path_array[i + 1]),
                             255,  
                             15)  
                print(f"Path cost: {cost:.2f}")
            
            cv2.circle(gray_img, start, 5, 255, -1)
            cv2.circle(gray_img, goal, 5, 255, -1)

            output_filename = f"path_map_{map_file}"
            cv2.imwrite(os.path.join(self.enlarged_dir, output_filename), gray_img)
            print(f"Successfully processed map: {map_file}")
            return True
        except Exception as e:
            print(f"Error saving successful path image: {str(e)}")
            return False

if __name__ == "__main__":
    planner = RRTStarPathPlanner()
    planner.process_all_maps()
