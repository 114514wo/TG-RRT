import cv2
import numpy as np
import os
import random
from math import sqrt, inf, pi, cos, sin
from math import sqrt, inf, pi, cos, sin, log  # 添加 log 导入
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool


class RRTStarPathPlanner:
    def __init__(self):
        # 设置文件夹路径
        self.desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        self.map_dir = os.path.join(self.desktop, "4")
        self.enlarged_dir = os.path.join(self.desktop, "d")

        # 创建输出文件夹（如果不存在）
        os.makedirs(self.enlarged_dir, exist_ok=True)

        # 调整RRT*参数
        self.step_size = 10  # 减小步长，使运动更精细
        self.max_iterations = 5000  # 增加迭代次数
        self.search_radius = 30  # 调整搜索半径
        self.goal_sample_rate = 0.3  # 增加目标采样率
        self.safety_distance = 3  # 稍微减小安全距离
        self.min_distance_to_goal = 15  # 调整到目标的最小距离

        # 添加多进程相关参数
        self.num_processes = os.cpu_count() or 4  # 获取CPU核心数，如果获取失败则使用4
        self.min_paths = 15  # 最少需要找到的路径数
        self.max_paths = 35  # 最多保留的路径数

        # 添加缓存
        self.collision_cache = {}
        self.distance_cache = {}

    def find_colored_point(self, img, color):
        """改进的颜点检测方法"""
        if color == 'green':
            # BGR空间中的绿色范围
            lower_bgr = np.array([0, 240, 0])
            upper_bgr = np.array([50, 255, 50])

            # HSV空间中的绿色范围
            lower_hsv = np.array([35, 100, 100])
            upper_hsv = np.array([85, 255, 255])
        else:  # blue
            # BGR空间中的蓝色范围
            lower_bgr = np.array([240, 0, 0])
            upper_bgr = np.array([255, 50, 50])

            # HSV空间中的蓝色范围
            lower_hsv = np.array([100, 100, 100])
            upper_hsv = np.array([130, 255, 255])

        # 在BGR空间查找
        mask_bgr = cv2.inRange(img, lower_bgr, upper_bgr)

        # 在HSV空间中查找
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

        # 合并两个掩码
        mask = cv2.bitwise_or(mask_bgr, mask_hsv)

        # 形态学操作改善检测效果
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.erode(mask, kernel, iterations=1)

        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # 找到最大的轮廓
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)

            if M["m00"] != 0:
                x = int(M["m10"] / M["m00"])
                y = int(M["m01"] / M["m00"])
                return (x, y)

        # 如果上述方法失败，尝试直接查找非零点
        points = cv2.findNonZero(mask)
        if points is not None and len(points) > 0:
            # 计算所有点的平均位置
            x = int(np.mean(points[:, 0, 0]))
            y = int(np.mean(points[:, 0, 1]))
            return (x, y)

        # 添加调试输出
        print(f"无法找到{color}点")
        # 保存调试图
        cv2.imwrite(f"debug_{color}_mask.png", mask)
        cv2.imwrite(f"debug_{color}_original.png", img)

        return None

    def distance(self, p1, p2):
        """计算两点间的欧氏距离"""
        return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def is_collision_free(self, p1, p2, img):
        """优化的碰撞检测方法"""
        if not (0 <= p1[0] < img.shape[1] and 0 <= p1[1] < img.shape[0] and
                0 <= p2[0] < img.shape[1] and 0 <= p2[1] < img.shape[0]):
            return False

        x1, y1 = p1
        x2, y2 = p2

        # 如果两点太近，直接返回
        if abs(x1 - x2) <= 1 and abs(y1 - y2) <= 1:
            # 分别检查两个点
            p1_free = int(img[y1, x1]) >= 200
            p2_free = int(img[y2, x2]) >= 200
            return p1_free and p2_free

        # 使用Bresenham算法获取路径点
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

        # 使用numpy的切片操作检查安全区域
        safety_distance = self.safety_distance
        for x, y in points:
            x_min = max(0, x - safety_distance)
            x_max = min(img.shape[1], x + safety_distance + 1)
            y_min = max(0, y - safety_distance)
            y_max = min(img.shape[0], y + safety_distance + 1)

            # 获检查是否有障碍物
            region = img[y_min:y_max, x_min:x_max]
            if np.any(region < 200):  # 如果区域内有任何像素值小于200
                return False

        return True

    def get_line_points(self, x1, y1, x2, y2):
        """获取线段上的所有点"""
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
        """找到最近节点"""
        distances = [self.distance(point, n) for n in nodes]
        return nodes[np.argmin(distances)]

    def steer(self, from_point, to_point, img):
        """在两点之间按步长生成新点"""
        dist = self.distance(from_point, to_point)
        if dist <= self.step_size:
            return to_point

        theta = np.arctan2(to_point[1] - from_point[1],
                           to_point[0] - from_point[0])

        # 确保新生成的点是整数坐标
        new_x = int(round(from_point[0] + self.step_size * np.cos(theta)))
        new_y = int(round(from_point[1] + self.step_size * np.sin(theta)))

        # 确保点在图像范围内
        new_x = max(0, min(new_x, img.shape[1] - 1))
        new_y = max(0, min(new_y, img.shape[0] - 1))

        return (new_x, new_y)

    def find_neighbors(self, nodes, point):
        """优化的邻居查找方法"""
        neighbors = []
        max_neighbors = 50  # 限制邻居数量

        for node in nodes:
            if len(neighbors) >= max_neighbors:
                break
            if self.distance(node, point) <= self.search_radius:
                neighbors.append(node)

        return neighbors

    def rrt_star(self, img, start, goal):
        """改进的RRT*算法实现，增加错误处理"""
        # 初始化数据结构
        nodes = [start]
        parent = {start: None}
        cost = {start: 0}
        valid_nodes = {start}

        best_cost = float('inf')
        best_path = None

        # 计算基本参数
        dist_to_goal = self.distance(start, goal)
        max_nodes = min(3000, int(dist_to_goal * 5))  # 限制最大节点数
        search_radius = min(self.search_radius, dist_to_goal * 0.3)

        def is_valid_node(node):
            """检查节点是否有效"""
            return (node in valid_nodes and
                    node in cost and
                    0 <= node[0] < img.shape[1] and
                    0 <= node[1] < img.shape[0])

        for i in range(self.max_iterations):
            if len(nodes) >= max_nodes:
                break

            # 智能采样
            if random.random() < self.goal_sample_rate:
                random_point = goal
            else:
                # 在起点和终点间的区域采样
                margin = 50
                x_min = max(0, min(start[0], goal[0]) - margin)
                x_max = min(img.shape[1] - 1, max(start[0], goal[0]) + margin)
                y_min = max(0, min(start[1], goal[1]) - margin)
                y_max = min(img.shape[0] - 1, max(start[1], goal[1]) + margin)

                random_point = (
                    random.randint(x_min, x_max),
                    random.randint(y_min, y_max)
                )

            # 找到最近的有效节点
            valid_distances = [(n, self.distance(random_point, n))
                               for n in valid_nodes]
            if not valid_distances:
                continue

            nearest_node = min(valid_distances, key=lambda x: x[1])[0]

            # 生成新节点
            new_node = self.steer(nearest_node, random_point, img)

            # 验证新节点
            if (not (0 <= new_node[0] < img.shape[1] and
                     0 <= new_node[1] < img.shape[0]) or
                    new_node in valid_nodes or
                    not self.is_collision_free(nearest_node, new_node, img)):
                continue

            # 在有限范围内查找邻居
            neighbors = []
            for node in valid_nodes:
                if len(neighbors) >= 30:  # 限制邻居数量
                    break
                if self.distance(node, new_node) <= search_radius:
                    neighbors.append(node)

            # 选择最佳父节点
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

            # 添加新节点
            nodes.append(new_node)
            valid_nodes.add(new_node)
            parent[new_node] = best_parent
            cost[new_node] = min_cost

            # 重新连接
            for neighbor in neighbors:
                if not is_valid_node(neighbor) or neighbor == best_parent:
                    continue

                potential_cost = cost[new_node] + self.distance(new_node, neighbor)
                if (potential_cost < cost[neighbor] and
                        self.is_collision_free(new_node, neighbor, img)):
                    parent[neighbor] = new_node
                    cost[neighbor] = potential_cost

            # 检查是否可以连接到目标
            if self.distance(new_node, goal) < self.min_distance_to_goal:
                if self.is_collision_free(new_node, goal, img):
                    final_node = self.steer(new_node, goal, img)
                    # 建路径
                    path_nodes = []
                    current = new_node
                    path_length = 0

                    # 使用循环代替递归构建路径
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

                            # 动态调整参数
                            search_radius = min(self.search_radius, current_cost * 0.2)

            # 定期清理无效节点
            if i % 100 == 0 and best_path is not None:
                # 保留最佳路径上的节点及其邻近节点
                keep_nodes = set(best_path)
                for node in best_path:
                    for n in list(valid_nodes)[-200:]:
                        if self.distance(node, n) <= search_radius:
                            keep_nodes.add(n)

                # 更新数据结构
                valid_nodes = {n for n in keep_nodes if is_valid_node(n)}
                nodes = list(valid_nodes)
                parent = {k: v for k, v in parent.items() if k in valid_nodes}
                cost = {k: v for k, v in cost.items() if k in valid_nodes}

        return best_path

    def extract_path(self, parent, node):
        """优化的路径提取方法"""
        path = []
        current = node
        max_path_length = 1000  # 添加最大路径长度限制

        while current is not None and len(path) < max_path_length:
            path.append(current)
            current = parent.get(current)

        if len(path) >= max_path_length:
            print("警告：路径长度超过限制")
            return None

        return path[::-1]

    def optimize_path(self, path, img):
        """优化路径"""
        if len(path) < 3:
            return path

        optimized_path = [path[0]]
        current_point = 0

        while current_point < len(path) - 1:
            # 尝试找到可以直接连接的最远点
            for i in range(len(path) - 1, current_point, -1):
                if self.is_collision_free(path[current_point], path[i], img):
                    optimized_path.append(path[i])
                    current_point = i
                    break
            else:
                current_point += 1
                if current_point < len(path):
                    optimized_path.append(path[current_point])

        # 路径平滑
        smoothed_path = self.smooth_path(optimized_path, img)
        return smoothed_path

    def process_map(self, map_file):
        """改进的地图处理方法"""
        try:
            # 读取地图
            img_path = os.path.join(self.map_dir, map_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"无法读取地图: {map_file}")
                return

            # 添加图像预处理
            # 对比度增强
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced = cv2.merge((cl, a, b))
            img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

            # 查找起点和终点
            start = self.find_colored_point(img, 'green')
            goal = self.find_colored_point(img, 'blue')

            if start is None or goal is None:
                print(f"无法找到起点或终点: {map_file}")
                return

            print(f"找到起点: {start}, 终点: {goal}")

            # 创建二值图像
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

            # 对二值图像进行处理
            kernel = np.ones((5, 5), np.uint8)
            binary = cv2.erode(binary, kernel, iterations=1)
            binary = cv2.dilate(binary, kernel, iterations=1)

            # 确保起点和终点域可通行
            cv2.circle(binary, start, 5, 255, -1)
            cv2.circle(binary, goal, 5, 255, -1)

            # 将所有路径颜色改为绿色
            PATH_COLOR = (0, 255, 0)  # 绿色

            # 用于存储多条路径
            all_paths = []
            path_costs = []

            # 设置路径数量范围
            MIN_PATHS = 15  # 最少路径
            MAX_PATHS = 35  # 最多绘制35条路径
            MAX_SEARCH_PATHS = 60  # 最多寻找60条路径
            num_attempts = 150  # 增加尝试次数，给予更多机会找到路径

            # 在找路径之前创建用于显示的图像副
            display_img = img.copy()

            # 计算基本参数
            dist_to_goal = self.distance(start, goal)
            max_nodes = min(3000, int(dist_to_goal * 5))  # 限制最大节点数
            search_radius = min(self.search_radius, dist_to_goal * 0.3)

            def is_valid_node(node):
                """检查节点是否有效"""
                return (node in valid_nodes and
                        node in cost and
                        0 <= node[0] < img.shape[1] and
                        0 <= node[1] < img.shape[0])

            # 快速行性检查
            def quick_feasibility_check():
                """快速可行性检查"""
                # 1. 检查起点和终点是否在障碍物内
                if binary[start[1], start[0]] < 200 or binary[goal[1], goal[0]] < 200:
                    return False, "起点或终点在障碍物内"

                # 2. 检查周围区域
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
                    return False, "起点或终点周围区域被阻塞"

                return True, "初步检查通过"

            # 执行快速可行性检查
            feasible, reason = quick_feasibility_check()
            if not feasible:
                print(f"地图 {map_file} 快速检查失败: {reason}")
                return False  # 直接返回，不生成失败图像

            # 准备任务参数
            num_attempts = 20
            args_list = [(start, goal, binary, i) for i in range(num_attempts)]

            all_paths = []
            path_costs = []
            failed_attempts = 0
            max_failed_attempts = 3  # 连续3次失败就终止

            # 使用进程池处理
            with Pool(processes=self.num_processes) as pool:
                try:
                    pending_tasks = []

                    # 初始化任务池
                    for i in range(min(self.num_processes, num_attempts)):
                        task = pool.apply_async(self.find_path_attempt, (args_list[i],))
                        pending_tasks.append((i, task))

                    next_task_idx = self.num_processes

                    while pending_tasks:
                        # 检查完成的任务
                        for task_idx, task in pending_tasks[:]:
                            if task.ready():
                                try:
                                    pending_tasks.remove((task_idx, task))
                                    path, cost = task.get(timeout=0.1)

                                    if path is not None:
                                        failed_attempts = 0  # 重置失败计数
                                        all_paths.append(path)
                                        path_costs.append(cost)
                                        print(f"找到第 {len(all_paths)} 条路径")

                                        if len(all_paths) >= self.min_paths:
                                            good_paths = [p for p, c in zip(all_paths, path_costs)
                                                          if c <= min(path_costs) * 1.2]
                                            if len(good_paths) >= self.min_paths:
                                                return self.save_successful_paths(img, start, goal, all_paths,
                                                                                  path_costs, map_file)
                                    else:
                                        failed_attempts += 1
                                        print(f"第 {failed_attempts} 次连续失败")
                                        if failed_attempts >= max_failed_attempts:
                                            print(f"连续 {max_failed_attempts} 次尝试失败，跳过此地图")
                                            return False  # 直接返回，不生成失败图像

                                except Exception as e:
                                    print(f"处理任务 {task_idx} 结果时出错: {str(e)}")
                                    failed_attempts += 1

                                # 添加新任务
                                if next_task_idx < len(args_list):
                                    new_task = pool.apply_async(self.find_path_attempt, (args_list[next_task_idx],))
                                    pending_tasks.append((next_task_idx, new_task))
                                    next_task_idx += 1

                    time.sleep(0.05)

                except Exception as e:
                    print(f"进程池执行出错: {str(e)}")
                    return False
                finally:
                    pool.terminate()
                    pool.join()

            # 如果没有找到足够的路径
            if not all_paths:
                print(f"未能为地图 {map_file} 生成任何有效路径")
                return False  # 直接返回，不生成失败图像

            return False

        except Exception as e:
            print(f"处理地图 {map_file} 时发生错误: {str(e)}")
            return False

    def paths_are_similar(self, path1, path2, threshold=10):
        """检查两路径是否相似"""
        if abs(len(path1) - len(path2)) > len(path1) * 0.3:  # 长度差异过大
            return False

        # 采样点进行比较
        sample_points = min(10, min(len(path1), len(path2)))
        indices = np.linspace(0, min(len(path1), len(path2)) - 1, sample_points, dtype=int)

        for i in indices:
            if self.distance(path1[i], path2[i]) > threshold:
                return False
        return True

    def process_all_maps(self):
        """按编号顺序处理所有地图"""
        if not os.path.exists(self.map_dir):
            print("Map文件夹不存在")
            return

        # 获取目标文件夹中已有的图片的最大序
        existing_files = [f for f in os.listdir(self.enlarged_dir)
                          if f.startswith('path_map_') and f.endswith(('.png', '.jpg', '.jpeg'))]

        max_existing_number = 0
        for file in existing_files:
            try:
                # 从文件名中提取数字 (path_map_X.png)
                number = int(''.join(filter(str.isdigit, file)))
                max_existing_number = max(max_existing_number, number)
            except ValueError:
                continue

        print(f"目标文件夹中已有图片的最大序号: {max_existing_number}")

        # 获取并排序所有地图文件
        map_files = []
        for map_file in os.listdir(self.map_dir):
            if map_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    # 从文件名中提取数字 (map_X.png 或 X.png)
                    number = int(''.join(filter(str.isdigit, map_file)))
                    if number > max_existing_number:  # 只处理序号更大的文件
                        map_files.append((number, map_file))
                except ValueError:
                    continue

        # 按序号排序
        map_files.sort(key=lambda x: x[0])

        if not map_files:
            print(f"没有需要理的新地图（序号大于 {max_existing_number} 的地图）")
            return

        print(f"将处理从 {map_files[0][0]} 到 {map_files[-1][0]} 的地图")

        # 处理地图
        for _, map_file in map_files:
            print(f"\n开始处理地图: {map_file}")
            self.process_map(map_file)

    def smooth_path(self, path, img):
        """平滑路径"""
        if len(path) < 3:
            return path

        smoothed = path.copy()
        change = True
        while change:
            change = False
            for i in range(1, len(smoothed) - 1):
                original = smoothed[i]
                # 尝试在前后点之间平滑
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
        """更新节点的所有子节点的成本"""
        children = [n for n in nodes if parent.get(n) == node]
        for child in children:
            cost[child] = cost[node] + self.distance(node, child)
            self.update_children_cost(child, cost, parent, nodes)

    def path_cost(self, path):
        """计算路径总成本"""
        if not path:
            return float('inf')

        cost = 0
        for i in range(len(path) - 1):
            cost += self.distance(path[i], path[i + 1])
        return cost

    def draw_tree(self, node):
        """绘制树的节点和边"""
        if node.parent:
            # 绘制节点之间的接线(绿色)
            plt.plot([node.x, node.parent.x],
                     [node.y, node.parent.y],
                     '-g', linewidth=1)

            # 绘制节点(绿色)
            plt.plot(node.x, node.y, 'go', markersize=3)

    def visualize(self):
        """设置可视化环境"""
        plt.clf()  # 清除当前图形

        # 绘制起点和终点
        plt.plot(self.start.x, self.start.y, 'bo', label='Start')
        plt.plot(self.goal.x, self.goal.y, 'ro', label='Goal')

        # 绘制障碍物
        self.plot_obstacles()

        # 设置图形性
        plt.grid(True)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('RRT Path Planning')
        plt.axis([self.min_rand, self.max_rand, self.min_rand, self.max_rand])

    def plan(self):
        """RRT主规划循环"""
        for i in range(self.max_iter):
            # 清除并重新绘制
            self.visualize()

            # ��成随机节点
            rnd = self.get_random_point()

            # 获取最近节点
            nearest_node = self.get_nearest_node(self.node_list, rnd)

            # 创建新节点
            new_node = self.steer(nearest_node, rnd, img)

            # 如果无碰撞则添加节点
            if not self.check_collision(new_node):
                self.node_list.append(new_node)
                # 绘制新添加的树枝
                self.draw_tree(new_node)

                # 检查是否到达目标
                if self.is_near_goal(new_node):
                    final_node = self.steer(new_node, self.goal, img)
                    if not self.check_collision(final_node):
                        # 绘制最终路径
                        self.generate_final_course(final_node)
                        break

            plt.pause(0.01)  # 添加短暂暂停以便观察生成过程

    def find_path_attempt(self, args):
        """单次路径寻找尝试"""
        start, goal, binary, attempt_number = args

        try:
            # 重置随机种子，确保每个进程有不同的随机序列
            random.seed(os.getpid() + attempt_number)

            # 初始化数据结构
            nodes = [start]
            parent = {start: None}
            cost = {start: 0}
            valid_nodes = {start}
            best_cost = float('inf')
            best_path = None

            # 计算基本参数
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

                # 智能采样
                if random.random() < self.goal_sample_rate:
                    random_point = goal
                else:
                    # 在起点和终点间的区域采样
                    margin = 50
                    x_min = max(0, min(start[0], goal[0]) - margin)
                    x_max = min(binary.shape[1] - 1, max(start[0], goal[0]) + margin)
                    y_min = max(0, min(start[1], goal[1]) - margin)
                    y_max = min(binary.shape[0] - 1, max(start[1], goal[1]) + margin)

                    random_point = (
                        random.randint(x_min, x_max),
                        random.randint(y_min, y_max)
                    )

                # 找到最近的有效节点
                nearest_node = min(valid_nodes, key=lambda n: self.distance(n, random_point))

                # 生成新节点
                new_node = self.steer(nearest_node, random_point, binary)

                # 验证新节点
                if (not (0 <= new_node[0] < binary.shape[1] and
                         0 <= new_node[1] < binary.shape[0]) or
                        new_node in valid_nodes or
                        not self.is_collision_free(nearest_node, new_node, binary)):
                    continue

                # 在有限范围内查找邻居
                neighbors = [n for n in valid_nodes
                             if self.distance(n, new_node) <= search_radius][:30]

                # 选择最佳父节点
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

                # 添加新节点
                nodes.append(new_node)
                valid_nodes.add(new_node)
                parent[new_node] = best_parent
                cost[new_node] = min_cost

                # 检查是否可以连接到目标
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
            print(f"尝试 {attempt_number} 失败: {str(e)}")
            return None, float('inf')

    def save_failed_image(self, img, start, goal, map_file, reason):
        """保存失败的图像"""
        try:
            # 创建空白图像
            gray_img = np.zeros(img.shape[:2], dtype=np.uint8)

            # 标记起点和终点
            cv2.circle(gray_img, start, 5, 255, -1)
            cv2.circle(gray_img, goal, 5, 255, -1)

            # 保存图像
            output_filename = f"failed_{reason}_{map_file}"
            cv2.imwrite(os.path.join(self.enlarged_dir, output_filename), gray_img)
            print(f"已保存失败图像: {output_filename}")
            return True
        except Exception as e:
            print(f"保存失败图像时出错: {str(e)}")
            return False

    def save_successful_paths(self, img, start, goal, all_paths, path_costs, map_file):
        """保存成功生成的路径图像"""
        try:
            # 将路径和成本打包并排序
            path_pairs = list(zip(all_paths, path_costs))
            path_pairs.sort(key=lambda x: x[1])  # 按照成本排序

            # 只保留前max_paths条最优路径
            path_pairs = path_pairs[:self.max_paths]

            # 创建空白图像
            gray_img = np.zeros(img.shape[:2], dtype=np.uint8)

            # 绘制所有路径
            for path, cost in path_pairs:
                path_array = np.array(path)
                for i in range(len(path_array) - 1):
                    cv2.line(gray_img,
                             tuple(path_array[i]),
                             tuple(path_array[i + 1]),
                             255,  # 白色
                             15)  # 线条宽度
                print(f"路径成本: {cost:.2f}")

            # 标记起点和终点
            cv2.circle(gray_img, start, 5, 255, -1)
            cv2.circle(gray_img, goal, 5, 255, -1)

            # 保存结果
            output_filename = f"path_map_{map_file}"
            cv2.imwrite(os.path.join(self.enlarged_dir, output_filename), gray_img)
            print(f"成功处理地图: {map_file}")
            return True

        except Exception as e:
            print(f"保存成功路径图像时出错: {str(e)}")
            return False


if __name__ == "__main__":
    planner = RRTStarPathPlanner()
    planner.process_all_maps()