import cv2
import numpy as np
import os
from heapq import heappush, heappop
from functools import lru_cache


class PathFinder:
    def __init__(self):
        """初始化路径规划器"""
        self.desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        self.map_dir = os.path.join(self.desktop, "Map")
        self.enlarged_dir = os.path.join(self.desktop, "A_19")
        os.makedirs(self.enlarged_dir, exist_ok=True)

        # 预计算常用值
        self.safety_distance = 4
        self.directions = np.array([
            (0, 1), (1, 0), (0, -1), (-1, 0),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ], dtype=np.int32)  # 指定数据类型提高性能

        # 预计算颜色范围
        self.color_ranges = {
            'green': (np.array([0, 240, 0]), np.array([50, 255, 50])),
            'blue': (np.array([240, 0, 0]), np.array([255, 50, 50]))
        }

        # 预计算对角线移动代价
        self.diagonal_cost = np.sqrt(2)

    def find_colored_point(self, img, color):
        """使用numpy向量化操作优化颜色点查找"""
        lower, upper = self.color_ranges[color]
        mask = cv2.inRange(img, lower, upper)
        points = np.nonzero(mask)
        return (int(np.median(points[1])), int(np.median(points[0]))) if points[0].size > 0 else None

    @lru_cache(maxsize=1024)
    def heuristic(self, a, b):
        """使用缓存优化启发式函数计算"""
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) * 1.1

    def get_neighbors(self, pos, img):
        """使用numpy向量化操作优化邻居节点查找"""
        new_positions = np.array(pos) + self.directions
        valid_mask = (
                (new_positions[:, 0] >= 0) &
                (new_positions[:, 0] < img.shape[1]) &
                (new_positions[:, 1] >= 0) &
                (new_positions[:, 1] < img.shape[0])
        )

        valid_positions = new_positions[valid_mask]
        return [tuple(pos) for pos in valid_positions if self.is_safe_point(pos, img)]

    def is_safe_point(self, point, img):
        """使用numpy切片操作优化安全点检查"""
        x, y = point
        x_min = max(0, x - self.safety_distance)
        x_max = min(img.shape[1], x + self.safety_distance + 1)
        y_min = max(0, y - self.safety_distance)
        y_max = min(img.shape[0], y + self.safety_distance + 1)
        return np.all(img[y_min:y_max, x_min:x_max] >= 200)

    def is_line_free(self, start, end, img):
        """优化直线检查算法"""
        x0, y0 = start
        x1, y1 = end
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        n = 1 + dx + dy
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        error = dx - dy
        dx *= 2
        dy *= 2

        for _ in range(n):
            x_min = max(0, x - self.safety_distance)
            x_max = min(img.shape[1], x + self.safety_distance + 1)
            y_min = max(0, y - self.safety_distance)
            y_max = min(img.shape[0], y + self.safety_distance + 1)

            if not np.all(img[y_min:y_max, x_min:x_max] >= 200):
                return False

            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx
        return True

    def smooth_path(self, path, img):
        """优化路径平滑算法"""
        if len(path) <= 2:
            return path

        smoothed = [path[0]]
        current_idx = 0

        while current_idx < len(path) - 1:
            current = path[current_idx]
            max_reachable_idx = current_idx + 1

            # 使用滑动窗口优化搜索
            for j in range(current_idx + 2, min(current_idx + 20, len(path))):
                if self.is_line_free(current, path[j], img):
                    max_reachable_idx = j
                else:
                    break

            smoothed.append(path[max_reachable_idx])
            current_idx = max_reachable_idx

        return smoothed

    def a_star(self, img, start, goal):
        """优化A*算法实现"""
        if not (start and goal):
            return []

        frontier = []
        heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        while frontier:
            current = heappop(frontier)[1]
            if current == goal:
                break

            for next_pos in self.get_neighbors(current, img):
                dx = abs(next_pos[0] - current[0])
                dy = abs(next_pos[1] - current[1])
                move_cost = self.diagonal_cost if (dx == 1 and dy == 1) else 1
                new_cost = cost_so_far[current] + move_cost

                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self.heuristic(goal, next_pos)
                    heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current

        if goal not in came_from:
            return []

        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = came_from.get(current)
        path.reverse()

        return self.smooth_path(path, img)

    def create_enlarged_path_image(self, path, original_shape, scale=19):
        """优化路径图像生成"""
        h, w = original_shape
        enlarged_path = np.zeros((h, w), dtype=np.uint8)
        if len(path) < 2:
            return enlarged_path

        line_thickness = int(scale)
        path_color = 192

        # 使用向量化操作绘制路径
        path_array = np.array(path)
        for i in range(len(path) - 1):
            cv2.line(enlarged_path, tuple(path_array[i]), tuple(path_array[i + 1]),
                     path_color, thickness=line_thickness, lineType=cv2.LINE_AA)
            if i > 0:
                cv2.circle(enlarged_path, tuple(path_array[i]),
                           line_thickness // 2, path_color, -1, lineType=cv2.LINE_AA)

        # 标记端点
        cv2.circle(enlarged_path, tuple(path_array[0]), line_thickness // 2,
                   path_color, -1, lineType=cv2.LINE_AA)
        cv2.circle(enlarged_path, tuple(path_array[-1]), line_thickness // 2,
                   path_color, -1, lineType=cv2.LINE_AA)

        # 优化发光效果
        blur_sizes = [size for size in [line_thickness, line_thickness // 2, line_thickness // 4]
                      if size >= 3 and size % 2 == 1]

        if blur_sizes:
            glow = enlarged_path.copy()
            for blur_size in blur_sizes:
                temp_glow = cv2.GaussianBlur(glow, (blur_size, blur_size), 0)
                enlarged_path = cv2.addWeighted(enlarged_path, 0.7, temp_glow, 0.3, 0)

        return cv2.normalize(enlarged_path, None, 0, 255, cv2.NORM_MINMAX)

    def process_map(self, map_file):
        """优化地图处理流程"""
        try:
            img_path = os.path.join(self.map_dir, map_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"无法读取地图: {map_file}")
                return

            binary = cv2.threshold(
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                200, 255, cv2.THRESH_BINARY
            )[1]

            start = self.find_colored_point(img, 'green')
            goal = self.find_colored_point(img, 'blue')
            if not (start and goal):
                print(f"无法找到起点或终点: {map_file}")
                return

            kernel = np.ones((9, 9), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            cv2.circle(binary, start, 10, 255, -1)
            cv2.circle(binary, goal, 10, 255, -1)

            path = self.a_star(binary, start, goal)
            if not path:
                print(f"无法找到路径: {map_file}")
                return

            enlarged_path = self.create_enlarged_path_image(path, img.shape[:2])
            cv2.imwrite(os.path.join(self.enlarged_dir, f"enlarged_{map_file}"), enlarged_path)
            print(f"成功处理地图: {map_file}")

        except Exception as e:
            print(f"处理地图 {map_file} 时发生错误: {str(e)}")

    def process_all_maps(self):
        """优化批量处理逻辑"""
        if not os.path.exists(self.map_dir):
            print("Map文件夹不存在")
            return

        map_files = sorted(
            [f for f in os.listdir(self.map_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
            key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else float('inf')
        )

        for map_file in map_files:
            print(f"\n开始处理地图: {map_file}")
            self.process_map(map_file)


if __name__ == "__main__":
    finder = PathFinder()
    finder.process_all_maps()
