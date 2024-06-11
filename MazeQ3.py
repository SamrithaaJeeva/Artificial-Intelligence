import random
import time
import statistics

class PriorityQueue:
    def __init__(self):
        self.queue = []

    def push(self, item, priority):
        entry = (priority, item)
        self.queue.append(entry)
        self._bubble_up(len(self.queue) - 1)

    def pop(self):
        if not self.is_empty():
            self._swap(0, len(self.queue) - 1)
            item = self.queue.pop()[1]
            self._bubble_down(0)
            return item
        return None

    def is_empty(self):
        return len(self.queue) == 0

    def _bubble_up(self, index):
        while index > 0:
            parent_index = (index - 1) // 2
            if self.queue[parent_index][0] > self.queue[index][0]:
                self._swap(parent_index, index)
                index = parent_index
            else:
                break

    def _bubble_down(self, index):
        while True:
            left_child_index = 2 * index + 1
            right_child_index = 2 * index + 2
            smallest = index

            if left_child_index < len(self.queue) and self.queue[left_child_index][0] < self.queue[smallest][0]:
                smallest = left_child_index

            if right_child_index < len(self.queue) and self.queue[right_child_index][0] < self.queue[smallest][0]:
                smallest = right_child_index

            if smallest != index:
                self._swap(index, smallest)
                index = smallest
            else:
                break

    def _swap(self, i, j):
        self.queue[i], self.queue[j] = self.queue[j], self.queue[i]


class Maze:
    def __init__(self, width, height, start, goal, barriers):
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.barriers = barriers
        self.maze = self.setup_maze()

    def setup_maze(self):
        maze = [[' ' for _ in range(self.width)] for _ in range(self.height)]

        start_x, start_y = self.index_to_coordinates(self.start)
        goal_x, goal_y = self.index_to_coordinates(self.goal)
        maze[start_y][start_x] = 'S'
        maze[goal_y][goal_x] = 'G'

        for barrier in self.barriers:
            barrier_x, barrier_y = self.index_to_coordinates(barrier)
            maze[barrier_y][barrier_x] = '#'

        return maze

    def index_to_coordinates(self, index):
        return index % self.width, index // self.width

    def print_maze(self):
        print('    ' + '   '.join(f'{y:<2}' for y in range(self.height)))
        print('   ' + '+'.join(['---'] * self.height))

        for x in range(self.width):
            print(f'{x:<2} |', end='')

            for y in range(self.height):
                if self.maze[y][x] == ' ':
                    print(f' {y * self.width + x:<2} |', end='')
                else:
                    print(f' {self.maze[y][x]:<2} |', end='')
            print('\n   ' + '+'.join(['---'] * self.height))

    def dfs_analysis(self, start, goal):
        start_time = time.perf_counter()
        visited = set()
        stack = [(start, [start])]

        end_time = 0

        while stack:
            node, path = stack.pop()
            if node not in visited:
                visited.add(node)

                if node == goal:
                    end_time = time.perf_counter()
                    return {
                        'path': path,
                        'time': (end_time - start_time) / 60,  # Convert seconds to minutes
                        'length': len(path) - 1
                    }

                neighbors = self.get_neighbors(node)

                for neighbor in neighbors:
                    if neighbor not in visited and neighbor not in self.barriers:
                        stack.append((neighbor, path + [neighbor]))

        return {
            'path': [],
            'time': (end_time - start_time) / 60,  # Convert seconds to minutes
            'length': 0
        }

    def a_star(self):
        priority_queue = PriorityQueue()
        start_time = time.perf_counter()
        start_node = self.start
        goal_node = self.goal

        priority_queue.push((start_node, [start_node]), self.calculate_manhattan_distance(start_node))
        visited = set()

        while not priority_queue.is_empty():
            current, path = priority_queue.pop()

            node = current

            if node not in visited:
                visited.add(node)

                if node == goal_node:
                    end_time = time.perf_counter()
                    return {
                        'path': path,
                        'time': (end_time - start_time) / 60,  # Convert seconds to minutes
                        'length': len(path) - 1
                    }
                neighbors = self.get_neighbors(node)

                for neighbor in neighbors:
                    if neighbor not in visited and neighbor not in self.barriers:
                        new_path = path + [neighbor]
                        priority = len(new_path) + self.calculate_manhattan_distance(neighbor)
                        priority_queue.push((neighbor, new_path), priority)
        return None

    def calculate_manhattan_distance(self, node):
        goal_x, goal_y = self.index_to_coordinates(self.goal)
        node_x, node_y = self.index_to_coordinates(node)
        return abs(node_x - goal_x) + abs(node_y - goal_y)

    def get_neighbors(self, node):
        x, y = self.index_to_coordinates(node)
        neighbors = []

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < self.width and 0 <= new_y < self.height:
                    neighbors.append(self.coordinates_to_index(new_x, new_y))

        return neighbors

    def coordinates_to_index(self, x, y):
        return y * self.width + x


def bubble_sort(arr):
    n = len(arr)

    for i in range(n - 1):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]


def custom_shuffle(arr):
    n = len(arr)
    for i in range(n - 1, 0, -1):
        j = random.randint(0, i)
        arr[i], arr[j] = arr[j], arr[i]          


                

def main():
    width, height = 6, 6
    num_mazes = 3
    dfs_times = []
    dfs_lengths = []
    a_star_times = []
    a_star_lengths = []

    for maze_num in range(1, num_mazes + 1):
        print(f"\nMaze {maze_num}:")

        start_node = random.randint(0, 11)
        goal_node = random.randint(24, 35)
        all_nodes = list(set(range(36)) - {start_node, goal_node})
        # Using the custom shuffle function instead of random.shuffle
        custom_shuffle(all_nodes)
        barrier_nodes = all_nodes[:4]

        maze = Maze(width=width, height=height, start=start_node, goal=goal_node, barriers=barrier_nodes)

        print("\nInitial Maze:")
        maze.print_maze()

        print("\nDFS Analysis:")
        dfs_result = maze.dfs_analysis(start_node, goal_node)

        if dfs_result['path']:
            sorted_dfs_path = dfs_result['path'][:-1].copy()
            bubble_sort(sorted_dfs_path)
            print("Visited Nodes (Processing Order):", sorted_dfs_path)
            print("Time to Find Goal (seconds):", dfs_result['time'])
            print("Final Path:", dfs_result['path'])
            dfs_times.append(dfs_result['time'])
            dfs_lengths.append(dfs_result['length'])
        else:
            print("Goal not reachable.")

        print("\nHeuristic Cost Calculation:")
        for node in range(width * height):
            heuristic_cost = maze.calculate_manhattan_distance(node)
            print(f"Node {node}: Heuristic Cost = {heuristic_cost}")

        print("\nA* Analysis:")
        a_star_result = maze.a_star()

        if a_star_result:
            sorted_a_star_path = a_star_result['path'][:-1].copy()
            bubble_sort(sorted_a_star_path)
            print("Visited Nodes (Processing Order):", a_star_result['path'][:-1])
            print("Time to Find Goal (seconds):", a_star_result['time'])
            print("Final Path:", a_star_result['path'])
            a_star_times.append(a_star_result['time'])
            a_star_lengths.append(a_star_result['length'])
        else:
            print("Goal not reachable.")

    print("\nAnalysis of Results:")
    print("DFS Times (minutes):", dfs_times)
    print("DFS Mean Time (minutes):", statistics.mean(dfs_times))
    print("DFS Variance Time (minutes):", statistics.variance(dfs_times))
    print("DFS Lengths:", dfs_lengths)
    print("DFS Mean Length:", statistics.mean(dfs_lengths))
    print("DFS Variance Length:", statistics.variance(dfs_lengths))

    print("\nA* Times (minutes):", a_star_times)
    print("A* Mean Time (minutes):", statistics.mean(a_star_times))
    print("A* Variance Time (minutes):", statistics.variance(a_star_times))
    print("A* Lengths:", a_star_lengths)
    print("A* Mean Length:", statistics.mean(a_star_lengths))
    print("A* Variance Length:", statistics.variance(a_star_lengths))


if __name__ == "__main__":
    main()
