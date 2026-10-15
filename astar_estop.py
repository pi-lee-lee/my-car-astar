#   터틀봇:    ros2 launch turtlebot3_bringup robot.launch.py
#              ros2 launch turtlebot3_bringup camera.launch.py format:=RGB888
#   Remote PC: 지도·AMCL(위치 추정)을 띄우고 RViz 에서 2D Pose Estimate 를 찍어 둔 상태에서
#              python3 astar_estop.py
#              → RViz 의 2D Goal Pose 로 목적지를 찍으면 경로를 계산해 출발
#
#   필요: pip install ultralytics  (첫 실행 때 yolov8n.pt 를 실행한 폴더에 내려받는다)
#   ※ Gazebo 시뮬에서는 image_transport_plugins 가 설치돼 있어야 /camera/image_raw/compressed 가 나온다.
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.signals import SignalHandlerOptions
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import CompressedImage
from math import pow, atan2, sqrt, sin, cos, pi
import heapq
import numpy as np
import cv2
from ultralytics import YOLO

class NodeAStar:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0
    def __eq__(self, other): return self.position == other.position
    def __lt__(self, other): return self.f < other.f

class IntegratedNavigation(Node):
    def __init__(self):
        super().__init__('integrated_navigation')

        # 여기가 바로 당신들이 수정할 파라미터
        self.lookahead_dist = 0.5  # 전방 주시 거리 : 멀리 보냐, 가까이 보냐에 따른 차이 발생함
        self.linear_vel = 0.2      # 이동 속도 : 느리냐, 빠르냐에 따른 차이 발생함
        self.stop_tolerance = 0.15 # 도착 인정 거리 : 가까우면 멈추냐, 멀어도 멈추냐에 따른 차이 발생함

        # 장애물에 부딪히지 않기 위한 안전거리 (x칸)
        self.safety_margin = 4 # 맵을 격자로 구분했을 때의 칸수

        # [비상정지 추가] 사람 검출 파라미터
        self.person_conf = 0.7      # YOLO 신뢰도 하한 (docs 참고자료 값). 로봇 카메라는 낮아서 다리만 보일 때가 많다. 사람을 놓치면 낮춘다
        self.person_hold_sec = 1.0  # 사람이 이 시간 동안 안 보여야 주행 재개 (한두 프레임 검출이 빠진 것으로 다시 출발하지 않게)
        self.camera_timeout = 1.0   # 마지막 검출 프레임이 이보다 오래되면 정지. YOLO 한 장 추론 시간보다 커야 한다
        self.show_window = True     # 검출 화면 띄우기 (화면 없는 환경에서 돌릴 때는 False)

        self.map_data = None
        self.map_resolution = 0.05
        self.map_origin = [0.0, 0.0]
        self.map_width = 0
        self.map_height = 0

        self.current_pose = None
        self.current_yaw = 0.0
        self.global_path = [] # 계획된 경로
        self.path_index = 0

        # [비상정지 추가] 검출 상태
        self.model = YOLO('yolov8n.pt')  # 모델은 노드를 만들 때 한 번만 불러온다
        self.last_frame_time = None      # 마지막으로 검출을 마친 프레임을 받은 시각
        self.last_person_time = None     # 마지막으로 사람을 본 시각
        self.estop_reason = None         # 지금 멈춰 있는 이유. None 이면 주행 중 (바뀔 때만 로그를 찍으려고 둔다)

        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_path = self.create_publisher(Path, '/planned_path', 10)

        # [수정] map_server 는 지도를 한 번만 발행한다(transient local). 같은 설정으로 받아야 노드를 나중에 켜도 지도를 받는다
        map_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.sub_map = self.create_subscription(OccupancyGrid, '/map', self.map_callback, map_qos)
        self.sub_pose = self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.pose_callback, 10)
        self.sub_goal = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)

        # [비상정지 추가] 터틀봇 카메라 구독 (docs 참고자료의 웹캠 cv2.VideoCapture(0) 자리)
        # 압축 토픽: 무압축 image_raw 는 장당 약 0.9MB 라 Wi-Fi 로 받으면 지연이 쌓인다
        # depth=1: YOLO 가 카메라 프레임 속도보다 느리므로, 밀린 옛 프레임 대신 항상 가장 최근 프레임을 받는다
        camera_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.sub_image = self.create_subscription(CompressedImage, '/camera/image_raw/compressed', self.image_callback, camera_qos)

        # 이 주기를 바꾸는 것도 주행 퍼포먼스에 영향을 줄 수 있음
        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info("Let's Run!")

    def map_callback(self, msg):
        self.map_resolution = msg.info.resolution
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_origin = [msg.info.origin.position.x, msg.info.origin.position.y]
        self.map_data = np.array(msg.data).reshape((self.map_height, self.map_width))

    def pose_callback(self, msg):
        self.current_pose = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        q = msg.pose.pose.orientation
        self.current_yaw = atan2(2.0*(q.w*q.z + q.x*q.y), 1.0-2.0*(q.y*q.y + q.z*q.z))

    # [비상정지 추가] 카메라 프레임마다 YOLO 로 사람을 찾는다 (docs 의 YOLOv8 참고자료를 카메라 토픽용으로 옮긴 것)
    def image_callback(self, msg):
        received = self.get_clock().now()
        frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None: return

        # 웹캠 예제의 cv2.flip(좌우 반전)은 거울처럼 보이게 하려는 것이라 로봇 카메라에서는 뺐다
        # verbose=False : 켜 두면 ultralytics 가 프레임마다 검출 결과를 출력한다
        results = self.model(frame, stream=True, conf=self.person_conf, verbose=False)
        person_detected = False

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                name = self.model.names[cls]
                if name == 'person':
                    person_detected = True
                color = (127, 0, 127)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        self.last_frame_time = received
        if person_detected:
            self.last_person_time = received
            if self.global_path:
                self.stop_robot() # 다음 control_loop 를 기다리지 않고 바로 세운다

        if self.show_window:
            cv2.imshow('frame', frame)
            cv2.waitKey(1)

    # [비상정지 추가] 지금 멈춰야 하는 이유를 돌려준다. 멈출 이유가 없으면 None
    def estop_check(self):
        now = self.get_clock().now()
        if self.last_frame_time is None or (now - self.last_frame_time).nanoseconds * 1e-9 > self.camera_timeout:
            return "카메라 영상이 들어오지 않음"
        if self.last_person_time is not None and (now - self.last_person_time).nanoseconds * 1e-9 < self.person_hold_sec:
            return "사람 감지"
        return None

    # goal_pose 토픽 발행 시
    # 1. 맵을 그리드 형태로 변환한 다음 최단 경로를 구한다.
    # 2. 구한 경로를 다시 원래의 형태로 변환한 다음 발행한다.
    def goal_callback(self, msg):
        if self.map_data is None or self.current_pose is None: return

        # [비상정지 추가] A* 계산은 몇 초 걸릴 수 있고, 그동안은 카메라 콜백도 돌지 못해 사람을 볼 수 없다.
        # 주행 중에 새 목적지를 받았다면 먼저 세워 두고 계산한다
        self.stop_robot()

        goal_pose = [msg.pose.position.x, msg.pose.position.y]

        # x,y 좌표 값을 그리드 상에서의 행,열 로 변환하는 작업
        start_grid = self.world_to_grid(self.current_pose)
        goal_grid = self.world_to_grid(goal_pose)

        self.get_logger().info("Calculating Path...")
        path_grid = self.run_astar(start_grid, goal_grid)

        if path_grid:
            self.global_path = [self.grid_to_world(p) for p in path_grid]
            self.path_index = 0
            self.publish_path_viz()
            self.get_logger().info("Path Found! Go!")
        else:
            self.get_logger().warn("No Path Found.")

    # 안전거리(마진)를 유지하는지 체크.... 안전거리보다 좁혀지지 않게!
    def check_safety(self, y, x) :
        margin = self.safety_margin

        for r in range(y - margin, y + margin + 1):
            for c in range(x - margin, x + margin + 1):
                if 0 <= r < self.map_height and 0 <= c < self.map_width:
                    if self.map_data[r][c] != 0: # 0이 아닌 게 장애물
                        return False
        return True

    def run_astar(self, start, end):
        if not (0 <= start[0] < self.map_height and 0 <= start[1] < self.map_width): return None
        if not (0 <= end[0] < self.map_height and 0 <= end[1] < self.map_width): return None

        start_node = NodeAStar(None, start)
        end_node = NodeAStar(None, end)
        open_list = []
        heapq.heappush(open_list, start_node)
        visited = set()
        moves = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]

        while open_list:
            current_node = heapq.heappop(open_list)
            if current_node.position in visited: continue
            visited.add(current_node.position)

            if current_node.position == end_node.position:
                path = []
                current = current_node
                while current:
                    path.append(current.position)
                    current = current.parent
                return path[::-1]

            for move in moves:
                ny, nx = current_node.position[0] + move[0], current_node.position[1] + move[1]
                if not (0 <= ny < self.map_height and 0 <= nx < self.map_width): continue
                if self.map_data[ny][nx] != 0: continue

                if not self.check_safety(ny, nx) : continue

                new_node = NodeAStar(current_node, (ny, nx))
                new_node.g = current_node.g + 1
                new_node.h = sqrt((ny - end[0])**2 + (nx - end[1])**2)
                new_node.f = new_node.g + new_node.h
                heapq.heappush(open_list, new_node)
        return None

    # 경로는 에이스타가 구해줬다는 가정 하에, 나는 이동만 하면 돼!
    def control_loop(self):
        if not self.global_path: return

        # [비상정지 추가] 멈출 이유가 있으면 경로는 그대로 두고 멈춰 있기만 한다. 이유가 사라지면 남은 경로를 이어서 간다
        reason = self.estop_check()
        if reason != self.estop_reason: # 상태가 바뀔 때만 로그
            if reason:
                self.get_logger().warn(f"비상정지! ({reason})")
            else:
                self.get_logger().info("비상정지 해제 — 주행 재개")
            self.estop_reason = reason
        if reason:
            self.stop_robot()
            return

        final_goal = self.global_path[-1] # 목적지

        # 목적지까지 가기 위해 계산해 본 거리
        dist_to_final = sqrt((final_goal[0]-self.current_pose[0])**2 + (final_goal[1]-self.current_pose[1])**2)

        if dist_to_final < self.stop_tolerance:
            self.global_path = []
            self.stop_robot()
            return

        # 도착 지점의 좌표
        target_x, target_y = final_goal

        # 경로가 점 10개짜리 경로라고 쳐봐....
        # 그러면 10번의 인덱싱을 할 수 있는거야!
        for i in range(self.path_index, len(self.global_path)):
            px, py = self.global_path[i]
            dist = sqrt((px - self.current_pose[0])**2 + (py - self.current_pose[1])**2)
            # 내가 주시하는 거리보다 조금 먼 정도의 목적지를 설정
            if dist >= self.lookahead_dist:
                target_x, target_y = px, py # 적절한 타겟으로 갱신!
                self.path_index = i
                break

        # 새로 갱신한 현실적인 목적지까지의 거리 기반으로 계산
        dx = target_x - self.current_pose[0]
        dy = target_y - self.current_pose[1]
        alpha = atan2(dy, dx) - self.current_yaw

        if alpha > pi: alpha -= 2*pi
        elif alpha < -pi: alpha += 2*pi

        angular_velocity = self.linear_vel * (2.0 * sin(alpha)) / self.lookahead_dist

        cmd = Twist()
        cmd.linear.x = self.linear_vel
        cmd.angular.z = angular_velocity

        if cmd.angular.z > 1.0: cmd.angular.z = 1.0
        if cmd.angular.z < -1.0: cmd.angular.z = -1.0

        self.pub_cmd.publish(cmd)

    def world_to_grid(self, world):
        return (int((world[1]-self.map_origin[1])/self.map_resolution), int((world[0]-self.map_origin[0])/self.map_resolution))

    def grid_to_world(self, grid):
        return [(grid[1]*self.map_resolution)+self.map_origin[0], (grid[0]*self.map_resolution)+self.map_origin[1]]

    # 계획된 경로 발행
    def publish_path_viz(self):
        msg = Path()
        msg.header.frame_id = 'map'
        for p in self.global_path:
            ps = PoseStamped()
            ps.pose.position.x, ps.pose.position.y = p[0], p[1]
            msg.poses.append(ps)
        self.pub_path.publish(msg)

    def stop_robot(self):
        self.pub_cmd.publish(Twist())

def main(args=None):
    # [수정] rclpy 기본 시그널 처리는 Ctrl+C 에서 컨텍스트부터 닫아 버려 아래 finally 의 정지 명령이 발행되지 못한다.
    # 끄고 KeyboardInterrupt 만 받아서, 정지 명령을 보낸 뒤 직접 shutdown 한다.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = IntegratedNavigation()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot() # 어떤 이유로 끝나든 정지 명령을 남긴다
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()
