# astar_estop.py

A* 경로 계획과 Pure Pursuit 경로 추종으로 TurtleBot3 를 목적지까지 주행하고,
카메라에 사람이 보이면 YOLOv8 검출로 즉시 비상정지하는 ROS 2 노드다.

## 실행

```bash
# 터틀봇
ros2 launch turtlebot3_bringup robot.launch.py
ros2 launch turtlebot3_bringup camera.launch.py format:=RGB888

# Remote PC: 지도(map_server)와 AMCL 을 띄우고 RViz 에서 2D Pose Estimate 로 초기 위치를 지정한 뒤
pip install ultralytics      # 첫 실행 때 yolov8n.pt 를 자동으로 내려받는다
python3 astar_estop.py
```

RViz 의 2D Goal Pose 로 목적지를 찍으면 경로를 계산해 출발한다.

## 동작 구조

1. **경로 계획.** `/goal_pose` 를 받으면 현재 위치와 목적지를 지도 격자 좌표로 바꾸고 A* 로 경로를 구한다.
   8방향 이동, 유클리드 거리 휴리스틱을 쓰며, 벽에서 `safety_margin` 칸 이내인 칸은 지나가지 않는다.
2. **경로 추종.** 0.1초마다 경로 위에서 `lookahead_dist` 만큼 앞의 점을 목표로 잡고,
   Pure Pursuit 식으로 각속도를 계산해 `/cmd_vel` 로 보낸다.
3. **비상정지.** 카메라 프레임마다 YOLOv8 로 사람을 찾는다. 아래 경우에는 경로를 유지한 채 정지하고,
   이유가 사라지면 남은 경로를 이어서 주행한다.
   - 사람이 검출됨. 마지막 검출 후 `person_hold_sec` 동안 정지를 유지한다.
   - 카메라 영상이 `camera_timeout` 이상 끊김.

## 토픽

| 방향 | 토픽 | 타입 | 비고 |
|---|---|---|---|
| 구독 | `/map` | `nav_msgs/OccupancyGrid` | QoS TRANSIENT_LOCAL. 노드를 늦게 켜도 지도를 받는다 |
| 구독 | `/amcl_pose` | `geometry_msgs/PoseWithCovarianceStamped` | 현재 위치와 방향 |
| 구독 | `/goal_pose` | `geometry_msgs/PoseStamped` | 목적지 |
| 구독 | `/camera/image_raw/compressed` | `sensor_msgs/CompressedImage` | QoS BEST_EFFORT, depth 1. 항상 최신 프레임만 처리한다 |
| 발행 | `/cmd_vel` | `geometry_msgs/Twist` | 주행 명령 |
| 발행 | `/planned_path` | `nav_msgs/Path` | RViz 경로 표시용 |

## 파라미터

| 이름 | 기본값 | 의미 |
|---|---|---|
| `lookahead_dist` | 0.5 m | 전방 주시 거리. 크면 부드럽게, 작으면 경로에 바짝 붙어 주행한다 |
| `linear_vel` | 0.2 m/s | 직진 속도 |
| `stop_tolerance` | 0.15 m | 도착으로 인정하는 거리 |
| `safety_margin` | 4 칸 | 벽과의 최소 간격. 해상도 0.05 m 기준 0.2 m |
| `person_conf` | 0.7 | YOLO 신뢰도 하한. 사람을 놓치면 낮춘다 |
| `person_hold_sec` | 1.0 s | 사람이 사라진 뒤 재출발까지 대기 시간 |
| `camera_timeout` | 1.0 s | 영상이 이 시간 이상 끊기면 정지 |
| `show_window` | True | 검출 화면 표시. 화면 없는 환경에서는 False |

## 안전 처리

- 새 목적지를 받으면 A* 계산 전에 먼저 정지한다. 계산 중에는 카메라 콜백이 돌지 못하기 때문이다.
- Ctrl+C 로 종료해도 정지 명령을 보낸 뒤 종료한다. 이를 위해 rclpy 기본 시그널 처리를 끄고 직접 shutdown 한다.
