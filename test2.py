

import asyncio
from mavsdk import System
from mavsdk.offboard import PositionNedYaw, OffboardError
import threading
import cv2
from ultralytics import YOLO
import time

# Shared waypoint queue
waypoints_queue = asyncio.Queue()

# Global drone reference for landing
drone_global = None

# ========================== CONNECT TO DRONE ==========================
async def connect():
    drone = System()
    await drone.connect(system_address="udp://:14540")  # PX4 SITL default port
    print("🔌 Connecting to drone...")

    async for state in drone.core.connection_state():
        if state.is_connected:
            print("✅ Drone connected")
            break

    return drone

# ====================== ARM, TAKEOFF, OFFBOARD =======================
async def arm_and_takeoff(drone):
    print("🔋 Arming drone...")
    await drone.action.arm()

    print("🚀 Taking off...")
    await drone.action.set_takeoff_altitude(3.0)
    await drone.action.takeoff()
    await asyncio.sleep(5)

    # Send initial setpoint before starting offboard
    print("📡 Sending initial offboard setpoint...")
    initial_position = PositionNedYaw(0.0, 0.0, -3.0, 0.0)
    await drone.offboard.set_position_ned(initial_position)

    try:
        await drone.offboard.start()
        print("✅ Offboard mode started")
    except OffboardError as e:
        print(f"❌ Offboard start failed: {e._result.result}")
        await drone.action.disarm()
        return False

    return True

# ==================== CONTINUOUS MOVEMENT LOOP =======================
# ==================== CONTINUOUS MOVEMENT LOOP =======================
async def move_loop(drone):
    current_position = PositionNedYaw(0.0, 0.0, -3.0, 0.0)
    last_send_time = asyncio.get_event_loop().time()

    while True:
        now = asyncio.get_event_loop().time()

        if not waypoints_queue.empty():
            wp = await waypoints_queue.get()
            # Update current position RELATIVELY
            current_position = PositionNedYaw(
                current_position.north_m + wp[0],
                current_position.east_m + wp[1],
                current_position.down_m + wp[2],
                0.0
            )
            print(f"🛰️ Moving to: {current_position.north_m}N, {current_position.east_m}E, {current_position.down_m}D")

        if now - last_send_time >= 0.1:
            try:
                await drone.offboard.set_position_ned(current_position)
                print(f"📤 Sending: NED({current_position.north_m}, {current_position.east_m}, {current_position.down_m})")
            except OffboardError as e:
                print(f"⚠️ Failed to send position: {e}")
            last_send_time = now

        await asyncio.sleep(0.05)


# ========================== USER INPUT THREAD ========================
def user_input():
    while True:
        cmd = input("📍 Enter waypoint (N E D) or type 'land': ")
        if cmd.lower() == "land":
            asyncio.run_coroutine_threadsafe(land(drone_global), asyncio.get_event_loop())
            break
        try:
            n, e, d = map(float, cmd.split())
            asyncio.run_coroutine_threadsafe(waypoints_queue.put((n, e, d)), asyncio.get_event_loop())
        except Exception as e:
            print(f"❌ Invalid input: {e}. Use format: 5 5 -3")

# ============================ LANDING ================================
async def land(drone):
    print("🛬 Landing...")
    await drone.action.land()
    await asyncio.sleep(6)
    print("🔻 Disarming...")
    await drone.action.disarm()
    print("✅ Drone landed and disarmed.")

# ===================== CAMERA AND YOLOv8 FUNCTIONALITY =================
async def camera_and_tracking(drone):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error opening camera.")
        return

    model = YOLO("yolov8n.pt")  # Load YOLOv8 model (ensure yolov8n.pt is available)
    
    selected_bbox = None
    selected_id = None
    selected_class = None
    selected_confidence = None
    tracking_initialized = False

    def select_object(event, x, y, flags, param):
        nonlocal selected_bbox, selected_id, selected_class, selected_confidence, tracking_initialized
        if event == cv2.EVENT_LBUTTONDOWN:
            for result in results:
                bbox = result.boxes.xyxy[0].cpu().numpy()
                if bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]:
                    selected_bbox = bbox
                    selected_id = result.boxes.id[0].item()
                    selected_class = result.names[int(result.boxes.cls[0].item())]
                    selected_confidence = result.boxes.conf[0].item()
                    tracking_initialized = True
                    break

    cv2.namedWindow("YOLOv8 Tracking")
    cv2.setMouseCallback("YOLOv8 Tracking", select_object)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True)
        annotated_frame = frame.copy()
        h, w = frame.shape[:2]
        frame_center = (w // 2, h // 2)

        movement_needed = False
        waypoints = []

        if tracking_initialized and selected_id is not None:
            for result in results:
                for i, bbox in enumerate(result.boxes.xyxy):
                    if result.boxes.id[i].item() == selected_id:
                        selected_bbox = bbox
                        cv2.rectangle(annotated_frame,
                                      (int(bbox[0]), int(bbox[1])),
                                      (int(bbox[2]), int(bbox[3])),
                                      (0, 255, 0), 2)
                        label = f"ID: {selected_id}, Class: {selected_class}, Conf: {selected_confidence:.2f}"
                        cv2.putText(annotated_frame, label, (int(bbox[0]), int(bbox[1]) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        bbox_center = ((int(bbox[0]) + int(bbox[2])) // 2, (int(bbox[1]) + int(bbox[3])) // 2)
                        cv2.circle(annotated_frame, bbox_center, 50, (0, 0, 255), -1)
                        cv2.circle(annotated_frame, frame_center, 50, (255, 0, 0), -1)

                        # if bbox_center[0] < frame_center[0] - 90:
                        #     print("📍 Direction: Move Right")
                        #     waypoints = [(0.0, 5.0, -3.0)]  # Move Right
                        #     movement_needed = True
                        # elif bbox_center[0] > frame_center[0] + 90:
                        #     print("📍 Direction: Move Left")
                        #     waypoints = [(0.0, -5.0, -3.0)]  # Move Left
                        #     movement_needed = True
                        # # Forward/Backward logic
                        # if bbox_center[1] > frame_center[1] + 90:
                        #     print("📍 Direction: Move Forward")
                        #     waypoints.append((3.0, 0.0, -3.0))  # Move Forward (NED: forward is positive North)
                        #     movement_needed = True
                        # elif bbox_center[1] < frame_center[1] - 90:
                        #     print("📍 Direction: Move Backward")
                        #     waypoints.append((-3.0, 0.0, -3.0))  # Move Backward
                        #     movement_needed = True
                        if bbox_center[0] < frame_center[0] - 90:
                            print("📍 Direction: Move Right")
                            waypoints = [(0.0, 5.0, 0.0)]  # Move Right
                            movement_needed = True
                        elif bbox_center[0] > frame_center[0] + 90:
                            print("📍 Direction: Move Left")
                            waypoints = [(0.0, -5.0, 0.0)]  # Move Left
                            movement_needed = True
                        # Forward/Backward logic
                        if bbox_center[1] > frame_center[1] + 90:
                            print("📍 Direction: Move Forward")
                            waypoints.append((3.0, 0.0, 0.0))  # Move Forward (NED: forward is positive North)
                            movement_needed = True
                        elif bbox_center[1] < frame_center[1] - 90:
                            print("📍 Direction: Move Backward")
                            waypoints.append((-3.0, 0.0, 0.0))  # Move Backward
                            movement_needed = True
                        break

        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if movement_needed:
            print("⏸️ Pausing camera feed to move drone...")
            cap.release()
            cv2.destroyAllWindows()
            for wp in waypoints:
                await waypoints_queue.put(wp)
                await asyncio.sleep(2)
            print("▶️ Resuming camera feed...")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("❌ Error reopening camera after movement.")
                break

    cap.release()
    cv2.destroyAllWindows()

# ============================ MAIN LOOP ==============================
async def main():
    global drone_global
    drone = await connect()
    drone_global = drone

    if not await arm_and_takeoff(drone):
        return

    # Start user input thread and movement loop
    asyncio.create_task(move_loop(drone))
    threading.Thread(target=user_input, daemon=True).start()

    # Start the camera and YOLO tracking
    await camera_and_tracking(drone)
    await land(drone)
    # await arm_and_takeoff(drone)

# ============================= START ================================
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("🛑 Interrupted by user.")




