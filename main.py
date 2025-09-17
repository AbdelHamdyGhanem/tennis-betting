import cv2
import math
from ultralytics import YOLO
from sort.sort import Sort
import os
import numpy as np
import random
import time

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Global variables for tracking selections
selected_bet = None
selected_amount = None
bet_confirmed = False

# Player tracking variables
player_names = {}
player_positions = {}
player_speeds = {}
player_distances = {}
name_assignment_order = []

# Notification variables - using time-based system for smooth output
notification_text = ""
notification_visible = False
notification_start_time = 0
notification_interval = 8.0  # seconds
notification_duration = 3.0  # seconds
notification_coords = (20, 100, 320, 180)
last_notification_time = 0

# Coordinates for clickable areas
option_coords = []
amount_coords = []
confirm_coords = (0, 0, 0, 0)
new_bet_coords = (0, 0, 0, 0)
dashboard_visible = True
toggle_button_coords = (0, 0, 0, 0)

# Enhanced player names list
player_names_list = [
    "Djokovic", "Federer", "Nadal", "Murray", "Tsitsipas",
    "Zverev", "Medvedev", "Thiem", "Berrettini", "Rublev",
    "Alcaraz", "Sinner", "Ruud", "Fritz", "Hurkacz",
    "Norrie", "Auger-Aliassime", "Shapovalov", "Kyrgios", "De Minaur"
]

bet_amount_options = ["$10", "$20", "$50", "$100"]
PIXELS_TO_METERS = 0.05

# OPTIMIZATION: Cache system for expensive operations
dashboard_overlay_cache = None
dashboard_cache_valid = False
dashboard_cache_frame_size = None
last_cache_update = 0
notification_overlay_cache = None
current_betting_options = []

# OPTIMIZATION: Reduce calculation frequency
DETECTION_INTERVAL = 5  # Run detection every 5 frames instead of 3
SPEED_CALC_INTERVAL = 2  # Update speed every 2 frames
DASHBOARD_UPDATE_INTERVAL = 10  # Update dashboard cache every 10 frames
NOTIFICATION_UPDATE_INTERVAL = 30  # Check notifications every 30 frames

def assign_player_name(obj_id):
    """Assign a unique name to a player based on their ID - OPTIMIZED"""
    if obj_id not in player_names:
        used_names = set(player_names.values())
        available_names = [name for name in player_names_list if name not in used_names]
        
        if available_names:
            # Simplified assignment logic
            idx = len(player_names) % len(available_names)
            assigned_name = available_names[idx]
            player_names[obj_id] = assigned_name
            name_assignment_order.append(obj_id)
        else:
            assigned_name = random.choice(player_names_list)
            player_names[obj_id] = assigned_name

    return player_names[obj_id]

def generate_notification():
    """Generate random tennis notification - CACHED"""
    events = [
        f"Ace probability: {random.randint(15,35)}%",
        f"Winner shot: {random.randint(40,80)}%",
        f"Double fault risk: {random.randint(5,25)}%",
        f"Break point chance: {random.randint(30,70)}%",
        f"Rally length 10+: {random.randint(20,60)}%"
    ]
    return random.choice(events)

def create_notification_overlay(frame_shape, text):
    """Create cached notification overlay"""
    h, w = frame_shape[:2]
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    
    x1, y1, x2, y2 = notification_coords
    x2 = x2 + 300

    # Draw background
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (40, 40, 40), -1)
    
    # Draw text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = x1 + 20
    text_y = y1 + ((y2 - y1) + th) // 2

    cv2.putText(overlay, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
    return overlay

def mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks for betting interface"""
    global selected_bet, selected_amount, bet_confirmed, dashboard_visible
    global notification_visible, confirm_coords, new_bet_coords, dashboard_cache_valid
    
    if event == cv2.EVENT_LBUTTONDOWN:        
        # Toggle dashboard button
        x1, y1, x2, y2 = toggle_button_coords
        if x1 <= x <= x2 and y1 <= y <= y2:
            dashboard_visible = not dashboard_visible
            dashboard_cache_valid = False  # Invalidate cache
            print(f"Dashboard toggled: {'ON' if dashboard_visible else 'OFF'}")
            return
        
        if not dashboard_visible:
            return
            
        # Betting option clicks
        for idx, (ox1, oy1, ox2, oy2) in enumerate(option_coords):
            if ox1 <= x <= ox2 and oy1 <= y <= oy2:
                if idx < len(current_betting_options):
                    player, action = current_betting_options[idx]
                    selected_bet = f"{player} {action}"
                    dashboard_cache_valid = False  # Invalidate cache
                    print(f"Selected bet: {selected_bet}")
                    break
        
        # Amount option clicks
        for idx, (ax1, ay1, ax2, ay2) in enumerate(amount_coords):
            if ax1 <= x <= ax2 and ay1 <= y <= ay2:
                selected_amount = bet_amount_options[idx]
                dashboard_cache_valid = False  # Invalidate cache
                print(f"Selected amount: {selected_amount}")
                break
        
        # Confirm button
        cx1, cy1, cx2, cy2 = confirm_coords
        if cx1 <= x <= cx2 and cy1 <= y <= cy2 and selected_bet and selected_amount:
            bet_confirmed = True
            dashboard_cache_valid = False  # Invalidate cache
            print(f"Bet confirmed: {selected_bet} for {selected_amount}")
        
        # New bet button
        nx1, ny1, nx2, ny2 = new_bet_coords
        if nx1 <= x <= nx2 and ny1 <= y <= ny2:
            selected_bet = None
            selected_amount = None
            bet_confirmed = False
            dashboard_cache_valid = False  # Invalidate cache
            print("New bet started - selections cleared")

def create_dashboard_overlay_optimized(frame_shape):
    """Create dashboard overlay - HEAVILY OPTIMIZED"""
    global current_betting_options, confirm_coords, new_bet_coords
    
    h, w = frame_shape[:2]
    sidebar_w = 600
    
    # Create overlay
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    
    if not dashboard_visible:
        return overlay
    
    # Dashboard background - single rectangle
    cv2.rectangle(overlay, (w - sidebar_w, 0), (w, h), (30, 30, 30), -1)
    
    y_cursor = 80
    section_spacing = 40
    box_height = 45
    inter_item_spacing = 12

    # Simplified box drawing function
    def draw_box(img, x1, y1, x2, y2, color):
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (60, 60, 60), 1)

    # Selected Bet section - simplified
    cv2.putText(overlay, "Selected Bet:", (w - sidebar_w + 20, y_cursor),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    y_cursor += 25
    draw_box(overlay, w - sidebar_w + 15, y_cursor, w - 20, y_cursor + box_height, (50, 50, 50))
    bet_display = (selected_bet[:35] + "...") if selected_bet and len(selected_bet) > 35 else (selected_bet or "None")
    cv2.putText(overlay, bet_display, (w - sidebar_w + 25, y_cursor + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    y_cursor += box_height + section_spacing

    # Amount section - simplified
    cv2.putText(overlay, "Amount:", (w - sidebar_w + 20, y_cursor),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    y_cursor += 25
    draw_box(overlay, w - sidebar_w + 15, y_cursor, w - 20, y_cursor + box_height, (50, 50, 50))
    cv2.putText(overlay, selected_amount if selected_amount else "None", (w - sidebar_w + 25, y_cursor + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    y_cursor += box_height + section_spacing

    # Betting options - pre-computed
    option_coords.clear()
    current_betting_options = [
        ("Server", "to win next point"),
        ("Receiver", "to win next point"),
        ("Server", "to serve ace"),
        ("Receiver", "to make winner"),
        ("Match", "to go to deuce")
    ]
    
    # Batch draw betting options
    for i, (player, action) in enumerate(current_betting_options):
        color = (0, 120, 255) if selected_bet and selected_bet == f"{player} {action}" else (80, 80, 80)
        draw_box(overlay, w - sidebar_w + 15, y_cursor, w - 20, y_cursor + box_height, color)
        cv2.putText(overlay, f"{i+1}. {player} {action}", (w - sidebar_w + 25, y_cursor + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        option_coords.append((w - sidebar_w + 15, y_cursor, w - 20, y_cursor + box_height))
        y_cursor += box_height + inter_item_spacing

    y_cursor += section_spacing // 2

    # Amount options - batch draw
    amount_coords.clear()
    for i, amt in enumerate(bet_amount_options):
        color = (0, 120, 255) if selected_amount == amt else (80, 80, 80)
        draw_box(overlay, w - sidebar_w + 15, y_cursor, w - 20, y_cursor + box_height, color)
        cv2.putText(overlay, f"{i+1}. {amt}", (w - sidebar_w + 25, y_cursor + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        amount_coords.append((w - sidebar_w + 15, y_cursor, w - 20, y_cursor + box_height))
        y_cursor += box_height + inter_item_spacing

    # Buttons
    confirm_y = y_cursor + 20
    confirm_coords = (w - sidebar_w + 50, confirm_y, w - sidebar_w + 250, confirm_y + box_height)
    new_bet_coords = (w - sidebar_w + 270, confirm_y, w - sidebar_w + 470, confirm_y + box_height)

    # Draw buttons
    confirm_color = (0, 255, 0) if not bet_confirmed else (0, 150, 0)
    draw_box(overlay, confirm_coords[0], confirm_coords[1], confirm_coords[2], confirm_coords[3], confirm_color)
    cv2.putText(overlay, "CONFIRM", (confirm_coords[0]+30, confirm_coords[1]+28), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    draw_box(overlay, new_bet_coords[0], new_bet_coords[1], new_bet_coords[2], new_bet_coords[3], (255, 140, 0))
    cv2.putText(overlay, "NEW BET", (new_bet_coords[0]+30, new_bet_coords[1]+28), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return overlay

def apply_dashboard_to_frame(frame):
    """Apply cached dashboard to frame - OPTIMIZED"""
    global toggle_button_coords, dashboard_overlay_cache, dashboard_cache_valid
    global dashboard_cache_frame_size, last_cache_update
    
    h, w = frame.shape[:2]
    sidebar_w = 600
    
    # Update cache only when needed
    if (not dashboard_cache_valid or 
        dashboard_cache_frame_size != (h, w) or
        dashboard_overlay_cache is None):
        
        dashboard_overlay_cache = create_dashboard_overlay_optimized(frame.shape)
        dashboard_cache_valid = True
        dashboard_cache_frame_size = (h, w)
        last_cache_update = time.time()
    
    # Apply cached dashboard
    if dashboard_visible and dashboard_overlay_cache is not None:
        # Optimized blending - only blend non-zero areas
        mask = np.any(dashboard_overlay_cache > 0, axis=2)
        if np.any(mask):  # Only blend if there's something to blend
            frame[mask] = cv2.addWeighted(frame[mask], 0.2, dashboard_overlay_cache[mask], 0.8, 0)
    
    # Draw toggle button (always visible) - simplified
    button_h = 60
    toggle_button_coords = (w - sidebar_w, 0, w, button_h)
    cv2.rectangle(frame, (toggle_button_coords[0], toggle_button_coords[1]),
                  (toggle_button_coords[2], toggle_button_coords[3]), (0, 140, 0), -1)
    
    # Simplified button text
    symbol = "v" if dashboard_visible else "^"
    cv2.putText(frame, "BETTING", (toggle_button_coords[0]+20, toggle_button_coords[1]+35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, symbol, (toggle_button_coords[2]-30, toggle_button_coords[1]+35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Status indicator - only when needed
    if bet_confirmed or (selected_bet and selected_amount):
        if bet_confirmed:
            status_text = f"BET: {selected_bet[:30]} ({selected_amount})"
            color = (0, 255, 0)
        else:
            status_text = f"READY: {selected_bet[:25]} ({selected_amount})"
            color = (0, 255, 255)
        cv2.putText(frame, status_text, (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return frame

def draw_player_info_optimized(frame, tracked_objects, frame_count):
    """Draw player information with reduced frequency updates"""
    height, width = frame.shape[:2]
    
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = obj
        obj_id = int(obj_id)
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
        # Ensure coordinates are within bounds
        x1 = max(0, min(x1, width-1))
        y1 = max(0, min(y1, height-1))
        x2 = max(0, min(x2, width-1))
        y2 = max(0, min(y2, height-1))
        
        # Player-specific colors
        player_color = (0, 255, 0) if obj_id % 2 == 0 else (255, 0, 255)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), player_color, 2)
        
        # Get player info
        name = player_names.get(obj_id, f"Player_{obj_id}")
        speed = player_speeds.get(obj_id, 0.0)
        distance = player_distances.get(obj_id, 0.0)
        
        # Simplified info box - smaller and more efficient
        info_height = 60
        info_width = min(85, x2 - x1 + 50)
        
        # Background box
        cv2.rectangle(frame, (x1, y1-info_height), (x1+info_width, y1), (0, 0, 0), -1)
        cv2.rectangle(frame, (x1, y1-info_height), (x1+info_width, y1), player_color, 1)
        
        # Simplified text - fewer lines, smaller font
        cv2.putText(frame, name[:8], (x1+3, y1-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"{speed:.0f}km/h", (x1+3, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(frame, f"{distance:.0f}m", (x1+3, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # Center point
        cx, cy = (x1+x2)//2, (y1+y2)//2
        cv2.circle(frame, (cx, cy), 3, player_color, -1)

def process_video(input_path, output_path=None):
    """OPTIMIZED main video processing function"""
    global selected_bet, selected_amount, bet_confirmed
    global notification_text, notification_visible, notification_start_time, last_notification_time
    global player_names, player_positions, player_speeds, player_distances
    global dashboard_overlay_cache, notification_overlay_cache
    
    if not os.path.exists(input_path):
        print(f"Input video not found: {input_path}")
        return

    # Video setup
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Could not open input video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    ret, frame = cap.read()
    if not ret:
        print("Could not read first frame.")
        return

    height, width = frame.shape[:2]
    print(f"Input video: {width}x{height}, {fps:.2f} FPS, {total_frames} frames")

    # Output setup
    out = None
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # OPTIMIZED: Better codec settings for performance
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Output will be saved to: {output_path}")

    # OPTIMIZED: Tracker with better settings for performance
    tracker = Sort(max_age=15, min_hits=2, iou_threshold=0.3)
    
    # Processing variables
    frame_count = 0
    last_detections = np.empty((0, 5))
    
    # Preview setup
    show_preview = output_path is None
    if show_preview:
        cv2.namedWindow("Tennis Betting Analysis", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Tennis Betting Analysis", 1280, 720)
        cv2.setMouseCallback("Tennis Betting Analysis", mouse_callback)
        
    paused = False
    start_time = time.time()
    last_notification_time = start_time
    
    print("OPTIMIZED processing started...")
    if show_preview:
        print("Controls: SPACE = Pause/Resume, Q = Quit, R = Reset bet")

    # OPTIMIZED main processing loop
    while ret:
        current_time = time.time()
        detections = []

        # OPTIMIZED: Run detection less frequently
        if frame_count % DETECTION_INTERVAL == 0:
            results = model(frame, verbose=False)
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                for box, cls, conf in zip(results[0].boxes.xyxy.cpu().numpy(),
                                        results[0].boxes.cls.cpu().numpy(),
                                        results[0].boxes.conf.cpu().numpy()):
                    if int(cls) == 0 and conf > 0.4:  # Slightly higher confidence threshold
                        x1, y1, x2, y2 = box
                        
                        # OPTIMIZED: Simplified filtering
                        box_area = (x2 - x1) * (y2 - y1)
                        min_area = (width * height) * 0.003  # Slightly larger minimum
                        max_area = (width * height) * 0.25   # Slightly smaller maximum
                        
                        if min_area < box_area < max_area:
                            detections.append([x1, y1, x2, y2, conf])

            if len(detections) > 0:
                last_detections = np.array(detections)
        
        # Update tracker
        tracked_objects = tracker.update(last_detections if len(last_detections) > 0 else np.empty((0, 5)))

        # OPTIMIZED: Update player tracking less frequently
        if frame_count % SPEED_CALC_INTERVAL == 0:
            current_positions = {}
            for obj in tracked_objects:
                x1, y1, x2, y2, obj_id = obj
                obj_id = int(obj_id)
                
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                current_positions[obj_id] = (center_x, center_y)
                
                # Assign name only when needed
                if obj_id not in player_names:
                    assign_player_name(obj_id)
                
                # Initialize tracking data
                if obj_id not in player_distances:
                    player_distances[obj_id] = 0.0
                    player_speeds[obj_id] = 0.0
                
                # OPTIMIZED: Calculate speed less frequently with smoothing
                if obj_id in player_positions:
                    prev_x, prev_y = player_positions[obj_id]
                    distance_pixels = np.sqrt((center_x - prev_x)**2 + (center_y - prev_y)**2)
                    distance_meters = distance_pixels * PIXELS_TO_METERS
                    
                    player_distances[obj_id] += distance_meters
                    
                    # Speed calculation with more smoothing
                    time_per_frame = SPEED_CALC_INTERVAL / fps
                    speed_ms = distance_meters / time_per_frame
                    speed_kmh = min(speed_ms * 3.6, 40)  # Cap at reasonable speed
                    
                    # Heavy smoothing for stable display
                    player_speeds[obj_id] = player_speeds[obj_id] * 0.8 + speed_kmh * 0.2

            player_positions.update(current_positions)

        # OPTIMIZED: Draw player info (every frame but optimized)
        draw_player_info_optimized(frame, tracked_objects, frame_count)

        # OPTIMIZED: Apply dashboard (cached)
        frame = apply_dashboard_to_frame(frame)
        
        # OPTIMIZED: Handle notifications less frequently
        if frame_count % NOTIFICATION_UPDATE_INTERVAL == 0:
            if current_time - last_notification_time >= notification_interval:
                notification_text = generate_notification()
                notification_visible = True
                notification_start_time = current_time
                last_notification_time = current_time
                # Pre-create notification overlay
                notification_overlay_cache = create_notification_overlay(frame.shape, notification_text)

            # Auto-hide notification
            if notification_visible and current_time - notification_start_time >= notification_duration:
                notification_visible = False
                notification_overlay_cache = None

        # Draw notification if visible (using cache)
        if notification_visible and notification_overlay_cache is not None:
            mask = np.any(notification_overlay_cache > 0, axis=2)
            if np.any(mask):
                frame[mask] = cv2.addWeighted(frame[mask], 0.3, notification_overlay_cache[mask], 0.7, 0)

        # Write frame
        if out:
            out.write(frame)

        # Handle preview
        if show_preview:
            if paused:
                # Simplified pause overlay
                cv2.rectangle(frame, (width//2-100, height//2-30), (width//2+100, height//2+30), (0, 0, 0), -1)
                cv2.putText(frame, "PAUSED", (width//2-50, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            cv2.imshow("Tennis Betting Analysis", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord(' '):
                paused = not paused
            elif key == ord('r'):
                selected_bet = None
                selected_amount = None
                bet_confirmed = False
                dashboard_cache_valid = False
                print("Bet selections reset")
                    
            if paused:
                continue

        # OPTIMIZED: Progress reporting less frequently
        if frame_count % 150 == 0:
            progress = (frame_count / total_frames) * 100
            elapsed = time.time() - start_time
            fps_actual = frame_count / elapsed if elapsed > 0 else 0
            print(f"Progress: {progress:.1f}% - Processing at {fps_actual:.1f} FPS")

        frame_count += 1
        ret, frame = cap.read()

    # Cleanup
    cap.release()
    if out:
        out.release()
        processing_time = time.time() - start_time
        print(f"\nOPTIMIZED processing completed!")
        print(f"Total time: {processing_time:.1f} seconds")
        print(f"Average processing FPS: {frame_count/processing_time:.1f}")
        if output_path:
            print(f"Output saved to: {output_path}")
        
    if show_preview:
        cv2.destroyAllWindows()
    
    # Final statistics
    print("\n=== FINAL PLAYER STATISTICS ===")
    for player_id, name in player_names.items():
        distance = player_distances.get(player_id, 0.0)
        speed = player_speeds.get(player_id, 0.0)
        print(f"{name} (ID: {player_id}): Distance: {distance:.1f}m, Speed: {speed:.1f} km/h")
    
    if bet_confirmed:
        print(f"\nFinal Bet: {selected_bet} - Amount: {selected_amount}")

if __name__ == "__main__":
    input_video = "test.mp4"
    
    # For preview mode (interactive dashboard)
    process_video(input_video)
    
    # For output video mode - MUCH SMOOTHER NOW
    # process_video(input_video, "output/tennis_betting_analysis_optimized.mp4")