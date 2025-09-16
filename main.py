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
confirm_coords = ()
new_bet_coords = ""
dashboard_visible = True
toggle_button_coords = ()

# Enhanced player names list
player_names_list = [
    "Djokovic", "Federer", "Nadal", "Murray", "Tsitsipas",
    "Zverez", "Medvedev", "Thiem", "Berrettini", "Rublev",
    "Alcaraz", "Sinner", "Ruud", "Fritz", "Hurkacz",
    "Norrie", "Auger-Aliassime", "Shapovalov", "Kyrgios", "De Minaur"
]

bet_amount_options = ["$10", "$20", "$50", "$100"]
PIXELS_TO_METERS = 0.05

# Optimization: Pre-create dashboard elements to avoid recreating every frame
dashboard_elements_cache = None
dashboard_cache_frame_size = None

def assign_player_name(obj_id):
    """Assign a unique name to a player based on their ID"""
    if obj_id not in player_names:
        used_names = set(player_names.values())
        available_names = [name for name in player_names_list if name not in used_names]
        
        if available_names:
            if len(player_names) == 0:
                assigned_name = available_names[0]
            elif len(player_names) == 1:
                assigned_name = available_names[1] if len(available_names) > 1 else available_names[0]
            else:
                assigned_name = available_names[0]
            
            player_names[obj_id] = assigned_name
            name_assignment_order.append(obj_id)
        else:
            assigned_name = random.choice(player_names_list)
            player_names[obj_id] = assigned_name

    return player_names[obj_id]

def generate_notification():
    """Generate random tennis notification"""
    events = [
        f"Ace probability: {random.randint(15,35)}%",
        f"Winner shot: {random.randint(40,80)}%",
        f"Double fault risk: {random.randint(5,25)}%",
        f"Break point chance: {random.randint(30,70)}%",
        f"Rally length 10+: {random.randint(20,60)}%"
    ]
    return random.choice(events)

def draw_notification(frame, text):
    """Draw notification overlay"""
    x1, y1, x2, y2 = notification_coords
    x2 = x2 + 300

    # Draw background
    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 40), -1)
    
    # Draw text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = x1 + 20
    text_y = y1 + ((y2 - y1) + th) // 2

    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
    return frame

def mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks for betting interface"""
    global selected_bet, selected_amount, bet_confirmed, dashboard_visible
    global notification_visible
    
    if event == cv2.EVENT_LBUTTONDOWN:        
        # Toggle dashboard button
        x1, y1, x2, y2 = toggle_button_coords
        if x1 <= x <= x2 and y1 <= y <= y2:
            dashboard_visible = not dashboard_visible
            return
        
        if not dashboard_visible:
            return
            
        # Betting option clicks
        for idx, (ox1, oy1, ox2, oy2) in enumerate(option_coords):
            if ox1 <= x <= ox2 and oy1 <= y <= oy2:
                if 'current_betting_options' in globals():
                    if idx < len(current_betting_options):
                        player, action = current_betting_options[idx]
                        selected_bet = f"{player} {action}"
        
        # Amount option clicks
        for idx, (ax1, ay1, ax2, ay2) in enumerate(amount_coords):
            if ax1 <= x <= ax2 and ay1 <= y <= ay2:
                selected_amount = bet_amount_options[idx]
        
        # Confirm button
        cx1, cy1, cx2, cy2 = confirm_coords
        if cx1 <= x <= cx2 and cy1 <= y <= cy2 and selected_bet and selected_amount:
            bet_confirmed = True
        
        # New bet button
        nx1, ny1, nx2, ny2 = new_bet_coords
        if nx1 <= x <= nx2 and ny1 <= y <= ny2:
            selected_bet = None
            selected_amount = None
            bet_confirmed = False

def create_dashboard_overlay(frame_shape):
    """Create dashboard overlay - called only when needed"""
    global current_betting_options
    
    h, w = frame_shape[:2]
    sidebar_w = 600
    
    # Create transparent overlay
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    
    if not dashboard_visible:
        return overlay
    
    # Dashboard background
    cv2.rectangle(overlay, (w - sidebar_w, 0), (w, h), (30, 30, 30), -1)
    
    y_cursor = 80  # Start after toggle button
    section_spacing = 40
    box_height = 45
    inter_item_spacing = 12

    def draw_rounded_box(img, top_left, bottom_right, color, radius=8, thickness=-1):
        x1, y1 = top_left
        x2, y2 = bottom_right
        cv2.rectangle(img, (x1+radius, y1), (x2-radius, y2), color, thickness)
        cv2.rectangle(img, (x1, y1+radius), (x2, y2-radius), color, thickness)
        cv2.circle(img, (x1+radius, y1+radius), radius, color, thickness)
        cv2.circle(img, (x2-radius, y1+radius), radius, color, thickness)
        cv2.circle(img, (x1+radius, y2-radius), radius, color, thickness)
        cv2.circle(img, (x2-radius, y2-radius), radius, color, thickness)

    # Selected Bet section
    cv2.putText(overlay, "Selected Bet:", (w - sidebar_w + 20, y_cursor),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    y_cursor += 25
    draw_rounded_box(overlay, (w - sidebar_w + 15, y_cursor), (w - 20, y_cursor + box_height), (50, 50, 50))
    bet_display = (selected_bet[:40] + "...") if selected_bet and len(selected_bet) > 40 else (selected_bet or "None")
    cv2.putText(overlay, bet_display, (w - sidebar_w + 25, y_cursor + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    y_cursor += box_height + section_spacing

    # Amount section
    cv2.putText(overlay, "Amount:", (w - sidebar_w + 20, y_cursor),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    y_cursor += 25
    draw_rounded_box(overlay, (w - sidebar_w + 15, y_cursor), (w - 20, y_cursor + box_height), (50, 50, 50))
    cv2.putText(overlay, selected_amount if selected_amount else "None", (w - sidebar_w + 25, y_cursor + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    y_cursor += box_height + section_spacing

    # Betting options
    option_coords.clear()
    current_betting_options = [
        ("Server", "to win next point"),
        ("Receiver", "to win next point"),
        ("Server", "to serve ace"),
        ("Receiver", "to make winner"),
        ("Match", "to go to deuce")
    ]
    
    for i, (player, action) in enumerate(current_betting_options):
        color = (80, 80, 80)
        bet_text = f"{player} {action}"
        
        if selected_bet and selected_bet == bet_text:
            color = (0, 120, 255)
            
        draw_rounded_box(overlay, (w - sidebar_w + 15, y_cursor), (w - 20, y_cursor + box_height), color)
        display_text = f"{i+1}. {bet_text}"
        cv2.putText(overlay, display_text, (w - sidebar_w + 25, y_cursor + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        option_coords.append((w - sidebar_w + 15, y_cursor, w - 20, y_cursor + box_height))
        y_cursor += box_height + inter_item_spacing

    y_cursor += section_spacing // 2

    # Amount options
    amount_coords.clear()
    for i, amt in enumerate(bet_amount_options):
        color = (80, 80, 80)
        if selected_amount == amt:
            color = (0, 120, 255)
        draw_rounded_box(overlay, (w - sidebar_w + 15, y_cursor), (w - 20, y_cursor + box_height), color)
        cv2.putText(overlay, f"{i+1}. {amt}", (w - sidebar_w + 25, y_cursor + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        amount_coords.append((w - sidebar_w + 15, y_cursor, w - 20, y_cursor + box_height))
        y_cursor += box_height + inter_item_spacing

    # Confirm & New Bet Buttons
    confirm_y = y_cursor + 20
    confirm_coords = (w - sidebar_w + 50, confirm_y, w - sidebar_w + 250, confirm_y + box_height)
    new_bet_coords = (w - sidebar_w + 270, confirm_y, w - sidebar_w + 470, confirm_y + box_height)

    draw_rounded_box(overlay, confirm_coords[:2], confirm_coords[2:],
                     (0, 255, 0) if not bet_confirmed else (0, 150, 0))
    cv2.putText(overlay, "CONFIRM", (confirm_coords[0]+30, confirm_coords[1]+28), 
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)

    draw_rounded_box(overlay, new_bet_coords[:2], new_bet_coords[2:], (255, 140, 0))
    cv2.putText(overlay, "NEW BET", (new_bet_coords[0]+30, new_bet_coords[1]+28), 
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)
    
    return overlay

def draw_betting_dashboard(frame):
    """Apply betting dashboard to frame"""
    global toggle_button_coords, dashboard_elements_cache, dashboard_cache_frame_size
    
    h, w = frame.shape[:2]
    sidebar_w = 600
    
    # Cache dashboard overlay to avoid recreating every frame
    if (dashboard_elements_cache is None or 
        dashboard_cache_frame_size != (h, w) or 
        frame.shape[:2] != dashboard_cache_frame_size):
        
        dashboard_elements_cache = create_dashboard_overlay(frame.shape)
        dashboard_cache_frame_size = (h, w)
    
    # Apply dashboard overlay
    if dashboard_visible:
        # Use optimized blending
        mask = dashboard_elements_cache > 0
        frame[mask] = cv2.addWeighted(frame, 0.15, dashboard_elements_cache, 0.85, 0)[mask]
    
    # Draw toggle button (always visible)
    button_h = 60
    toggle_button_coords = (w - sidebar_w, 0, w, button_h)
    cv2.rectangle(frame, (toggle_button_coords[0], toggle_button_coords[1]),
                  (toggle_button_coords[2], toggle_button_coords[3]), (0, 140, 0), -1)
    symbol = "▼" if dashboard_visible else "▲"
    cv2.putText(frame, "TENNIS BETTING", (toggle_button_coords[0]+15, toggle_button_coords[1]+35),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, symbol, (toggle_button_coords[2]-45, toggle_button_coords[1]+35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return frame

def process_video(input_path, output_path=None):
    """Main video processing function - optimized for smooth output"""
    global selected_bet, selected_amount, bet_confirmed
    global notification_text, notification_visible, notification_start_time, last_notification_time
    global player_names, player_positions, player_speeds, player_distances
    global dashboard_elements_cache
    
    if not os.path.exists(input_path):
        print(f"Input video not found: {input_path}")
        return

    # Video capture setup
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Could not open input video.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    ret, frame = cap.read()
    if not ret:
        print("Could not read first frame.")
        return

    height, width = frame.shape[:2]
    print(f"Input video: {width}x{height}, {fps:.2f} FPS, {total_frames} frames")

    # Output writer setup - CRITICAL for smooth video
    out = None
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Use high-quality codec settings
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print("Could not create output video writer.")
            return
        print(f"Output will be saved to: {output_path}")

    # Tracker setup - optimized settings
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    
    # Processing variables
    frame_count = 0
    detect_every = 3  # Detect every 3rd frame for balance of performance and accuracy
    last_detections = np.empty((0, 5))
    
    # Preview setup
    show_preview = output_path is None
    if show_preview:
        cv2.namedWindow("Tennis Betting Analysis", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Tennis Betting Analysis", mouse_callback)
    
    paused = False
    start_time = time.time()
    last_notification_time = start_time
    
    print("Processing started - optimized for smooth output...")
    if show_preview:
        print("Controls: SPACE = Pause/Resume, Q = Quit")

    # Main processing loop - EVERY FRAME IS PROCESSED AND SAVED
    while ret:
        current_time = time.time()
        detections = []

        # Run detection periodically for performance
        if frame_count % detect_every == 0:
            results = model(frame, verbose=False)
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                for box, cls, conf in zip(results[0].boxes.xyxy.cpu().numpy(),
                                        results[0].boxes.cls.cpu().numpy(),
                                        results[0].boxes.conf.cpu().numpy()):
                    if int(cls) == 0 and conf > 0.35:  # person class with reasonable confidence
                        x1, y1, x2, y2 = box
                        
                        # Quick filtering for tennis players
                        box_width = x2 - x1
                        box_height = y2 - y1
                        box_area = box_width * box_height
                        
                        # Size filtering
                        min_area = (width * height) * 0.002
                        max_area = (width * height) * 0.3
                        
                        # Position filtering
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        margin_x = width * 0.05
                        margin_y = height * 0.1
                        
                        # Aspect ratio filtering
                        aspect_ratio = box_height / box_width if box_width > 0 else 0
                        
                        if (min_area < box_area < max_area and 
                            margin_x < center_x < width - margin_x and
                            margin_y < center_y < height - margin_y and
                            aspect_ratio > 1.2):
                            detections.append([x1, y1, x2, y2, conf])

            if len(detections) > 0:
                last_detections = np.array(detections)
            # Keep using last detections if no new ones found
        
        # Update tracker with current detections
        tracked_objects = tracker.update(last_detections if len(last_detections) > 0 else np.empty((0, 5)))

        # Update player tracking
        current_positions = {}
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = obj
            obj_id = int(obj_id)
            
            # Calculate center position
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            current_positions[obj_id] = (center_x, center_y)
            
            # Assign name
            assign_player_name(obj_id)
            
            # Initialize tracking data
            if obj_id not in player_distances:
                player_distances[obj_id] = 0.0
                player_speeds[obj_id] = 0.0
            
            # Calculate speed and distance (every frame for smooth tracking)
            if obj_id in player_positions:
                prev_x, prev_y = player_positions[obj_id]
                distance_pixels = np.sqrt((center_x - prev_x)**2 + (center_y - prev_y)**2)
                distance_meters = distance_pixels * PIXELS_TO_METERS
                
                # Update total distance
                player_distances[obj_id] += distance_meters
                
                # Calculate speed
                time_per_frame = 1.0 / fps
                speed_ms = distance_meters / time_per_frame
                speed_kmh = speed_ms * 3.6
                # Apply smoothing to reduce jitter
                current_speed = max(0, min(speed_kmh, 50))
                player_speeds[obj_id] = player_speeds[obj_id] * 0.7 + current_speed * 0.3

        # Update positions
        player_positions.update(current_positions)

        # Draw player tracking information
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = obj
            obj_id = int(obj_id)
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Ensure coordinates are within bounds
            x1 = max(0, min(x1, width-1))
            y1 = max(0, min(y1, height-1))
            x2 = max(0, min(x2, width-1))
            y2 = max(0, min(y2, height-1))
            
            cx, cy = (x1+x2)//2, (y1+y2)//2

            # Player-specific colors
            player_color = (0, 255, 0) if obj_id % 2 == 0 else (255, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), player_color, 2)
            
            # Get player info
            name = player_names.get(obj_id, f"Player_{obj_id}")
            speed = player_speeds.get(obj_id, 0.0)
            distance = player_distances.get(obj_id, 0.0)
            
            # Draw info box
            info_height = 70
            cv2.rectangle(frame, (x1, y1-info_height), (x2+90, y1), (0, 0, 0), -1)
            cv2.rectangle(frame, (x1, y1-info_height), (x2+90, y1), player_color, 2)
            
            # Draw text info
            cv2.putText(frame, name, (x1+5, y1-50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"{speed:.1f} km/h", (x1+5, y1-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(frame, f"{distance:.1f}m", (x1+5, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Draw center point
            cv2.circle(frame, (cx, cy), 5, player_color, -1)

        # Add betting dashboard
        frame = draw_betting_dashboard(frame)
        
        # Handle notifications (time-based for consistent timing)
        if current_time - last_notification_time >= notification_interval:
            notification_text = generate_notification()
            notification_visible = True
            notification_start_time = current_time
            last_notification_time = current_time

        # Auto-hide notification after duration
        if notification_visible and current_time - notification_start_time >= notification_duration:
            notification_visible = False

        # Draw notification if visible
        if notification_visible:
            frame = draw_notification(frame, notification_text)

        # CRITICAL: Write every frame to output for smooth video
        if out:
            out.write(frame)

        # Handle preview mode
        if show_preview:
            if paused:
                # Draw pause overlay
                overlay = frame.copy()
                cv2.rectangle(overlay, (width//2-160, height//2-50), (width//2+160, height//2+50), (0, 0, 0), -1)
                cv2.rectangle(overlay, (width//2-160, height//2-50), (width//2+160, height//2+50), (0, 255, 255), 3)
                cv2.putText(overlay, "PAUSED", (width//2-55, height//2-10), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 255), 2)
                cv2.putText(overlay, "Press SPACE to resume", (width//2-110, height//2+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                frame = overlay
            
            cv2.imshow("Tennis Betting Analysis", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord(' '):
                paused = not paused
                if paused:
                    print("PAUSED")
                else:
                    print("RESUMED")
                    
            # Skip to next frame if paused
            if paused:
                continue

        # Progress reporting
        if frame_count % 120 == 0:
            progress = (frame_count / total_frames) * 100
            elapsed = time.time() - start_time
            fps_actual = frame_count / elapsed if elapsed > 0 else 0
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) - Processing at {fps_actual:.1f} FPS")

        frame_count += 1
        ret, frame = cap.read()

    # Cleanup
    cap.release()
    if out:
        out.release()
        processing_time = time.time() - start_time
        print(f"\nProcessing completed successfully!")
        print(f"Total time: {processing_time:.1f} seconds")
        print(f"Average processing FPS: {frame_count/processing_time:.1f}")
        print(f"Output saved to: {output_path}")
        
    if show_preview:
        cv2.destroyAllWindows()
    
    # Print final statistics
    print("\n=== FINAL PLAYER STATISTICS ===")
    for player_id, name in player_names.items():
        distance = player_distances.get(player_id, 0.0)
        speed = player_speeds.get(player_id, 0.0)
        print(f"{name} (ID: {player_id}): Distance: {distance:.2f}m, Speed: {speed:.1f} km/h")
    
    if bet_confirmed:
        print(f"\nFinal Bet: {selected_bet} - Amount: {selected_amount}")

if __name__ == "__main__":
    input_video = "test.mp4"
    process_video(input_video)