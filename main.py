import cv2
import math
from ultralytics import YOLO
from sort.sort import Sort
import os
import numpy as np
import random
import time

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # yolov8n = fastest

# Global variables for tracking selections
selected_bet = None
selected_amount = None
bet_confirmed = False

# Player tracking variables
player_names = {}  # Dictionary to store player names by ID
player_positions = {}  # Dictionary to store previous positions for speed calculation
player_speeds = {}  # Dictionary to store current speeds
player_distances = {}  # Dictionary to store total distance covered by each player
name_assignment_order = []  # Track the order of name assignments

# Notification global variables
notification_text = ""
notification_visible = False
notification_start_time = 0
notification_interval = 13  # seconds between notifications
notification_duration = 2.5  # seconds before auto-hide
notification_coords = (20, 100, 320, 180)  # x1, y1, x2, y2
notification_closed = False

# Coordinates for clickable areas
option_coords = []
amount_coords = []
confirm_coords = ()
new_bet_coords = ""
dashboard_visible = True
toggle_button_coords = ()  # x1, y1, x2, y2

# Enhanced player names list with more variety
player_names_list = [
    "Djokovic", "Federer", "Nadal", "Murray", "Tsitsipas",
    "Zverev", "Medvedev", "Thiem", "Berrettini", "Rublev",
    "Alcaraz", "Sinner", "Ruud", "Fritz", "Hurkacz",
    "Norrie", "Auger-Aliassime", "Shapovalov", "Kyrgios", "De Minaur"
]

# Tennis-specific betting options - Template structure
betting_options_template = [
    ("PLAYER1", "to win next point"),
    ("PLAYER2", "to win next point"), 
    ("PLAYER1", "to serve ace"),
    ("PLAYER2", "to make winner"),
    ("Match", "to go to deuce")
]

bet_amount_options = ["$10", "$20", "$50", "$100"]

# Calibration factor: pixels to meters conversion
# This should be calibrated based on court dimensions
# Standard tennis court is 23.77m long, adjust this based on your video
PIXELS_TO_METERS = 0.05  # Adjust this value based on your video scale

# --- Enhanced name assignment function ---
def assign_player_name(obj_id):
    """Assign a unique name to a player based on their ID"""
    if obj_id not in player_names:
        # Get unused names
        used_names = set(player_names.values())
        available_names = [name for name in player_names_list if name not in used_names]
        
        if available_names:
            # For tennis, typically assign names based on court position or detection order
            if len(player_names) == 0:
                # First player detected
                assigned_name = available_names[0]
            elif len(player_names) == 1:
                # Second player detected - pick a different name
                assigned_name = available_names[1] if len(available_names) > 1 else available_names[0]
            else:
                # Additional players (rare in tennis)
                assigned_name = available_names[0]
            
            player_names[obj_id] = assigned_name
            name_assignment_order.append(obj_id)
            print(f"Assigned name '{assigned_name}' to Player ID {obj_id}")
        else:
            # Reuse names if we run out
            assigned_name = random.choice(player_names_list)
            player_names[obj_id] = assigned_name
            print(f"Reusing name '{assigned_name}' for Player ID {obj_id}")

    return player_names[obj_id]

# --- Generate random notification ---
def generate_notification():
    events = [
        f"Ace probability: {random.randint(15,35)}%",
        f"Winner shot: {random.randint(40,80)}%",
        f"Double fault risk: {random.randint(5,25)}%",
        f"Break point chance: {random.randint(30,70)}%",
        f"Rally length 10+: {random.randint(20,60)}%"
    ]
    return random.choice(events)

# --- Draw notification on frame ---
def draw_notification(frame, text):
    x1, y1, x2, y2 = notification_coords
    x2 = x2 + 300

    # Opaque rectangle (dark background)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 40), -1)

    # Format probability nicely
    if "probability" in text.lower() or ":" in text:
        pass  # Keep original format
    
    # Put text (white, vertically centered)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2

    # Get text size
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = x1 + 20
    text_y = y1 + ((y2 - y1) + th) // 2

    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
    
    return frame

# --- Mouse callback ---
def mouse_callback(event, x, y, flags, param):
    global selected_bet, selected_amount, bet_confirmed, dashboard_visible
    global notification_visible, notification_closed
    
    if event == cv2.EVENT_LBUTTONDOWN:        
        # Toggle dashboard button
        x1, y1, x2, y2 = toggle_button_coords
        if x1 <= x <= x2 and y1 <= y <= y2:
            dashboard_visible = not dashboard_visible
            return
        
        if not dashboard_visible:
            return
            
        # Betting option clicks - Using Server/Receiver terminology
        for idx, (ox1, oy1, ox2, oy2) in enumerate(option_coords):
            if ox1 <= x <= ox2 and oy1 <= y <= oy2:
                # Get the actual betting options list that was used to generate the display
                if 'current_betting_options' in globals():
                    if idx < len(current_betting_options):
                        player, action = current_betting_options[idx]
                        selected_bet = f"{player} {action}"
                        print(f"Selected bet: {selected_bet}")
        
        # Amount option clicks
        for idx, (ax1, ay1, ax2, ay2) in enumerate(amount_coords):
            if ax1 <= x <= ax2 and ay1 <= y <= ay2:
                selected_amount = bet_amount_options[idx]
        
        # Confirm button
        cx1, cy1, cx2, cy2 = confirm_coords
        if cx1 <= x <= cx2 and cy1 <= y <= cy2 and selected_bet and selected_amount:
            bet_confirmed = True
            print(f"New Bet Created: {selected_bet} Amount: {selected_amount}")
        
        # New bet button
        nx1, ny1, nx2, ny2 = new_bet_coords
        if nx1 <= x <= nx2 and ny1 <= y <= ny2:
            selected_bet = None
            selected_amount = None
            bet_confirmed = False
            print("Ready for a new bet!")

# --- Enhanced betting dashboard with player names ---
def draw_betting_dashboard(frame):
    global toggle_button_coords, dashboard_visible, option_coords, amount_coords
    global confirm_coords, new_bet_coords, current_betting_options
    
    h, w, _ = frame.shape
    sidebar_w = 600
    overlay = frame.copy()

    # --- Sidebar ---
    if dashboard_visible:
        cv2.rectangle(overlay, (w - sidebar_w, 0), (w, h), (30, 30, 30), -1)
        frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

    # --- Toggle Button ---
    button_h = 60
    toggle_button_coords = (w - sidebar_w, 0, w, button_h)
    cv2.rectangle(frame, (toggle_button_coords[0], toggle_button_coords[1]),
                  (toggle_button_coords[2], toggle_button_coords[3]), (0, 140, 0), -1)
    symbol = "▼" if dashboard_visible else "▲"
    cv2.putText(frame, "TENNIS BETTING", (toggle_button_coords[0]+20, toggle_button_coords[1]+40),
                cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, symbol, (toggle_button_coords[2]-50, toggle_button_coords[1]+40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if not dashboard_visible:
        return frame

    # --- Dashboard contents ---
    y_cursor = button_h + 20
    section_spacing = 50
    box_height = 50
    box_radius = 10
    inter_item_spacing = 15

    def draw_rounded_box(img, top_left, bottom_right, color, radius=10, thickness=-1):
        x1, y1 = top_left
        x2, y2 = bottom_right
        cv2.rectangle(img, (x1+radius, y1), (x2-radius, y2), color, thickness)
        cv2.rectangle(img, (x1, y1+radius), (x2, y2-radius), color, thickness)
        cv2.circle(img, (x1+radius, y1+radius), radius, color, thickness)
        cv2.circle(img, (x2-radius, y1+radius), radius, color, thickness)
        cv2.circle(img, (x1+radius, y2-radius), radius, color, thickness)
        cv2.circle(img, (x2-radius, y2-radius), radius, color, thickness)

    # --- Selected Bet ---
    cv2.putText(frame, "Selected Bet:", (w - sidebar_w + 20, y_cursor + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    y_cursor += 30
    draw_rounded_box(frame, (w - sidebar_w + 15, y_cursor), (w - 20, y_cursor + box_height), (50, 50, 50), box_radius)
    cv2.putText(frame, selected_bet if selected_bet else "None", (w - sidebar_w + 25, y_cursor + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
    y_cursor += box_height + section_spacing

    # --- Amount ---
    cv2.putText(frame, "Amount:", (w - sidebar_w + 20, y_cursor),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    y_cursor += 30
    draw_rounded_box(frame, (w - sidebar_w + 15, y_cursor), (w - 20, y_cursor + box_height), (50, 50, 50), box_radius)
    cv2.putText(frame, selected_amount if selected_amount else "None", (w - sidebar_w + 25, y_cursor + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
    y_cursor += box_height + section_spacing

    # --- Dynamic Betting Options (using Server/Receiver instead of player names) ---
    option_coords.clear()
    
    # Always use Server/Receiver terminology regardless of detected players
    current_betting_options = [
        ("Server", "to win next point"),
        ("Receiver", "to win next point"),
        ("Server", "to serve ace"),
        ("Receiver", "to make winner"),
        ("Match", "to go to deuce")
    ]
    
    # Draw betting options with Server/Receiver terminology
    for i, (player, action) in enumerate(current_betting_options):
        color = (80, 80, 80)  # Default color
        bet_text = f"{player} {action}"
        
        # Check if this option is selected
        if selected_bet and selected_bet == bet_text:
            color = (0, 120, 255)  # Highlight color
            
        draw_rounded_box(frame, (w - sidebar_w + 15, y_cursor), (w - 20, y_cursor + box_height), color, box_radius)
        
        # Display the betting option with Server/Receiver terminology
        display_text = f"{i+1}. {bet_text}"
        cv2.putText(frame, display_text, (w - sidebar_w + 25, y_cursor + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        option_coords.append((w - sidebar_w + 15, y_cursor, w - 20, y_cursor + box_height))
        y_cursor += box_height + inter_item_spacing

    y_cursor += section_spacing // 2

    # --- Amount Options ---
    amount_coords.clear()
    for i, amt in enumerate(bet_amount_options):
        color = (80, 80, 80)
        if selected_amount == amt:
            color = (0, 120, 255)
        draw_rounded_box(frame, (w - sidebar_w + 15, y_cursor), (w - 20, y_cursor + box_height), color, box_radius)
        cv2.putText(frame, f"{i+1}. {amt}", (w - sidebar_w + 25, y_cursor + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        amount_coords.append((w - sidebar_w + 15, y_cursor, w - 20, y_cursor + box_height))
        y_cursor += box_height + inter_item_spacing

    # --- Confirm & New Bet Buttons ---
    confirm_y = y_cursor + 20
    confirm_coords = (w - sidebar_w + 50, confirm_y, w - sidebar_w + 250, confirm_y + box_height)
    new_bet_coords = (w - sidebar_w + 270, confirm_y, w - sidebar_w + 470, confirm_y + box_height)

    draw_rounded_box(frame, (confirm_coords[0], confirm_coords[1]), (confirm_coords[2], confirm_coords[3]),
                     (0, 255, 0) if not bet_confirmed else (0, 150, 0), box_radius)
    cv2.putText(frame, "CONFIRM", (confirm_coords[0]+25, confirm_coords[1]+35), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 0), 2)

    draw_rounded_box(frame, (new_bet_coords[0], new_bet_coords[1]), (new_bet_coords[2], new_bet_coords[3]), (255, 140, 0), box_radius)
    cv2.putText(frame, "NEW BET", (new_bet_coords[0]+25, new_bet_coords[1]+35), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 0), 2)

    return frame

def process_video(input_path, output_path=None):
    global selected_bet, selected_amount, bet_confirmed
    global notification_text, notification_visible, notification_start_time, notification_closed
    global last_notification_time, player_names, player_positions, player_speeds, player_distances
    
    if not os.path.exists(input_path):
        print(f"Input video not found: {input_path}")
        return

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

    height, width, *_ = frame.shape
    print(f"Input video: {width}x{height}, {fps:.2f} FPS, {total_frames} frames")

    # Output writer - Optional
    out = None
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print("Could not create output video.")
            return
        print(f"Output will be saved to: {output_path}")
    else:
        print("No output file will be created - live preview only")

    # Improved tracker settings to reduce flickering
    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
    frame_count = 0
    detect_every = 1  # Reduced from 2 to 1 for smoother tracking at faster speeds
    last_detections = np.empty((0, 5))

    # Interactive mode
    show_preview = True  # Enable for betting dashboard
    playback_speed = 3.2  # Speed multiplier - 2x faster than normal
    
    # Setup interactive window
    if show_preview:
        cv2.namedWindow("Tennis Betting Analysis", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Tennis Betting Analysis", mouse_callback)
    
    last_notification_time = time.time()
    paused = False
    
    print(f"🎬 Processing started with betting dashboard at {playback_speed}x speed...")
    print("🎾 Players will be assigned names as they are detected...")
    print("⏯️  Controls: SPACE = Pause/Resume, Q = Quit")

    while ret:
        detections = []

        # Run YOLO detection
        if frame_count % detect_every == 0:
            results = model(frame, verbose=False)
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                for box, cls, conf in zip(results[0].boxes.xyxy.cpu().numpy(),
                                        results[0].boxes.cls.cpu().numpy(),
                                        results[0].boxes.conf.cpu().numpy()):
                    if int(cls) == 0 and conf > 0.3:  # person class
                        x1, y1, x2, y2 = box
                        
                        # Filter for tennis players (ignore background people)
                        box_width = x2 - x1
                        box_height = y2 - y1
                        box_area = box_width * box_height
                        
                        # Size filtering: focus on larger detections (main players)
                        min_area = (width * height) * 0.002  # At least 0.2% of frame
                        max_area = (width * height) * 0.3    # At most 30% of frame
                        
                        # Position filtering: focus on court area (ignore extreme edges)
                        margin_x = width * 0.05   # 5% margin from sides
                        margin_y = height * 0.1   # 10% margin from top/bottom
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        # Aspect ratio filtering: people should be taller than wide
                        aspect_ratio = box_height / box_width if box_width > 0 else 0
                        
                        if (min_area < box_area < max_area and 
                            margin_x < center_x < width - margin_x and
                            margin_y < center_y < height - margin_y and
                            aspect_ratio > 1.2):  # Players should be taller than wide
                            detections.append([x1, y1, x2, y2, conf])

            if len(detections) > 0:
                detections_np = np.array(detections)
                last_detections = detections_np
            else:
                detections_np = np.empty((0, 5))
                detections_np = last_detections
        else:
            detections_np = last_detections

        tracked_objects = tracker.update(detections_np)

        # Calculate speeds and distances, assign names
        current_positions = {}
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = obj
            obj_id = int(obj_id)
            
            # Calculate center position
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            current_positions[obj_id] = (center_x, center_y)
            
            # Assign name using enhanced function
            player_name = assign_player_name(obj_id)
            
            # Initialize distance tracking for new players
            if obj_id not in player_distances:
                player_distances[obj_id] = 0.0
            
            # Calculate speed and distance if we have previous position
            speed = 0.0
            if obj_id in player_positions:
                prev_x, prev_y = player_positions[obj_id]
                # Calculate distance moved (in pixels)
                distance_pixels = np.sqrt((center_x - prev_x)**2 + (center_y - prev_y)**2)
                
                # Convert pixels to meters
                distance_meters = distance_pixels * PIXELS_TO_METERS
                
                # Add to total distance covered
                player_distances[obj_id] += distance_meters
                
                # Calculate speed using normal match speed (not accelerated)
                # Assuming roughly 30 pixels = 1 meter and original fps (not playback speed)
                speed_ms = (distance_pixels * 0.033) * (fps / playback_speed)  # Account for playback speed
                speed_kmh = speed_ms * 3.6  # convert to km/h
                speed = max(0, min(speed_kmh, 50))  # Cap at reasonable tennis speed
            
            player_speeds[obj_id] = speed
        
        # Update positions for next frame
        player_positions.update(current_positions)

        # Draw tracked boxes with names, speeds, and distances
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = obj
            obj_id = int(obj_id)
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Ensure coordinates are within frame bounds
            x1 = max(0, min(x1, width-1))
            y1 = max(0, min(y1, height-1))
            x2 = max(0, min(x2, width-1))
            y2 = max(0, min(y2, height-1))
            
            cx, cy = (x1+x2)//2, (y1+y2)//2

            # Draw bounding box with player-specific color
            player_color = (0, 255, 0) if obj_id % 2 == 0 else (255, 0, 255)  # Green or Magenta
            cv2.rectangle(frame, (x1, y1), (x2, y2), player_color, 2)
            
            # Get player name, speed, and distance
            name = player_names.get(obj_id, f"Player_{obj_id}")  # Fallback to Player_ID if name not found
            speed = player_speeds.get(obj_id, 0.0)
            distance = player_distances.get(obj_id, 0.0)
            
            # Create info box background for better readability
            info_bg_height = 80
            cv2.rectangle(frame, (x1, y1-info_bg_height), (x2+100, y1), (0, 0, 0), -1)  # Black background
            cv2.rectangle(frame, (x1, y1-info_bg_height), (x2+100, y1), player_color, 2)  # Colored border
            
            # Draw player name (larger, more prominent)
            cv2.putText(frame, name, (x1+5, y1-55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Draw speed in km/h
            speed_text = f"{speed:.1f} km/h"
            cv2.putText(frame, speed_text, (x1+5, y1-35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Draw total distance covered in meters
            distance_text = f"{distance:.1f}m"
            cv2.putText(frame, distance_text, (x1+5, y1-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Draw center point
            cv2.circle(frame, (cx, cy), 6, player_color, -1)

        # Add betting dashboard
        frame_with_dashboard = draw_betting_dashboard(frame.copy())
        
        # Handle notifications
        if show_preview:
            current_time = time.time()
            
            # Show new notification every 13 seconds
            if current_time - last_notification_time >= notification_interval:
                notification_text = generate_notification()
                notification_visible = True
                notification_closed = False
                notification_start_time = current_time
                last_notification_time = current_time
                print("Notification triggered:", notification_text)

            # Auto-hide after 2.5 seconds
            if notification_visible and not notification_closed and current_time - notification_start_time >= notification_duration:
                notification_visible = False

            # Draw notification if visible
            if notification_visible:
                frame_with_dashboard = draw_notification(frame_with_dashboard, notification_text)

        # Save frame to output file if specified
        if out:
            out.write(frame_with_dashboard)

        # Interactive preview with enhanced pause functionality
        if show_preview:
            # Add pause indicator to the frame
            if paused:
                # Draw pause indicator
                pause_overlay = frame_with_dashboard.copy()
                cv2.rectangle(pause_overlay, (width//2-200, height//2-50), (width//2+250, height//2+50), (0, 0, 0), -1)
                cv2.rectangle(pause_overlay, (width//2-200, height//2-50), (width//2+250, height//2+50), (0, 255, 255), 3)
                cv2.putText(pause_overlay, "PAUSED", (width//2-70, height//2-10), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 2)
                cv2.putText(pause_overlay, "Press SPACE to resume", (width//2-120, height//2+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                frame_with_dashboard = pause_overlay
            
            cv2.imshow("Tennis Betting Analysis", frame_with_dashboard)
            
            # Handle key presses
            wait_time = 1 if paused else 10  # Wait longer when paused for better responsiveness
            key = cv2.waitKey(wait_time) & 0xFF
            
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord(' '):
                paused = not paused
                if paused:
                    print("Video PAUSED - Press SPACE to resume")
                else:
                    print("Video RESUMED")
                    
            # When paused, don't advance to next frame
            if paused:
                continue

        # Progress indicator
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")

        frame_count += 1
        
        # Skip frames based on playback speed for faster processing (only when not paused)
        if not paused:
            skip_frames = int(playback_speed - 1)
            for _ in range(skip_frames):
                ret, temp_frame = cap.read()
                if not ret:
                    break
                frame_count += 1
            ret, frame = cap.read()

    cap.release()
    if out:
        out.release()
        print(f"Done! Processed {frame_count} frames. Output saved to {output_path}")
    else:
        print(f"Done! Processed {frame_count} frames. No output file created.")
        
    if show_preview:
        cv2.destroyAllWindows()
    
    # Print final statistics
    print("\n=== PLAYER STATISTICS ===")
    for player_id, name in player_names.items():
        distance = player_distances.get(player_id, 0.0)
        speed = player_speeds.get(player_id, 0.0)
        print(f"{name} (ID: {player_id}): {distance:.2f}m covered, Current speed: {speed:.1f} km/h")
    
    # Print final bet
    if bet_confirmed:
        print(f"\nFinal Bet: {selected_bet} Amount: {selected_amount}")
    else:
        print("\nNo bet was confirmed.")

if __name__ == "__main__":
    input_video = "test_2.mp4"
    # Don't create output file - set to None for live preview only
    output_video = None  # Change to a path if you want to save output
    process_video(input_video, output_video)
