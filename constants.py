class Constants:
    # --- WINDOW ---
    window_name = "Bluestacks App Player"
    focused_window = False

    # --- HARDWARE ---
    # Set to True if you have an NVIDIA GPU and CUDA installed correctly
    nvidia_gpu = True

    # --- MODEL (VISION) ---
    # Path to the YOLOv8 model
    model_file_path = "yolov8_model/yolov8.pt" if nvidia_gpu else "yolov8_model/yolov8_openvino_model"
    imgsz = (384, 640)
    half = True
    classes = ["Player", "Bush", "Enemy", "Cubebox"]
    threshold = [0.37, 0.47, 0.57, 0.65]

    # Perspective correction factor
    heightScaleFactor = 0.15

    # --- BOT LOGIC ---
    midpoint_offset = 12
    movement_key = "right"

    #! TRAINING SETTINGS
    img_size = 84
    
    # Frame stacking для observation
    frame_stack = 4  # Количество кадров для стека
    
    # Размер observation vector (расширенный)
    # [0]: damage flag
    # [1]: health %
    # [2]: cube count (визуально обнаружено)
    # [3]: enemy count (в поле зрения)
    # [4:7]: ближайший Enemy (dx, dy, dist)
    # [7:10]: враг #2 (dx, dy, dist)
    # [10:13]: ближайший Cubebox (dx, dy, dist)
    # [13:16]: ближайший Bush (dx, dy, dist)
    # [16:20]: стены W, S, A, D
    # [20]: poison flag
    # [21:23]: poison direction (vec_x, vec_y)
    # [24]: time in match (нормализованное, 0-1)
    # [25]: avg distance to all enemies
    vector_size = 26

if __name__ == "__main__":
    from modules.print import bcolors
    status = "GPU (CUDA)" if Constants.nvidia_gpu else "CPU (OpenVINO)"
    print(f"AI Constants loaded. Mode: {status}")
