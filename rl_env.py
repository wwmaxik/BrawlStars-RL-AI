import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pyautogui as py
import time
import os
from math import sqrt
from collections import deque
from modules.windowcapture import WindowCapture
from modules.detection import Detection
from constants import Constants
import cv2
import threading
from dashboard import DashboardData
from modules.tui import BrawlTUI

# MAXIMIZE FPS
py.PAUSE = 0.0
py.FAILSAFE = False

class BrawlStarsYoloEnv(gym.Env):
    def __init__(self, frame_stack=Constants.frame_stack):
        super().__init__()
        self.tui = BrawlTUI()
        self.tui.start()

        self.wincap = WindowCapture(Constants.window_name)
        self.windowSize = self.wincap.get_dimension()
        self.wincap.set_window()
        self.detector = Detection(self.windowSize, Constants.model_file_path, Constants.classes, Constants.heightScaleFactor)
        self.wincap.start()
        self.detector.start()
        self.center_window = (self.windowSize[0] / 2, int((self.windowSize[1] / 2) + Constants.midpoint_offset))

        # Action Space: MultiDiscrete для стабильности
        # [0]: Movement X (0=left, 1=none, 2=right)
        # [1]: Movement Y (0=up, 1=none, 2=down)
        # [2]: Attack (0=none, 1=attack)
        # [3]: Super (0=none, 1=use)
        # [4]: Gadget (0=none, 1=use)
        self.action_space = spaces.MultiDiscrete([3, 3, 2, 2, 2])
        
        # Observation Space с Frame Stacking
        self.img_size = Constants.img_size
        self.frame_stack = frame_stack
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0, high=255, 
                shape=(3 * self.frame_stack, self.img_size, self.img_size), 
                dtype=np.uint8
            ),
            "vector": spaces.Box(low=-1.0, high=1.0, shape=(Constants.vector_size,), dtype=np.float32)
        })

        # Frame buffer для stacking
        self.frame_buffer = deque(maxlen=self.frame_stack)

        # State tracking
        self.match_count = 0
        self.total_steps = 0
        self.env_closed = False
        self.took_damage = False
        self.in_poison = False
        self.poison_zones = {"w": False, "s": False, "a": False, "d": False}
        self.match_start_time = self.last_seen_object_time = time.time()
        self.match_max_duration = 180.0  # 3 минуты макс

        # Input Threading
        self.target_keys = set()
        self.pressed_keys = set()
        self.action_trigger = {"space": False, "e": False, "f": False}
        self.input_lock = threading.Lock()

        # Stats
        self.current_fps = 0
        self.last_step_time = time.time()

        # Reward helpers
        self.prev_box_dist = 1.0
        self.prev_enemy_dist = 1.0
        self.known_boxes_count = 0
        self.prev_movement_keys = set()
        self.cumulative_damage = 0.0
        self.boxes_destroyed_this_match = 0

        # Health tracking
        self.current_health_pct = 1.0  # Текущее HP (через отслеживание урона)
        self.last_health_pct = 1.0
        self.health_decreased = False
        
        # Idle tracking (предотвращение "Idle Disconnect")
        self.idle_steps = 0
        self.max_idle_steps = 10  # После N шагов без движения — штраф
        
        # Idle Disconnect detection state
        self.idle_disconnect_detected = False
        self.last_reload_time = 0
        self.reload_cooldown = 5.0  # Не нажимать R чаще чем раз в 5 сек

        # Monitor threads
        self.dmg_thread = threading.Thread(target=self._damage_monitor, daemon=True)
        self.dmg_thread.start()
        self.action_thread = threading.Thread(target=self._input_executor, daemon=True)
        self.action_thread.start()

    def _input_executor(self):
        last_press_times = {"space": 0, "e": 0, "f": 0}
        cooldowns = {"space": 0.35, "e": 5.0, "f": 3.0}
        while not self.env_closed:
            with self.input_lock:
                current_target = self.target_keys.copy()
                to_press = {k: v for k, v in self.action_trigger.items() if v}
                for k in self.action_trigger: self.action_trigger[k] = False
            for k in self.pressed_keys - current_target: py.keyUp(k)
            for k in current_target - self.pressed_keys: py.keyDown(k)
            self.pressed_keys = current_target
            now = time.time()
            for key, active in to_press.items():
                if now - last_press_times[key] > cooldowns[key]:
                    py.press(key); last_press_times[key] = now
            time.sleep(0.005)

    def _damage_monitor(self):
        while not self.env_closed:
            time.sleep(0.015)
            screenshot = self.wincap.screenshot
            if screenshot is not None:
                self.detector.update(screenshot)
                DashboardData.raw_frame = screenshot
                h, w = screenshot.shape[:2]
                cx, cy = w // 2, int(h // 2 + Constants.midpoint_offset)
                rois = {
                    "w": screenshot[max(0, cy-180):max(0, cy-50), max(0, cx-80):min(w, cx+80)],
                    "s": screenshot[cy+40:min(h, cy+150), max(0, cx-80):min(w, cx+80)],
                    "a": screenshot[max(0, cy-100):min(h, cy+100), max(0, cx-180):max(0, cx-50)],
                    "d": screenshot[max(0, cy-100):min(h, cy+100), min(w, cx+50):min(w, cx+180)]
                }
                lp, up = np.array([30, 25, 160]), np.array([95, 140, 255])
                active_sides = []
                for side, reg in rois.items():
                    if reg.size == 0: continue
                    mask = cv2.inRange(cv2.cvtColor(reg, cv2.COLOR_BGR2HSV), lp, up)
                    self.poison_zones[side] = (cv2.countNonZero(mask) / (reg.shape[0] * reg.shape[1])) > 0.12
                    if self.poison_zones[side]: active_sides.append(side.upper())
                self.in_poison = len(active_sides) > 0
                DashboardData.stats["Poison"] = f"YES ({', '.join(active_sides)})" if self.in_poison else "NO"
                
                # --- IMPROVED: Calculate poison direction vector ---
                poison_vec_x, poison_vec_y = 0.0, 0.0
                if self.in_poison:
                    if self.poison_zones['w']: poison_vec_y = -1.0
                    if self.poison_zones['s']: poison_vec_y = 1.0
                    if self.poison_zones['a']: poison_vec_x = -1.0
                    if self.poison_zones['d']: poison_vec_x = 1.0
                    # Normalize
                    mag = sqrt(poison_vec_x**2 + poison_vec_y**2)
                    if mag > 0:
                        poison_vec_x /= mag
                        poison_vec_y /= mag
                self.poison_direction = (poison_vec_x, poison_vec_y)
                
                # --- IDLE DISCONNECT DETECTION ---
                # Обнаружение окна "Idle Disconnect" по серой плашке в центре
                # и нажатие R для перезагрузки
                self._check_idle_disconnect(screenshot, h, w, cx, cy)

                if self.detector.player_topleft and self.detector.player_bottomright:
                    tl, br = self.detector.player_topleft, self.detector.player_bottomright
                    tl_abs = (tl[0] + self.wincap.offset_x, tl[1] + self.wincap.offset_y); br_abs = (br[0] + self.wincap.offset_x, br[1] + self.wincap.offset_y)
                    width, height = abs(tl_abs[0] - br_abs[0]), abs(tl_abs[1] - br_abs[1])
                    try:
                        if (py.pixelMatchesColor(int(tl_abs[0] + width/3), int(tl_abs[1] - height/2), (204, 34, 34), tolerance=20) or
                            py.pixelMatchesColor(int(tl_abs[0] + 2*width/3), int(tl_abs[1] - height/2), (204, 34, 34), tolerance=20)):
                            self.took_damage = True
                    except OSError: pass

    def _get_health_percentage(self):
        """Определяет % здоровья игрока.
        
        Вместо ненадёжного пиксельного анализа используем 
        отслеживание урона: стартуем с 100%, уменьшаем при получении урона.
        """
        # _damage_monitor уже отслеживает took_damage
        # Каждое попадание = примерно 10-15% урона
        if self.took_damage:
            self.current_health_pct = max(0.0, self.current_health_pct - 0.12)
            self.took_damage = False
        return self.current_health_pct

    def _check_idle_disconnect(self, screenshot, h, w, cx, cy):
        """Обнаружение окна 'Idle Disconnect' и нажатие R.
        
        Детекция по:
        1. Серая плашка в центре экрана (темнее игры)
        2. Бирюзовая кнопка RELOAD (cyan/teal цвет)
        """
        # Проверяем cooldown чтобы не спамить R
        now = time.time()
        if now - self.last_reload_time < self.reload_cooldown:
            self.idle_disconnect_detected = False
            return
        
        # ROI в центре экрана где появляется окно
        # Окно примерно 40% ширины и 25% высоты экрана
        win_w = int(w * 0.45)
        win_h = int(h * 0.30)
        x1 = cx - win_w // 2
        y1 = cy - win_h // 2
        x2 = cx + win_w // 2
        y2 = cy + win_h // 2
        
        # Берём регион центральной плашки
        popup_roi = screenshot[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
        
        if popup_roi.size == 0:
            self.idle_disconnect_detected = False
            return
        
        # Детекция по бирюзовому цвету кнопки RELOAD
        # RELOAD текст: примерно cyan/teal (#4ECDC4 или类似)
        hsv = cv2.cvtColor(popup_roi, cv2.COLOR_BGR2HSV)
        
        # Cyan/teal диапазон для RELOAD кнопки
        lower_reload = np.array([85, 40, 150])
        upper_reload = np.array([105, 255, 255])
        reload_mask = cv2.inRange(hsv, lower_reload, upper_reload)
        
        reload_pixels = cv2.countNonZero(reload_mask)
        
        # Также детектируем серую плашку (тёмно-серый фон)
        lower_gray = np.array([0, 0, 40])
        upper_gray = np.array([20, 25, 80])
        gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)
        gray_pixels = cv2.countNonZero(gray_mask)
        
        total_popup_pixels = popup_roi.shape[0] * popup_roi.shape[1]
        
        # Если нашли и серую плашку И бирюзовые пиксели — это Idle Disconnect
        has_gray_popup = gray_pixels > total_popup_pixels * 0.15  # >15% серого
        has_reload_btn = reload_pixels > 20  # Минимум 20 бирюзовых пикселей
        
        if has_gray_popup and has_reload_btn:
            if not self.idle_disconnect_detected:
                self.idle_disconnect_detected = True
                self._log_event("[SYSTEM ] Idle Disconnect detected! Pressing R...")
                DashboardData.stats["State"] = "RELOADING"
            
            # Нажимаем R для перезагрузки
            try:
                py.press('r')
                self.last_reload_time = now
                self._log_event("[SYSTEM ] Pressed R to reload match")
            except Exception as e:
                self._log_event(f"[ERROR  ] Failed to press R: {e}")
        else:
            self.idle_disconnect_detected = False

    def _get_obs(self):
        vec_obs = np.zeros(Constants.vector_size, dtype=np.float32)
        screenshot = self.wincap.screenshot
        any_on = False
        if self.detector.results:
            for res in self.detector.results:
                if len(res) > 0: any_on = True; break
        
        # --- Image processing с frame stacking ---
        if screenshot is not None:
            img_resized = cv2.resize(screenshot, (self.img_size, self.img_size))
            if any_on:
                img_display = cv2.resize(img_resized, (self.img_size * 5, self.img_size * 5), interpolation=cv2.INTER_NEAREST)
                DashboardData.ai_frame = img_display
            img_obs = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        else:
            img_obs = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        
        # Добавляем текущий кадр в буфер
        self.frame_buffer.append(img_obs)
        
        # Создаём стек кадров (H, W, 3*N)
        if len(self.frame_buffer) < self.frame_stack:
            # Дополняем первым кадром если буфер не заполнен
            while len(self.frame_buffer) < self.frame_stack:
                self.frame_buffer.appendleft(self.frame_buffer[0])
        
        stacked_frames = np.concatenate(list(self.frame_buffer), axis=2)  # (H, W, 3*N)
        # Transpose для PyTorch: (3*N, H, W)
        stacked_frames = np.transpose(stacked_frames, (2, 0, 1))
        
        # --- Vector observation (расширенный) ---
        if self.detector.results and len(self.detector.results) >= 4:
            p_pos = self.detector.results[0][0] if self.detector.results[0] else self.center_window
            
            # [0] Damage flag
            if self.took_damage: vec_obs[0] = 1.0; self.took_damage = False
            
            # [1] Health percentage
            current_health = self._get_health_percentage()
            vec_obs[1] = current_health * 2.0 - 1.0  # Нормализация в [-1, 1]
            if current_health < self.last_health_pct:
                self.health_decreased = True
            self.last_health_pct = current_health
            
            # [2] Cube count (обнаружено)
            boxes_list = self.detector.results[3] if len(self.detector.results) > 3 else []
            box_count = len(boxes_list)
            vec_obs[2] = min(box_count / 10.0, 1.0)  # Нормализация (макс 10 коробок)
            
            # [3] Enemy count (в поле зрения)
            enemies_list = self.detector.results[2] if len(self.detector.results) > 2 else []
            enemy_count = len(enemies_list)
            vec_obs[3] = min(enemy_count / 8.0, 1.0)  # Нормализация (макс 8 врагов)
            
            # [20] Poison flag
            if self.in_poison: vec_obs[20] = 1.0
            
            # [21:23] Poison direction
            vec_obs[21], vec_obs[22] = self.poison_direction
            
            # [24] Time in match (нормализованное)
            match_duration = time.time() - self.match_start_time
            vec_obs[24] = min(match_duration / self.match_max_duration, 1.0)
            
            def get_closest_relative(index):
                items = self.detector.results[index]
                if not items: return 0.0, 0.0, 1.0
                closest = items[0]; min_dist = 99999
                for item in items:
                    dist = sqrt(((item[0]-p_pos[0])/(self.windowSize[0]/24))**2 + ((item[1]-p_pos[1])/(self.windowSize[1]/17))**2)
                    if dist < min_dist: min_dist = dist; closest = item
                return (closest[0] - p_pos[0]) / (self.windowSize[0] / 2), (closest[1] - p_pos[1]) / (self.windowSize[1] / 2), min(min_dist / 15.0, 1.0)
            
            def get_second_closest(index):
                items = self.detector.results[index]
                if not items or len(items) < 2: return 0.0, 0.0, 1.0
                # Сортируем по расстоянию
                sorted_items = sorted(items, key=lambda item: sqrt(((item[0]-p_pos[0])/(self.windowSize[0]/24))**2 + ((item[1]-p_pos[1])/(self.windowSize[1]/17))**2))
                second = sorted_items[1] if len(sorted_items) > 1 else sorted_items[0]
                dist = sqrt(((second[0]-p_pos[0])/(self.windowSize[0]/24))**2 + ((second[1]-p_pos[1])/(self.windowSize[1]/17))**2)
                return (second[0] - p_pos[0]) / (self.windowSize[0] / 2), (second[1] - p_pos[1]) / (self.windowSize[1] / 2), min(dist / 15.0, 1.0)
            
            def get_avg_distance(index):
                items = self.detector.results[index]
                if not items: return 1.0
                total_dist = 0
                for item in items:
                    dist = sqrt(((item[0]-p_pos[0])/(self.windowSize[0]/24))**2 + ((item[1]-p_pos[1])/(self.windowSize[1]/17))**2)
                    total_dist += dist
                return min((total_dist / len(items)) / 15.0, 1.0)
            
            # [4:7] Ближайший враг
            vec_obs[4:7] = get_closest_relative(2)
            # [7:10] Враг #2
            vec_obs[7:10] = get_second_closest(2)
            # [10:13] Ближайшая коробка
            vec_obs[10:13] = get_closest_relative(3)
            # [13:16] Ближайший куст
            vec_obs[13:16] = get_closest_relative(1)
            
            # [25] Среднее расстояние до всех врагов
            vec_obs[25] = get_avg_distance(2)
            
            # [16:20] Стены через Canny edge detection
            if screenshot is not None:
                h_img, w_img = screenshot.shape[:2]; cx, cy, chk, bsz = int(self.windowSize[0] / 2), int(self.windowSize[1] / 2 + Constants.midpoint_offset), 40, 15
                try:
                    gray = cv2.cvtColor(screenshot, cv2.COLOR_BGRGRAY); edges = cv2.Canny(gray, 50, 150)
                    def is_blocked(x, y):
                        if x - bsz < 0 or x + bsz > w_img or y - bsz < 0 or y + bsz > h_img: return 1.0
                        roi = edges[int(y-bsz):int(y+bsz), int(x-bsz):int(x+bsz)]
                        return 1.0 if (np.sum(roi) / 255) > 50 else 0.0
                    vec_obs[16], vec_obs[17], vec_obs[18], vec_obs[19] = is_blocked(cx, cy-chk), is_blocked(cx, cy+chk), is_blocked(cx-chk, cy), is_blocked(cx+chk, cy)
                except: pass
        
        return {"image": stacked_frames, "vector": vec_obs}

    def step(self, action):
        self.total_steps += 1
        
        # Декодирование MultiDiscrete action
        move_x, move_y, attack, spr, gdgt = action
        
        target_keys, any_on = set(), False
        if self.detector.results:
            for res in self.detector.results:
                if len(res) > 0: any_on = True; break
        
        curr = time.time()
        dt = curr - self.last_step_time; self.current_fps = int(1.0 / dt) if dt > 0 else 0; self.last_step_time = curr

        # --- Movement ---
        if any_on:
            self.last_seen_object_time = curr
            
            # Декодируем movement (0=left/up, 1=none, 2=right/down)
            if move_x == 0: target_keys.add('a')
            elif move_x == 2: target_keys.add('d')
            
            if move_y == 0: target_keys.add('w')
            elif move_y == 2: target_keys.add('s')

            # --- ABSOLUTE POISON AVOIDANCE (Overriding AI) ---
            if self.poison_zones['w']:
                if 'w' in target_keys: target_keys.remove('w')
                target_keys.add('s')
            if self.poison_zones['s']:
                if 's' in target_keys: target_keys.remove('s')
                target_keys.add('w')
            if self.poison_zones['a']:
                if 'a' in target_keys: target_keys.remove('a')
                target_keys.add('d')
            if self.poison_zones['d']:
                if 'd' in target_keys: target_keys.remove('d')
                target_keys.add('a')

        # --- IMPROVED Reward Function ---
        match_duration = curr - self.match_start_time

        # Base survival reward (увеличено)
        reward = 0.01 * (1 + match_duration / 60.0)

        # --- Idle Penalty (предотвращение "Idle Disconnect") ---
        # Если бот не двигается (move_x=1 и move_y=1 = "none")
        is_moving = (move_x != 1 or move_y != 1)
        if is_moving:
            self.idle_steps = 0  # Сброс idle counter
        else:
            self.idle_steps += 1
            if self.idle_steps > self.max_idle_steps:
                idle_penalty = -0.05 * (self.idle_steps - self.max_idle_steps)
                reward += idle_penalty  # Нарастающий штраф за idling
                if self.idle_steps % 20 == 0:  # Логируем каждые 20 idle шагов
                    self._log_event(f"[IDLE   ] Not moving! idle_steps={self.idle_steps} penalty={idle_penalty:.3f}")

        # --- Action execution ---
        with self.input_lock:
            if self.target_keys != target_keys: 
                reward -= 0.005  # Уменьшен штраф за смену направления
                self.target_keys = target_keys
            
            if any_on:
                if attack == 1: self.action_trigger["space"] = True
                if spr == 1: self.action_trigger["e"] = True
                if gdgt == 1: self.action_trigger["f"] = True

        obs = self._get_obs()
        
        if any_on:
            enemies_list = self.detector.results[2] if len(self.detector.results) > 2 else []
            boxes_list = self.detector.results[3] if len(self.detector.results) > 3 else []
            
            # --- Combat rewards ---
            if attack == 1:
                if not enemies_list and not boxes_list: 
                    reward -= 0.03; self._log_event("[PENALTY] Ammo wasted")
                else: 
                    reward += 0.02; self._log_event("[REWARD ] Precision fire")
            
            # --- Box collection ---
            curr_box_dist = obs["vector"][12]  # [10:13] cubebox
            if curr_box_dist < self.prev_box_dist: 
                reward += 0.005
            self.prev_box_dist = curr_box_dist
            
            curr_cnt = len(boxes_list)
            if curr_cnt < self.known_boxes_count and curr_box_dist < 0.3:
                reward += 0.3;  # УВЕЛИЧЕНО с 0.2
                self.boxes_destroyed_this_match += 1
                self._log_event(f"[REWARD ] BOX DESTROYED! (Total: {self.boxes_destroyed_this_match})")
            self.known_boxes_count = curr_cnt
            
            # --- Enemy engagement ---
            if enemies_list:
                dist = obs["vector"][6]  # [4:7] enemy
                if dist < self.prev_enemy_dist and dist < 0.2: 
                    reward -= 0.01; self._log_event("[PENALTY] Too close to enemy!")
                elif 0.3 < dist < 0.6: 
                    reward += 0.005
                self.prev_enemy_dist = dist
                
                # Награда за нахождение в кустах когда враг рядом
                bush_list = self.detector.results[1] if len(self.detector.results) > 1 else []
                if bush_list:
                    bush_dist = obs["vector"][15]  # [13:16] bush
                    if bush_dist < 0.3 and dist > 0.25:
                        reward += 0.02; self._log_event("[REWARD ] Stealth mode!")

        # --- Damage & Poison penalties ---
        if obs["vector"][0] > 0.5: 
            reward -= 0.15; self._log_event("[PENALTY] DAMAGE TAKEN!")
            self.cumulative_damage += 0.15
        
        if obs["vector"][20] > 0.5: 
            reward -= 0.8; self._log_event("[FATAL  ] IN POISON!")  # УВЕЛИЧЕНО с -0.5
        
        # --- Награда за низкое здоровье (учит выживать) ---
        health_pct = (obs["vector"][1] + 1.0) / 2.0  # Denormalize
        if health_pct < 0.3:
            reward -= 0.05  # Стимулирует осторожность при низком HP
        
        # --- Проверка конца матча ---
        if curr - self.last_seen_object_time > 4.0:
            # Финальный расчёт награды
            final_r = min(match_duration / 120.0, 2.0)  # УВЕЛИЧЕНО макс с 1.5 до 2.0
            reward += final_r
            
            # Бонус за выживание с низким здоровьем
            if health_pct < 0.5 and match_duration > 60:
                final_r += 0.5
                reward += 0.5
                self._log_event(f"[BONUS  ] Survived with low HP! +0.5")
            
            # Бонус за собранные кубы
            cube_bonus = self.boxes_destroyed_this_match * 0.2
            reward += cube_bonus
            if cube_bonus > 0:
                self._log_event(f"[BONUS  ] Cube bonus: +{cube_bonus:.2f}")
            
            self._log_event(f"MATCH END | Survived {round(match_duration,1)}s | Final: +{round(final_r,2)} | Cubes: {self.boxes_destroyed_this_match}")
            
            with self.input_lock: self.target_keys.clear()
            return obs, reward, True, False, {"episode_reward": reward, "match_duration": match_duration}

        # --- UI Updates ---
        self._update_ui(match_duration, reward, attack, obs)
        return obs, reward, False, False, {}

    def _log_event(self, msg):
        self.tui.add_log(msg); DashboardData.add_log(msg)
    
    def _update_ui(self, duration, reward, attack, obs):
        # --- AI Vision State для TUI ---
        obs_vec = obs["vector"]
        
        # Denormalize health
        health_pct = (obs_vec[1] + 1.0) / 2.0 * 100  # 0-100%
        health_bar_len = int(health_pct / 5)
        health_bar = "█" * max(0, min(health_bar_len, 20)) + "░" * max(0, 20 - health_bar_len)
        
        # Enemy distances
        e1_dist = obs_vec[6] if obs_vec[6] > 0 else 0
        e2_dist = obs_vec[9] if obs_vec[9] > 0 else 0
        box_dist = obs_vec[12] if obs_vec[12] > 0 else 0
        bush_dist = obs_vec[15] if obs_vec[15] > 0 else 0
        
        def fmt_dist(val, max_val=1.0):
            if val <= 0 or val > max_val:
                return "---"
            return f"{val:.2f}"
        
        # Poison
        poison_status = "YES" if self.in_poison else "NO"
        poison_dir = "---"
        if self.in_poison:
            px, py = self.poison_direction
            dirs = []
            if py < -0.3: dirs.append("UP")
            if py > 0.3: dirs.append("DOWN")
            if px < -0.3: dirs.append("LEFT")
            if px > 0.3: dirs.append("RIGHT")
            poison_dir = " ".join(dirs) if dirs else "CENTER"
        
        # Walls
        wall_str = ""
        for i in range(16, 20):
            wall_str += "1" if obs_vec[i] > 0.5 else "0"
        
        ai_state = {
            "health_pct": health_pct,
            "health_bar": health_bar,
            "enemy_count": int(obs_vec[3] * 8),
            "box_count": int(obs_vec[2] * 10),
            "enemy1_dist": fmt_dist(e1_dist),
            "enemy2_dist": fmt_dist(e2_dist),
            "box_dist": fmt_dist(box_dist),
            "bush_dist": fmt_dist(bush_dist),
            "poison_status": poison_status,
            "poison_dir": poison_dir,
            "walls": wall_str,
            "cubes_destroyed": self.boxes_destroyed_this_match,
            "idle_steps": self.idle_steps,
        }
        stats = {"Match": self.match_count, "Steps": self.total_steps, "Duration": f"{round(duration,1)}s", "FPS": self.current_fps, "Reward": round(reward, 4), "Poison": DashboardData.stats.get("Poison", "NO")}
        self.tui.update_header(self.match_count, self.total_steps)
        self.tui.update_game_stats_with_vision(stats, ai_state)
        self.tui.update_train_stats(DashboardData.train_stats)
        self.tui.update_footer(self.target_keys, str(attack > 0.5), reward)
        DashboardData.stats.update(stats); DashboardData.stats["Keys Pressed"] = str(list(self.target_keys))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self.match_count += 1
        self.tui.update_header(self.match_count, self.total_steps)
        self._log_event(f"MATCH #{self.match_count} STARTING")
        with self.input_lock: self.target_keys.clear()
        self.prev_box_dist = 1.0; self.prev_enemy_dist = 1.0; self.known_boxes_count = 0
        self.frame_buffer.clear()
        self.cumulative_damage = 0.0
        self.boxes_destroyed_this_match = 0
        self.current_health_pct = 1.0
        self.last_health_pct = 1.0
        self.health_decreased = False
        self.idle_steps = 0
        self.idle_disconnect_detected = False
        self.last_reload_time = 0
        
        while True:
            any_on = False
            if self.detector.results:
                for res in self.detector.results:
                    if len(res) > 0: any_on = True; break
            if any_on:
                self.match_start_time = self.last_seen_object_time = time.time(); break
            time.sleep(0.5)
        
        obs = self._get_obs()
        return obs, {}

    def close(self):
        self.env_closed = True; self.tui.stop(); self.wincap.stop(); self.detector.stop(); cv2.destroyAllWindows()
