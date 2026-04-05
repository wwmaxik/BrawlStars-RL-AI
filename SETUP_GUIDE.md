# 🚀 Setup & Quick Start Guide

Пошаговая инструкция для запуска Brawl Stars RL Bot.

---

## 📋 Pre-requirements

### 1. Operating System
- **Windows 10/11** (64-bit)
- Linux/Mac **не поддерживается** (требуется BlueStacks)

### 2. Python
- **Версия:** 3.11.6
- **Скачать:** [python.org/downloads](https://www.python.org/downloads/release/python-3116/)
- ⚠️ Другие версии **не тестировались**

### 3. BlueStacks 5
- **Скачать:** [bluestacks.com](https://www.bluestacks.com/download.html)
- Установите **BlueStacks 5** (не NXT)
- Версия: **5.16.0 или выше**

### 4. Brawl Stars
- Установите через **Google Play Store** в BlueStacks
- Войдите в аккаунт Supercell ID (опционально)

### 5. GPU (Опционально, но рекомендуется)
- **NVIDIA GPU** с CUDA support
- Проверьте совместимость: [CUDA GPUs](https://developer.nvidia.com/cuda-gpus)
- Установите [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

---

## 🛠️ Step-by-Step Setup

### Step 1: Clone Repository

```bash
git clone https://github.com/wwmaxik/BrawlStarsBot.git
cd BrawlStarsBot
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Если ошибка с torch:**
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 4: Download YOLOv8 Model

⚠️ **Файл модели НЕ включён в репозиторий** (слишком большой)

Вам нужно:
1. Обучить свою YOLOv8 модель на скриншотах Brawl Stars, ИЛИ
2. Использовать предобученную модель (если есть)

Положите файл модели в:
```
yolov8_model/yolov8.pt
```

Или используйте OpenVINO модель для CPU:
```
yolov8_model/yolov8_openvino_model/
```

### Step 5: Configure BlueStacks

#### 5.1. Настройки окна
- Откройте BlueStacks
- Разрешение: **1920x1080** (рекомендуется)
- Окно должно быть **развернуто** (не в фуллскрин)

#### 5.2. Настройки управления
Настройте клавиши в BlueStacks:

| Клавиша | Действие в игре |
|---------|-----------------|
| W | Вперёд |
| A | Влево |
| S | Назад |
| D | Вправо |
| Space | Атака |
| E | Super Attack |
| F | Gadget |

#### 5.3. Настройки производительности
- **CPU:** 4 cores minimum
- **RAM:** 4GB minimum
- **GPU Mode:** NVIDIA (если есть GPU)
- **FPS:** 60

### Step 6: Launch Brawl Stars

1. Откройте Brawl Stars в BlueStacks
2. Перейдите в **Solo Showdown**
3. **НЕ нажимайте "Play"** — бот сделает это сам (если настроен)
4. Оставьте окно BlueStacks **открытым и видимым**

---

## 🎮 Running the Bot

### Option 1: Training (Recommended)

```bash
python rl_train.py
```

**Что происходит:**
1. Загружается существующая модель (если есть)
2. Создаётся среда `BrawlStarsYoloEnv`
3. Запускается Flask Dashboard (http://localhost:5000)
4. Начинается обучение RecurrentPPO

### Option 2: Web Dashboard Only

Если хотите только мониторить:

```bash
python -c "from dashboard import start_dashboard; start_dashboard(); import time; time.sleep(99999)"
```

Затем откройте: http://localhost:5000

### Option 3: TensorBoard Monitoring

В **новом терминале**:

```bash
tensorboard --logdir ./logs/tensorboard
```

Откройте: http://localhost:6006

---

## 📊 Understanding the Output

### Terminal Output (TUI)

```
┌─────────────────────────────────────────────────────────────┐
│ ⌛ 14:32:15    BRAWL STARS AI | MATCH #42    STEPS: 15234  │
├─────────────────────────────────────────────────────────────┤
│ 📜 Activity Stream    │  🎮 Game Data  │  🧠 Neural Network│
│ [14:32:15] MATCH #42  │  Match: 42     │  LR: 0.000100     │
│ [14:32:16] REWARD     │  Steps: 15234  │  Entropy: 1.234   │
│ [14:32:17] PENALTY    │  Duration: 45s │  Value: 0.567     │
│                       │  FPS: 15       │                   │
│                       │  Reward: +0.23 │                   │
├─────────────────────────────────────────────────────────────┤
│ MOVEMENT: [D W]  |  WEAPON: 🔥 FIRING  |  REWARD: +0.0234 ████│
└─────────────────────────────────────────────────────────────┘
```

### Log Messages

| Message | Meaning |
|---------|---------|
| `[REWARD] Precision fire` | Бот стреляет по цели (+0.02) |
| `[REWARD] BOX DESTROYED!` | Коробка уничтожена (+0.30) |
| `[REWARD] Stealth mode!` | Бот в кустах с врагом (+0.02) |
| `[PENALTY] DAMAGE TAKEN!` | Бот получил урон (-0.15) |
| `[PENALTY] Ammo wasted` | Стрельба в пустоту (-0.03) |
| `[FATAL] IN POISON!` | Бот в газе (-0.80) |
| `[BONUS] Survived with low HP!` | Выживание с низким HP (+0.5) |
| `MATCH END` | Матч завершён |

---

## 🔧 Troubleshooting

### ❌ "Bluestacks App Player not found"

**Решение:**
1. Проверьте что BlueStacks запущен
2. Откройте `constants.py` и измените `window_name`
3. Запустите для проверки:
```bash
python -c "import win32gui; print([w for w in win32gui.GetWindowText(win32gui.GetForegroundWindow())])"
```

### ❌ "CUDA not available"

**Решение:**
1. Проверьте GPU:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
2. Если `False`:
   - Установите [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
   - Или измените в `constants.py`: `nvidia_gpu = False`

### ❌ "ModuleNotFoundError: No module named 'gymnasium'"

**Решение:**
```bash
pip install -r requirements.txt
```

### ❌ Bot doesn't move

**Возможные причины:**
1. BlueStacks окно **не в фокусе** — кликните на окно
2. Настройки управления **не настроены** — проверьте WASD
3. YOLO модель **не загружена** — проверьте путь в `constants.py`

### ❌ Low FPS (< 10)

**Решение:**
1. Уменьшите разрешение BlueStacks до 1280x720
2. Включите GPU режим: `nvidia_gpu = True`
3. Уменьшите `YOLO_IMG_SIZE` в `constants.py`

### ❌ Bot walks into poison

**Это нормально** во время начального обучения! Бот учится избегать газа через penalty `-0.80`. После ~100 матчей должен научиться.

---

## 📈 Expected Training Timeline

| Steps | Matches | Expected Behavior |
|-------|---------|-------------------|
| 0-10k | 1-10 | Random movement, frequent deaths |
| 10k-50k | 10-50 | Learns basic movement |
| 50k-100k | 50-100 | Starts collecting boxes |
| 100k-500k | 100-500 | Learns to avoid poison |
| 500k-1M | 500-1000 | Basic combat strategies |
| 1M-2M | 1000-2000 | Advanced kiting, stealth |

⏱️ **Реальное время:** ~15 FPS = ~1M steps за **18-24 часа** непрерывного обучения

---

## 🎯 Tips for Better Training

1. **Используйте танковых бравлеров:**
   - Frank, Sam, Buster, El Primo
   - Больше HP = больше времени на обучение

2. **Выбирайте простые карты:**
   - Island Invasion (много кустов)
   - Без стен и препятствий

3. **Запустите на ночь:**
   - Обучение медленное, оставляйте на 8+ часов

4. **Мониторьте TensorBoard:**
   - Reward должен расти со временем
   - Если падает — уменьшите `learning_rate`

5. **Сохраняйте best модель:**
   - Автоматически сохраняется в `brawl_yolo_recurrent_ppo_best.zip`

---

## 🆘 Still Having Issues?

1. Check [README.md](README.md) for general info
2. Check [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup
3. Create an [Issue](https://github.com/wwmaxik/BrawlStarsBot/issues) with:
   - Error message
   - Screenshot
   - `constants.py` settings
   - BlueStacks version

---

Happy Training! 🤖🎮
