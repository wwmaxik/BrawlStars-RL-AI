# 🤖 Brawl Stars RL Bot

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3116/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.1-red.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-8.0.217-purple.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-AGPLv3-orange.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)](https://github.com/wwmaxik/BrawlStarsBot)

> Автономный RL-агент для игры в **Brawl Stars (Solo Showdown)** через эмулятор BlueStacks. Сочетает **YOLOv8** для компьютерного зрения с **RecurrentPPO (LSTM)** для принятия решений в реальном времени.

---

## ✨ Возможности

| Фича | Описание |
|------|----------|
| 🎯 **YOLOv8 Detection** | Обнаружение игроков, врагов, кустов и коробок в реальном времени |
| 🧠 **RecurrentPPO (LSTM)** | RL-агент с памятью для принятия последовательных решений |
| 🖼️ **Frame Stacking** | Стекирование 4 кадров для понимания динамики игры |
| 🎮 **MultiDiscrete Actions** | Стабильное пространство действий (движение, атака, супер, гаджет) |
| ☠️ **Poison Avoidance** | Автоматическое обнаружение и избегание ядовитого газа |
| 💚 **Health Monitoring** | Отслеживание HP игрока для принятия осторожных решений |
| 📊 **TensorBoard** | Полное логирование метрик обучения |
| 🌐 **Web Dashboard** | Живая визуализация через Flask (скриншоты + статистика) |
| 🎨 **Rich TUI** | Красивый терминальный интерфейс с цветными логами |
| ⚡ **Multi-threading** | Асинхронный захват экрана, detection и ввод |

---

## 🏗️ Архитектура

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Brawl Stars RL Bot                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  BlueStacks  │───▶│ WindowCapture│───▶│   YOLOv8 Detector  │  │
│  │   Emulator   │    │  (60 FPS)    │    │   (15-30 FPS)      │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│                                      │                              │
│                                      ▼                              │
│                          ┌──────────────────────┐                  │
│                          │  BrawlStarsYoloEnv   │                  │
│                          │  ┌────────────────┐  │                  │
│                          │  │ Frame Stacking │  │                  │
│                          │  │  Health Detect │  │                  │
│                          │  │ Poison Monitor │  │                  │
│                          │  │ Reward Engine  │  │                  │
│                          │  └────────────────┘  │                  │
│                          └──────────────────────┘                  │
│                                      │                              │
│                                      ▼                              │
│                    ┌─────────────────────────────────┐             │
│                    │     RecurrentPPO (LSTM)         │             │
│                    │  ┌───────────────────────────┐  │             │
│                    │  │ CNN: Vision Encoder       │  │             │
│                    │  │ MLP: Vector Encoder       │  │             │
│                    │  │ LSTM: Temporal Memory     │  │             │
│                    │  │ Policy/Value Heads        │  │             │
│                    │  └───────────────────────────┘  │             │
│                    └─────────────────────────────────┘             │
│                                      │                              │
│                                      ▼                              │
│                          ┌──────────────────────┐                  │
│                          │  PyAutoGUI Input     │                  │
│                          │  WASD + Space+E+F    │                  │
│                          └──────────────────────┘                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Установка

### Требования
- **ОС:** Windows 10/11
- **Python:** 3.11.6
- **BlueStacks 5:** [Скачать](https://www.bluestacks.com/download.html)
- **GPU:** NVIDIA GPU с CUDA (опционально, но рекомендуется)

### Шаги установки

1. **Клонируйте репозиторий:**
```bash
git clone https://github.com/wwmaxik/BrawlStarsBot.git
cd BrawlStarsBot
```

2. **Установите зависимости:**
```bash
pip install -r requirements.txt
```

3. **Подготовьте BlueStacks:**
- Установите BlueStacks 5
- Установите Brawl Stars
- Настройте управление (WASD, Space, E, F)
- Запустите Solo Showdown

4. **Запустите обучение:**
```bash
python rl_train.py
```

5. **Откройте Web Dashboard:**
Перейдите на [http://localhost:5000](http://localhost:5000)

6. **Мониторинг через TensorBoard:**
```bash
tensorboard --logdir ./logs/tensorboard
```

---

## 🧠 Observation Space

### Image Input
- **Shape:** `(12, 84, 84)` — 4 кадра × 3 RGB канала
- **Описание:** Стекирование последних 4 кадров для понимания движения

### Vector Input (26 элементов)
| Индекс | Описание | Диапазон |
|--------|----------|----------|
| 0 | Damage taken flag | {0, 1} |
| 1 | Health % (нормализовано) | [-1, 1] |
| 2 | Количество коробок | [0, 1] |
| 3 | Количество врагов | [0, 1] |
| 4-6 | Ближайший враг (dx, dy, dist) | [-1, 1] |
| 7-9 | Враг #2 (dx, dy, dist) | [-1, 1] |
| 10-12 | Ближайшая коробка (dx, dy, dist) | [-1, 1] |
| 13-15 | Ближайший куст (dx, dy, dist) | [-1, 1] |
| 16-19 | Стены (W, S, A, D) | {0, 1} |
| 20 | Poison flag | {0, 1} |
| 21-22 | Poison direction (vec_x, vec_y) | [-1, 1] |
| 24 | Время матча (нормализовано) | [0, 1] |
| 25 | Среднее расстояние до врагов | [0, 1] |

---

## 🎮 Action Space

**MultiDiscrete([3, 3, 2, 2, 2])** = 72 возможные комбинации

| Индекс | Действие | Значения |
|--------|----------|----------|
| 0 | Movement X | 0=Left, 1=None, 2=Right |
| 1 | Movement Y | 0=Up, 1=None, 2=Down |
| 2 | Attack | 0=None, 1=Space |
| 3 | Super | 0=None, 1=E |
| 4 | Gadget | 0=None, 1=F |

---

## 💰 Reward Function

| Событие | Награда/Штраф |
|---------|--------------|
| Выживание (за шаг) | `+0.01 * (1 + duration/60)` |
| Приближение к коробке | `+0.005` |
| Уничтожение коробки | `+0.30` |
| Стрельба по целям | `+0.02` |
| Стрельба в пустоту | `-0.03` |
| Слишком близко к врагу | `-0.01` |
| Безопасная дистанция | `+0.005` |
| Stealth (кусты + враг) | `+0.02` |
| Получение урона | `-0.15` |
| Нахождение в газе | `-0.80` |
| Низкое здоровье (<30%) | `-0.05` |
| Финальный бонус | до `+2.0` |
| Бонус за кубы | `+0.2` за коробку |

---

## ⚙️ Hyperparameters

| Параметр | Значение |
|----------|----------|
| Algorithm | RecurrentPPO (LSTM) |
| Policy | MultiInputLstmPolicy |
| n_steps | 4096 |
| batch_size | 128 |
| learning_rate | 1e-4 |
| gamma | 0.995 |
| ent_coef | 0.05 |
| clip_range | 0.2 |
| vf_coef | 0.5 |
| max_grad_norm | 0.5 |
| gae_lambda | 0.95 |
| n_epochs | 10 |
| lstm_hidden_size | 256 |
| total_timesteps | 2,000,000 |

---

## 📁 Структура проекта

```
BrawlStarsBot/
├── constants.py              # Глобальные настройки
├── rl_env.py                 # Gymnasium среда
├── rl_train.py               # Скрипт обучения
├── dashboard.py              # Flask веб-дашборд
├── requirements.txt          # Python зависимости
├── LICENSE                   # MIT License
├── README.md                 # Этот файл
├── AGENT.md                  # AI Agent контекст
├── CONTRIBUTING.md           # Руководство по контрибьюту
│
├── modules/
│   ├── windowcapture.py      # Захват экрана (win32gui)
│   ├── detection.py          # YOLOv8 инференс
│   ├── tui.py                # Terminal UI (Rich)
│   └── print.py              # Цветной вывод
│
├── tests/
│   ├── test_rl_env.py        # Unit тесты
│   └── test_integration.py   # Интеграционные тесты
│
├── yolov8_model/
│   └── yolov8.pt             # YOLOv8 модель (не в git)
│
├── logs/
│   └── tensorboard/          # TensorBoard логи (не в git)
│
└── .gitignore
```

---

## 🧪 Тестирование

```bash
# Запустить все тесты
pytest tests/ -v

# Запустить с покрытием
pytest tests/ -v --cov=.
```

---

## 📊 Мониторинг обучения

### Web Dashboard
- **URL:** http://localhost:5000
- **Функции:**
  - Прямой эфир из BlueStacks
  - AI Vision (84x84 CNN Input)
  - Живая статистика и логи

### TensorBoard
```bash
tensorboard --logdir ./logs/tensorboard
```
- **URL:** http://localhost:6006
- **Метрики:**
  - Reward per episode
  - Learning Rate
  - Entropy/Value Loss
  - Clip Fraction
  - Explained Variance

---

## 🤝 Контрибьют

См. [CONTRIBUTING.md](CONTRIBUTING.md) для руководства по контрибьюту.

---

## ⚠️ Дисклеймер

> **Использование бота может привести к потере трофеев!** Бот предназначен для обучения и исследования в области RL. Используйте на свой страх и риск.

---

## 📚 Источники вдохновения

- [OpenCV Object Detection in Games Python Tutorial](https://www.youtube.com/watch?v=KecMlLUuiE4&list=PL1m2M8LQlzfKtkKq2lK5xko4X-8EZzFPI) by "Learn Code By Gaming"
- [How To Train YOLOv5 For Recognizing Game Objects In Real-Time](https://betterprogramming.pub/how-to-train-yolov5-for-recognizing-custom-game-objects-in-real-time-9d78369928a8) by Jes Fink-Jensen

---

## 📄 Лицензия

GNU Affero General Public License v3.0 (AGPLv3). См. [LICENSE](LICENSE) для подробностей.

---

## 🌟 Star History

Если проект полезен — поставь ⭐️!
