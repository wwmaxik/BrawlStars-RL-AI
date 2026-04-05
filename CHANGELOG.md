# 📝 Changelog

Все значимые изменения проекта Brawl Stars RL Bot.

Формат: [Semantic Versioning](https://semver.org/lang/ru/)

---

## [2.0.0] - 2026-04-04

### 🚀 Добавлено
- **Расширенный Observation Space** (15 → 26 измерений)
  - Health percentage игрока
  - Количество врагов и коробок
  - Второй ближайший враг
  - Направление ядовитого газа (вектор)
  - Время матча (нормализованное)
  - Среднее расстояние до всех врагов
- **Frame Stacking** (4 последовательных кадра)
  - AI теперь понимает движение и динамику
  - Image shape: `(3, 84, 84)` → `(12, 84, 84)`
- **MultiDiscrete Action Space**
  - Замена continuous на discrete: `MultiDiscrete([3, 3, 2, 2, 2])`
  - Более стабильное обучение
- **Health Detection**
  - Мониторинг HP через отслеживание урона (100% → -12% за удар)
  - Штраф за низкое здоровье (<30%)
- **Idle Penalty**
  - Предотвращение "Idle Disconnect" от Supercell
  - Нарастающий штраф после 10 шагов без движения: `-0.05 * (idle-10)`
  - Индикатор idle в TUI (⏸️ Idle: N steps ⚠️)
- **Auto-Reload on Disconnect**
  - Автоматическое обнаружение окна "Idle Disconnect"
  - Детекция по серой плашке + бирюзовой кнопке RELOAD
  - Автоматическое нажатие R для перезагрузки матча
- **TensorBoard Integration**
  - Полное логирование всех метрик
  - Отслеживание лучшей модели
- **Best Model Tracking**
  - Автоматическое сохранение `brawl_yolo_recurrent_ppo_best.zip`
- **Reward Shaping**
  - Stealth bonus (+0.02 за кусты с врагом)
  - Cube bonus в конце матча (+0.2 за коробку)
  - Увеличен survival reward (0.001 → 0.01)
  - Увеличен poison penalty (0.5 → 0.8)
  - Low HP penalty (-0.05)
- **Unit Tests**
  - `tests/test_rl_env.py` — observation, actions, rewards
  - `tests/test_integration.py` — интеграция и edge cases
- **Documentation**
  - `README.md` — полная переработка
  - `SETUP_GUIDE.md` — пошаговая инструкция
  - `CONTRIBUTING.md` — руководство для контрибьюторов
  - `AGENT.md` — обновлён для v2.0
  - `.env.example` — шаблон конфигурации
  - `CHANGELOG.md` — этот файл

### ⚙️ Изменено
- **Hyperparameters** (оптимизированы для стабильности)
  - `n_steps`: 2048 → 4096
  - `batch_size`: 64 → 128
  - `learning_rate`: 3e-4 → 1e-4
  - `gamma`: 0.99 → 0.995
  - `ent_coef`: 0.02 → 0.05
  - `total_timesteps`: 1M → 2M
  - `lstm_hidden_size`: default → 256
- **Network Architecture**
  - Добавлены `policy_kwargs` с расширенными слоями
  - Vision: [128, 128]
  - Policy/Value: [256, 128]

### 🗑️ Удалено
- Старые модели (`brawl_yolo_recurrent_ppo.zip`, `*_best.zip`)
- Временные файлы (`debug_screenshot.png`, `err.txt`, `error.txt`, `out.txt`)
- Устаревшие файлы из v1.0

### 📝 Изменено
- **Лицензия**: MIT → AGPLv3

### ⚠️ Breaking Changes
- **Несовместимо с моделями v1.0**
  - Изменён observation space
  - Изменён action space
  - Требуется обучение с нуля

---

## [1.0.0] - 2024-XX-XX

### 🚀 Добавлено
- Базовая RL среда для Brawl Stars
- YOLOv8 detection (Player, Bush, Enemy, Cubebox)
- RecurrentPPO с LSTM
- Continuous action space
- BlueStacks интеграция через PyAutoGUI
- Web Dashboard (Flask)
- Terminal UI (Rich)
- Poison gas avoidance
- Damage monitoring
- Multi-threading architecture

### 📝 Известные проблемы
- Маленький observation space (15 измерений)
- Нестабильное обучение с continuous actions
- Нет frame stacking
- Нет health monitoring
- Минимальная документация

---

## Версии

| Версия | Дата | Описание |
|--------|------|----------|
| 2.0.0 | 2026-04-04 | Полная переработка архитектуры |
| 1.0.0 | 2024-XX-XX | Первоначальная версия |

---

[2.0.0]: https://github.com/wwmaxik/BrawlStarsBot/releases/tag/v2.0.0
[1.0.0]: https://github.com/wwmaxik/BrawlStarsBot/releases/tag/v1.0.0
