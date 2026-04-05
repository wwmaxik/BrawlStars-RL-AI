# 🤝 Contributing Guide

Спасибо за интерес к проекту **Brawl Stars RL Bot**! Мы приветствуем любой вклад: баг-репорты, фичи, документацию, тесты.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Features](#suggesting-features)
  - [Pull Requests](#pull-requests)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Commit Messages](#commit-messages)
- [Testing](#testing)

---

## 📜 Code of Conduct

- Будьте уважительны к другим контрибьюторам
- Конструктивная критика приветствуется
- Помогайте новичкам в проекте

---

## 🚀 Getting Started

1. **Форкните репозиторий**
2. **Склонируйте свою копию:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/BrawlStarsBot.git
   cd BrawlStarsBot
   ```

3. **Создайте ветку для изменений:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Внесите изменения и закоммитьте:**
   ```bash
   git add .
   git commit -m "feat: add awesome feature"
   ```

5. **Запуште и создайте Pull Request:**
   ```bash
   git push origin feature/your-feature-name
   ```

---

## 🐛 Reporting Bugs

Создайте **Issue** с标签 `bug` и включите:

- **Краткое описание** проблемы
- **Шаги воспроизведения**
- **Ожидаемое поведение** vs **Фактическое поведение**
- **Скриншоты/Логи** (если есть)
- **Окружение:**
  - OS: Windows 10/11
  - Python: 3.11.6
  - BlueStacks версия
  - GPU (если используется CUDA)

---

## 💡 Suggesting Features

Создайте **Issue** с标签 `enhancement` и опишите:

- **Что** вы хотите добавить
- **Зачем** это нужно (use case)
- **Как** это может быть реализовано (опционально)

---

## 🔀 Pull Requests

### Перед отправкой PR:

- [ ] Код следует [Coding Standards](#coding-standards)
- [ ] Все тесты проходят (`pytest tests/ -v`)
- [ ] Добавлены тесты для новых фич
- [ ] Обновлена документация (если нужно)
- [ ] Изменения описаны в описании PR

### Название PR:

Используйте [Conventional Commits](#commit-messages):
- `feat: add poison direction to observation`
- `fix: correct health normalization`
- `docs: update README with new architecture`

### Описание PR:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change)
- [ ] New feature (non-breaking change)
- [ ] Breaking change (affects existing functionality)
- [ ] Documentation update

## Testing
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] Manually tested on BlueStacks

## Screenshots (if applicable)
Add screenshots of new features or UI changes
```

---

## 🛠️ Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/BrawlStarsBot.git
cd BrawlStarsBot

# Install dependencies
pip install -r requirements.txt

# Install dev tools
pip install pytest pytest-cov black flake8

# Run tests
pytest tests/ -v

# Format code
black .

# Lint
flake8 .
```

---

## 📝 Coding Standards

### Style

- **Black** для форматирования
- **flake8** для линтинга
- Следуйте **PEP 8**

### Naming Conventions

| Тип | Convention | Example |
|-----|------------|---------|
| Classes | PascalCase | `BrawlStarsYoloEnv` |
| Functions | snake_case | `calculate_reward()` |
| Variables | snake_case | `match_duration` |
| Constants | UPPER_SNAKE_CASE | `WINDOW_NAME` |
| Private methods | _leading_underscore | `_get_health()` |

### Type Hints

Используйте type hints где возможно:

```python
def calculate_reward(self, duration: float, boxes: int) -> float:
    """Calculate match reward.
    
    Args:
        duration: Match duration in seconds
        boxes: Number of boxes destroyed
    
    Returns:
        Calculated reward value
    """
    reward = 0.01 * (1 + duration / 60.0)
    reward += boxes * 0.3
    return reward
```

### Docstrings

Используйте Google-style docstrings:

```python
def function_name(param1: str, param2: int) -> bool:
    """Short description of function.
    
    Longer description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When something goes wrong
    """
    pass
```

---

## 💬 Commit Messages

Используйте **Conventional Commits**:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types:

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation changes |
| `style` | Code style changes (formatting, no logic) |
| `refactor` | Code refactoring |
| `test` | Test changes |
| `perf` | Performance improvements |
| `ci` | CI/CD changes |
| `chore` | Maintenance/chore tasks |

### Examples:

```bash
feat(observation): add health percentage to observation vector
fix(reward): correct poison penalty calculation
docs(readme): update architecture diagram
test(env): add unit tests for action space
perf(detection): optimize YOLO inference speed
refactor(env): simplify reward calculation
```

---

## 🧪 Testing

### Запуск тестов:

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=. --cov-report=html

# Specific test file
pytest tests/test_rl_env.py -v

# Specific test
pytest tests/test_rl_env.py::TestRewardFunction -v
```

### Writing Tests:

- Тестируйте **публичные API**
- Используйте **mock** для внешних зависимостей
- Один тест = одна проверка
- Название теста описывает **что** тестируется

```python
def test_poison_direction_normalization(self):
    """Poison direction vector should be normalized when gas is detected."""
    dx, dy = calc_poison_direction({'w': True, 'd': True})
    expected = 1.0 / (2 ** 0.5)
    assert abs(dx - expected) < 0.001
    assert abs(dy - expected) < 0.001
```

---

## ❓ Questions?

- Создайте **Discussion** для вопросов
- Или напишите в **Issues** с标签 `question`

---

Спасибо за вклад! 🎉
