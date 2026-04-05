"""
Unit tests for Brawl Stars RL Environment
Tests: observation space, reward function, poison detection, action space
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from constants import Constants


class TestObservationSpace:
    """Тесты для observation space"""
    
    def test_vector_size(self):
        """Проверка что размер vector observation соответствует константе"""
        assert Constants.vector_size == 26
    
    def test_frame_stack_size(self):
        """Проверка что frame stack размер корректный"""
        assert Constants.frame_stack == 4
        expected_channels = 3 * Constants.frame_stack
        assert expected_channels == 12
    
    def test_observation_vector_bounds(self):
        """Проверка что все элементы observation vector в пределах [-1, 1]"""
        # Создаём mock vector
        vec = np.zeros(Constants.vector_size, dtype=np.float32)
        
        # [0] damage flag: 0 или 1
        vec[0] = 1.0
        assert 0.0 <= vec[0] <= 1.0
        
        # [1] health: нормализовано в [-1, 1]
        vec[1] = 0.5  # 75% здоровья
        assert -1.0 <= vec[1] <= 1.0
        
        # [2] box count: [0, 1]
        vec[2] = 0.3  # 3 коробки из 10
        assert 0.0 <= vec[2] <= 1.0
        
        # [3] enemy count: [0, 1]
        vec[3] = 0.5  # 4 врага из 8
        assert 0.0 <= vec[3] <= 1.0
        
        # [4:7] ближайший враг (dx, dy, dist)
        vec[4:7] = [0.3, -0.2, 0.5]
        assert all(-1.0 <= v <= 1.0 for v in vec[4:7])
        
        # [20] poison flag
        vec[20] = 1.0
        assert 0.0 <= vec[20] <= 1.0
        
        # [21:23] poison direction
        vec[21] = 0.707
        vec[22] = -0.707
        assert all(-1.0 <= v <= 1.0 for v in vec[21:23])
        
        # [24] time in match
        vec[24] = 0.5  # половина матча
        assert 0.0 <= vec[24] <= 1.0
        
        # [25] avg enemy distance
        vec[25] = 0.6
        assert 0.0 <= vec[25] <= 1.0
    
    def test_image_stacked_shape(self):
        """Проверка что stacked image имеет правильную форму"""
        img_size = Constants.img_size
        channels = 3 * Constants.frame_stack
        expected_shape = (channels, img_size, img_size)
        assert expected_shape == (12, 84, 84)


class TestActionSpace:
    """Тесты для action space"""
    
    def test_multi_discrete_shape(self):
        """Проверка что MultiDiscrete action space имеет правильную форму"""
        # [move_x, move_y, attack, super, gadget]
        # [3, 3, 2, 2, 2]
        expected_spaces = [3, 3, 2, 2, 2]
        assert len(expected_spaces) == 5
        
        # Проверяем что каждое значение корректно
        assert expected_spaces[0] == 3  # left, none, right
        assert expected_spaces[1] == 3  # up, none, down
        assert expected_spaces[2] == 2  # no attack, attack
        assert expected_spaces[3] == 2  # no super, super
        assert expected_spaces[4] == 2  # no gadget, gadget
    
    def test_action_decoding(self):
        """Проверка декодирования действий"""
        # Пример: [2, 0, 1, 0, 1] = right, up, attack, no super, gadget
        action = [2, 0, 1, 0, 1]
        
        move_x, move_y, attack, spr, gdgt = action
        
        # Проверяем bounds
        assert 0 <= move_x <= 2
        assert 0 <= move_y <= 2
        assert 0 <= attack <= 1
        assert 0 <= spr <= 1
        assert 0 <= gdgt <= 1
        
        # Проверяем логику movement
        expected_keys = set()
        if move_x == 0: expected_keys.add('a')
        elif move_x == 2: expected_keys.add('d')
        
        if move_y == 0: expected_keys.add('w')
        elif move_y == 2: expected_keys.add('s')
        
        assert 'd' in expected_keys  # move_x = 2
        assert 'w' in expected_keys  # move_y = 0


class TestRewardFunction:
    """Тесты для reward function"""
    
    def test_survival_reward(self):
        """Проверка базового reward за выживание"""
        # Формула: 0.01 * (1 + match_duration / 60.0)
        def calc_survival_reward(duration):
            return 0.01 * (1 + duration / 60.0)
        
        # Начало матча
        assert calc_survival_reward(0) == 0.01
        
        # Через 1 минуту
        assert calc_survival_reward(60) == 0.02
        
        # Через 2 минуты
        assert calc_survival_reward(120) == 0.03
    
    def test_box_destroyed_reward(self):
        """Проверка reward за уничтожение коробки"""
        box_reward = 0.3
        assert box_reward > 0
        assert box_reward == 0.3
    
    def test_poison_penalty(self):
        """Проверка penalty за нахождение в газе"""
        poison_penalty = 0.8
        assert poison_penalty < 0
        assert abs(poison_penalty) == 0.8
        # Должно быть больше чем damage penalty
        assert abs(poison_penalty) > 0.15
    
    def test_damage_penalty(self):
        """Проверка penalty за получение урона"""
        damage_penalty = 0.15
        assert abs(damage_penalty) == 0.15
    
    def test_final_match_reward(self):
        """Проверка финального reward за матч"""
        # Формула: min(match_duration / 120.0, 2.0)
        def calc_final_reward(duration):
            return min(duration / 120.0, 2.0)
        
        # Короткий матч (30 сек)
        assert calc_final_reward(30) == 0.25
        
        # Средний матч (60 сек)
        assert calc_final_reward(60) == 0.5
        
        # Долгий матч (120 сек)
        assert calc_final_reward(120) == 1.0
        
        # Очень долгий матч (180 сек) - capped at 2.0
        assert calc_final_reward(180) == 2.0
    
    def test_cube_bonus(self):
        """Проверка бонуса за кубы"""
        # Формула: boxes_destroyed * 0.2
        def calc_cube_bonus(count):
            return count * 0.2
        
        assert calc_cube_bonus(0) == 0.0
        assert calc_cube_bonus(1) == 0.2
        assert calc_cube_bonus(5) == 1.0
        assert calc_cube_bonus(10) == 2.0


class TestPoisonDetection:
    """Тесты для poison detection"""
    
    def test_poison_direction_calculation(self):
        """Проверка расчёта направления газа"""
        def calc_poison_direction(zones):
            vec_x, vec_y = 0.0, 0.0
            if zones.get('w'): vec_y = -1.0
            if zones.get('s'): vec_y = 1.0
            if zones.get('a'): vec_x = -1.0
            if zones.get('d'): vec_x = 1.0
            
            mag = (vec_x**2 + vec_y**2) ** 0.5
            if mag > 0:
                vec_x /= mag
                vec_y /= mag
            return vec_x, vec_y
        
        # Газ только сверху
        dx, dy = calc_poison_direction({'w': True})
        assert dx == 0.0
        assert dy == -1.0
        
        # Газ снизу и справа
        dx, dy = calc_poison_direction({'s': True, 'd': True})
        expected_val = 1.0 / (2**0.5)
        assert abs(dx - expected_val) < 0.001
        assert abs(dy - expected_val) < 0.001
        
        # Нет газа
        dx, dy = calc_poison_direction({})
        assert dx == 0.0
        assert dy == 0.0
        
        # Газ со всех сторон
        dx, dy = calc_poison_direction({'w': True, 's': True, 'a': True, 'd': True})
        # W+S компенсируют, A+D компенсируют
        assert abs(dx) < 0.001
        assert abs(dy) < 0.001
    
    def test_poison_flag(self):
        """Проверка poison flag в observation vector"""
        # [20] poison flag
        poison_zones = {'w': False, 's': False, 'a': False, 'd': False}
        in_poison = any(poison_zones.values())
        assert not in_poison
        
        poison_zones['w'] = True
        in_poison = any(poison_zones.values())
        assert in_poison


class TestHealthDetection:
    """Тесты для health detection"""
    
    def test_health_normalization(self):
        """Проверка нормализации здоровья"""
        # health_pct -> vec_obs[1] = health_pct * 2.0 - 1.0
        def normalize_health(pct):
            return pct * 2.0 - 1.0
        
        def denormalize_health(val):
            return (val + 1.0) / 2.0
        
        # 100% здоровья
        assert normalize_health(1.0) == 1.0
        assert denormalize_health(1.0) == 1.0
        
        # 50% здоровья
        assert normalize_health(0.5) == 0.0
        assert denormalize_health(0.0) == 0.5
        
        # 0% здоровья
        assert normalize_health(0.0) == -1.0
        assert denormalize_health(-1.0) == 0.0
        
        # 25% здоровья
        assert normalize_health(0.25) == -0.5
        assert denormalize_health(-0.5) == 0.25


class TestConstants:
    """Тесты для констант"""
    
    def test_img_size(self):
        """Проверка размера изображения"""
        assert Constants.img_size == 84
    
    def test_classes_count(self):
        """Проверка количества классов"""
        assert len(Constants.classes) == 4
        assert "Player" in Constants.classes
        assert "Bush" in Constants.classes
        assert "Enemy" in Constants.classes
        assert "Cubebox" in Constants.classes
    
    def test_thresholds_count(self):
        """Проверка что количество порогов = количеству классов"""
        assert len(Constants.threshold) == len(Constants.classes)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
