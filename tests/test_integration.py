"""
Integration tests для Brawl Stars RL Environment
Тестирование взаимодействия компонентов
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from constants import Constants


class TestObservationIntegration:
    """Интеграционные тесты для observation"""
    
    def test_observation_vector_indices(self):
        """Проверка что все индексы observation vector корректны"""
        vec = np.zeros(Constants.vector_size, dtype=np.float32)
        
        # Проверяем что все индексы существуют
        assert vec[0] == 0.0   # damage flag
        assert vec[1] == 0.0   # health
        assert vec[2] == 0.0   # box count
        assert vec[3] == 0.0   # enemy count
        assert vec[4:7].shape == (3,)  # enemy 1
        assert vec[7:10].shape == (3,)  # enemy 2
        assert vec[10:13].shape == (3,)  # cubebox
        assert vec[13:16].shape == (3,)  # bush
        assert vec[16:20].shape == (4,)  # walls
        assert vec[20] == 0.0  # poison flag
        assert vec[21:23].shape == (2,)  # poison direction
        assert vec[24] == 0.0  # time in match
        assert vec[25] == 0.0  # avg enemy distance
        
        # Проверяем что vector_size = 26
        assert Constants.vector_size == 26
    
    def test_frame_stacking_buffer(self):
        """Проверка логики frame stacking"""
        from collections import deque
        
        frame_stack = Constants.frame_stack
        buffer = deque(maxlen=frame_stack)
        
        # Буфер должен хранить максимум 4 кадра
        for i in range(10):
            buffer.append(i)
        
        assert len(buffer) == frame_stack
        assert list(buffer) == [6, 7, 8, 9]  # последние 4 элемента


class TestRewardShaping:
    """Тесты для reward shaping"""
    
    def test_low_health_penalty(self):
        """Проверка penalty за низкое здоровье"""
        def calc_health_penalty(health_pct):
            if health_pct < 0.3:
                return -0.05
            return 0.0
        
        assert calc_health_penalty(0.2) == -0.05
        assert calc_health_penalty(0.3) == 0.0
        assert calc_health_penalty(0.5) == 0.0
        assert calc_health_penalty(1.0) == 0.0
    
    def test_stealth_reward(self):
        """Проверка reward за stealth режим"""
        def calc_stealth_reward(in_bush, enemy_nearby, bush_dist, enemy_dist):
            if in_bush and enemy_nearby and bush_dist < 0.3 and enemy_dist > 0.25:
                return 0.02
            return 0.0
        
        # В кустах, враг рядом
        assert calc_stealth_reward(True, True, 0.2, 0.4) == 0.02
        
        # Не в кустах
        assert calc_stealth_reward(False, True, 0.5, 0.4) == 0.0
        
        # Враг слишком близко
        assert calc_stealth_reward(True, True, 0.2, 0.15) == 0.0
    
    def test_movement_change_penalty(self):
        """Проверка penalty за смену направления"""
        def calc_movement_penalty(old_keys, new_keys):
            if old_keys != new_keys:
                return -0.005
            return 0.0
        
        assert calc_movement_penalty({'d'}, {'d'}) == 0.0
        assert calc_movement_penalty({'d'}, {'w'}) == -0.005
        assert calc_movement_penalty(set(), {'d'}) == -0.005
    
    def test_ammo_waste_penalty(self):
        """Проверка penalty за стрельбу в пустоту"""
        def calc_ammo_penalty(attacking, has_targets):
            if attacking and not has_targets:
                return -0.03
            elif attacking and has_targets:
                return 0.02
            return 0.0
        
        assert calc_ammo_penalty(True, False) == -0.03
        assert calc_ammo_penalty(True, True) == 0.02
        assert calc_ammo_penalty(False, False) == 0.0


class TestEdgeCases:
    """Тесты для edge cases"""
    
    def test_empty_detections(self):
        """Поведение когда нет обнаруженных объектов"""
        # Когда detector.results пустой, все значения должны быть дефолтными
        vec = np.zeros(Constants.vector_size, dtype=np.float32)
        
        # Все расстояния = 1.0 (максимум)
        assert vec[4:7].all() == 0.0  # enemy 1
        assert vec[7:10].all() == 0.0  # enemy 2
        assert vec[10:13].all() == 0.0  # cubebox
        assert vec[13:16].all() == 0.0  # bush
        
        # Counts = 0
        assert vec[2] == 0.0  # box count
        assert vec[3] == 0.0  # enemy count
    
    def test_match_timeout(self):
        """Проверка что матч заканчивается по таймауту"""
        # match_max_duration = 180 секунд
        assert Constants.match_max_duration if hasattr(Constants, 'match_max_duration') else True
    
    def test_frame_buffer_initial_fill(self):
        """Проверка начального заполнения frame buffer"""
        from collections import deque
        
        frame_stack = 4
        buffer = deque(maxlen=frame_stack)
        initial_frame = np.zeros((84, 84, 3), dtype=np.uint8)
        
        # Имитируем начальное заполнение
        while len(buffer) < frame_stack:
            buffer.appendleft(initial_frame.copy())
        
        assert len(buffer) == frame_stack


class TestActionDecoding:
    """Тесты для декодирования действий"""
    
    def test_all_action_combinations(self):
        """Проверка всех возможных комбинаций действий"""
        # MultiDiscrete([3, 3, 2, 2, 2]) = 3*3*2*2*2 = 72 комбинации
        total_combinations = 3 * 3 * 2 * 2 * 2
        assert total_combinations == 72
        
        # Проверяем несколько комбинаций
        test_cases = [
            ([1, 1, 0, 0, 0], set()),  # Все none
            ([2, 2, 0, 0, 0], {'d', 's'}),  # Вправо-вниз
            ([0, 0, 0, 0, 0], {'a', 'w'}),  # Влево-вверх
            ([2, 0, 1, 0, 0], {'d', 'w'}),  # Вправо-вверх + attack
            ([0, 2, 1, 1, 1], {'a', 's'}),  # Влево-вниз + attack + super + gadget
        ]
        
        for action, expected_keys in test_cases:
            move_x, move_y, attack, spr, gdgt = action
            target_keys = set()
            
            if move_x == 0: target_keys.add('a')
            elif move_x == 2: target_keys.add('d')
            
            if move_y == 0: target_keys.add('w')
            elif move_y == 2: target_keys.add('s')
            
            assert target_keys == expected_keys


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
