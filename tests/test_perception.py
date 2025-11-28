import unittest
import numpy as np
import torch
from perception import Perception

class TestPerception(unittest.TestCase):
    def setUp(self):
        # Initialize with default device (CUDA if available)
        self.perception = Perception()  

    def test_obs_shape(self):
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        obs = self.perception.get_obs(frame)
        self.assertEqual(obs.shape, (9,))
        self.assertTrue(np.isfinite(obs).all())

    def test_velocity_update(self):
        # Simulate movement
        # Frame 1: Center
        self.perception.player_history.clear()
        self.perception.compute_velocity(320, 320)
        
        import time
        time.sleep(0.02) # Ensure time advances for dt check
        
        # Frame 2: Moved right 10 pixels
        vx, vy = self.perception.compute_velocity(330, 320)
        
        # vx should be positive (10/640)
        self.assertGreater(vx, 0)
        self.assertEqual(vy, 0)

    def test_obs_values_bounded(self):
        frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        obs = self.perception.get_obs(frame)
        # Check bounds [-1, 1] roughly
        # dist_goal might be slightly > 1 but clipped in code
        self.assertTrue((obs >= -1.0).all())
        self.assertTrue((obs <= 1.0).all())

if __name__ == '__main__':
    unittest.main()
