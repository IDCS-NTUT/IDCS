"""Deterministic return cadence checks, with no decoder, clocks or sockets."""
import unittest
from common.video_health import ReturnVideoHealth
from rpi.display.status import SystemStatus


class ReturnCadenceTests(unittest.TestCase):
    def stream(self,health,fps,duration,start=0):
        warnings=[]
        for step in range(round(duration*100)+1):
            now=start+step/100
            # Precise frame schedule independent of UI redraw cadence.
            if step==0 or int(step*fps/100)!=int((step-1)*fps/100):
                health.mark(now)
            warnings.append(health.warning(now))
        return warnings

    def test_startup_and_threshold_are_not_false_alarms(self):
        for fps in (20,30,60):
            with self.subTest(fps=fps):
                health=ReturnVideoHealth()
                self.assertNotIn('LOW VIDEO FPS',self.stream(health,fps,4))

    def test_slow_video_requires_a_continuous_second_then_recovers(self):
        health=ReturnVideoHealth()
        warnings=self.stream(health,15,3)
        self.assertNotIn('LOW VIDEO FPS',warnings[:201])
        self.assertIn('LOW VIDEO FPS',warnings[201:])
        recovered=self.stream(health,30,2,start=3.01)
        self.assertIsNone(recovered[-1])

    def test_brief_dip_and_ui_redraws_do_not_hide_low_rate(self):
        health=ReturnVideoHealth()
        self.stream(health,30,2)
        for step in range(1,31):
            self.assertNotEqual(health.warning(2+step/100),'LOW VIDEO FPS')
        self.stream(health,30,2,start=2.31)
        self.assertIsNone(health.warning(4.31))
        health=ReturnVideoHealth()
        # 100 redraws/s still means 10 decoded frames/s, not 100 fps.
        self.assertEqual(self.stream(health,10,3)[-1],'LOW VIDEO FPS')

    def test_loss_and_reconnect_reset_low_rate_timer(self):
        health=ReturnVideoHealth()
        self.assertEqual(health.warning(0),'NO VIDEO')
        self.stream(health,10,3)
        self.assertEqual(health.warning(4.1),'VIDEO LOST')
        health.mark(4.2)
        self.assertIsNone(health.warning(4.2))
        self.assertIsNone(health.warning(4.3))

    def test_duplicate_and_out_of_order_frames_are_ignored(self):
        health=ReturnVideoHealth()
        health.mark(1)
        for _ in range(100):
            health.mark(1);health.mark(.5)
        self.assertEqual(health.fps(1),1)

    def test_pi_alert_and_unknown_switch_states(self):
        status=SystemStatus()
        for step in range(301):
            now=step/100
            if step%10==0:status.on_video_frame(now)
            alerts=status.alerts(now)
        self.assertIn('LOW VIDEO FPS',alerts)
        self.assertNotIn('fps',' '.join(status.link_lines(3)))
        status.on_panel({'emergency':False,'fire_arm':True,'inputs_valid':{'safety':False}},3)
        self.assertEqual(status.fire_text(3),'fire --')
        self.assertEqual(status.estop_text(3),'E-stop clear')
        self.assertEqual(status.estop_text(5),'E-stop ?')
        status.on_panel({'emergency':True,'fire_arm':True,'fire':True},5)
        self.assertEqual(status.estop_text(5),'E-stop ON')
        self.assertIn('EMERGENCY STOP',status.alerts(5))


if __name__=='__main__':unittest.main()
