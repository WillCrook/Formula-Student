import unittest
import numpy as np
from telemetry import (Channel, Session, find_crossings, interpolate_channel,
                       interp_distance, local_xy, split_laps, align_laps, validate_series,
                       gps_acceleration, STANDARD_GRAVITY_MPS2)
from analyse import load_analysis, gate_check

class GeometryTests(unittest.TestCase):
    def test_interpolated_direction_and_finite_segment(self):
        # Vertical southbound gate: west -> east is negative -> positive.
        line=np.array([[0.,1.],[0.,-1.]])
        time=np.arange(6.)
        xy=np.array([[-1,0],[1,0],[-1,0],[-1,3],[1,3],[-1,3]],float)
        crossings,audit=find_crossings(time,xy,line)
        self.assertEqual(len(crossings),1)
        self.assertAlmostEqual(crossings[0].time,.5)
        np.testing.assert_allclose(crossings[0].xy,[0,0])
        self.assertEqual(audit["direction_candidates"],2)
        self.assertEqual(audit["outside_segment"],1)
        self.assertAlmostEqual(crossings[0].line_fraction,.5)

    def test_exact_sample_and_touch(self):
        line=np.array([[0.,1.],[0.,-1.]])
        time=np.arange(5.)
        crossing,_=find_crossings(time,np.array([[-1,0],[0,0],[1,0],[0,0],[1,0]]),line)
        self.assertEqual(len(crossing),1)
        self.assertEqual(crossing[0].time,1)
        touch,_=find_crossings(time,np.array([[-1,0],[0,0],[-1,0],[0,0],[-1,0]]),line)
        self.assertEqual(touch,[])

    def test_projection_scale_and_origin(self):
        xy=local_xy(np.array([[0,0],[.001,0],[0,.001]]),np.array([0,0]))
        np.testing.assert_allclose(xy[0],[0,0],atol=1e-10)
        self.assertAlmostEqual(xy[1,0],111.31949,places=4)
        self.assertAlmostEqual(xy[2,1],110.57428,places=4)

    def test_degenerate_line_rejected(self):
        with self.assertRaises(ValueError):
            find_crossings(np.array([0.,1.]),np.array([[0.,0.],[1.,1.]]),np.zeros((2,2)))

class AlignmentTests(unittest.TestCase):
    def test_channels_use_own_clock_and_no_extrapolation(self):
        channel=Channel(np.array([10.,12.,14.]),np.array([0.,20.,40.]),"%")
        value=interpolate_channel(channel,np.array([9.,11.,13.,15.]))
        np.testing.assert_allclose(value[1:3],[10.,30.])
        self.assertTrue(np.isnan(value[0]) and np.isnan(value[-1]))

    def test_repeated_distance_departure_policy(self):
        values=interp_distance([0,1,1,2],[0,1,3,4],np.array([0.,1.,2.]))
        np.testing.assert_allclose(values,[0,3,4])
        with self.assertRaises(ValueError):
            interp_distance([0,2,1],[0,1,2],np.array([1.]))

    def test_timestamp_validation(self):
        for time in [[0,0,1],[0,2,1]]:
            with self.assertRaises(ValueError):
                validate_series(np.array(time),np.ones(3),"test")

class SuppliedDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.session,cls.crossings,cls.detection,cls.laps,cls.u,cls.aligned,cls.fastest=load_analysis()

    def test_independent_baseline_sanity_check(self):
        self.assertEqual(len(self.crossings),14)
        self.assertEqual(len(self.laps),13)
        self.assertEqual(self.fastest.number,12)
        ordered=sorted(self.laps,key=lambda l:l.duration)
        self.assertEqual(ordered[1].number,11)
        self.assertAlmostEqual(self.fastest.duration,62.407274,places=5)
        self.assertAlmostEqual(ordered[1].duration-self.fastest.duration,.274309,places=5)

    def test_boundaries_coverage_and_delta_endpoints(self):
        for i,lap in enumerate(self.laps):
            self.assertEqual(lap.time[0],self.crossings[i].time)
            self.assertEqual(lap.time[-1],self.crossings[i+1].time)
            self.assertEqual(lap.distance[0],0)
            self.assertTrue(np.all(np.diff(lap.distance)>0))
            delta=self.aligned[lap.number]["delta"]
            self.assertAlmostEqual(delta[0],0)
            self.assertAlmostEqual(delta[-1],lap.duration-self.fastest.duration)
            self.assertTrue(all(np.all(np.isfinite(v)) for v in lap.channels.values()))

    def test_physical_gate_sections_sum_to_lap_gap(self):
        gates=gate_check(self.laps,self.u,self.aligned,self.fastest)
        self.assertEqual(len(gates["diagnostics"]),2*len(self.laps))
        for lap in self.laps:
            self.assertAlmostEqual(sum(gates["section_delta_s"][str(lap.number)]),
                                   lap.duration-self.fastest.duration)


class AccelerationTests(unittest.TestCase):
    def setUp(self):
        self.time=np.arange(0,12.01,.1)

    def test_constant_speed_and_rotation_translation(self):
        t=self.time
        xy=np.column_stack([10*t,3*t])
        lat,long,_=gps_acceleration(t,xy)
        np.testing.assert_allclose(lat[5:-5],0,atol=1e-11)
        np.testing.assert_allclose(long[5:-5],0,atol=1e-11)
        angle=.8
        rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        lat_rot,long_rot,_=gps_acceleration(t,xy@rotation.T+[1000,-500])
        np.testing.assert_allclose(lat_rot,lat,atol=1e-10,equal_nan=True)
        np.testing.assert_allclose(long_rot,long,atol=1e-10,equal_nan=True)

    def test_acceleration_and_braking_in_g(self):
        t=self.time
        direction=np.array([-.6,.8])
        for acceleration,initial in [(2.,8.),(-1.,20.)]:
            xy=(initial*t+.5*acceleration*t*t)[:,None]*direction
            lat,long,_=gps_acceleration(t,xy)
            np.testing.assert_allclose(lat[5:-5],0,atol=1e-11)
            np.testing.assert_allclose(long[5:-5],acceleration/STANDARD_GRAVITY_MPS2,atol=1e-11)

    def test_circle_both_turn_directions(self):
        t=self.time
        radius,omega=20.,.4
        for turn in [-1,1]:
            xy=radius*np.column_stack([np.cos(omega*t),turn*np.sin(omega*t)])
            lat,long,_=gps_acceleration(t,xy)
            np.testing.assert_allclose(lat[5:-5],turn*radius*omega**2/STANDARD_GRAVITY_MPS2,rtol=.01)
            np.testing.assert_allclose(long[5:-5],0,atol=1e-10)

    def test_centred_filter_does_not_delay_symmetric_event(self):
        t=self.time
        omega=np.pi
        xy=np.column_stack([10*t-.6/omega*np.cos(omega*(t-5)),np.zeros(len(t))])
        lat,long,_=gps_acceleration(t,xy)
        event=(t>=4.5)&(t<=5.5)
        self.assertAlmostEqual(t[event][np.argmax(long[event])],5.)

    def test_low_speed_and_edges_are_unavailable(self):
        t=self.time
        lat,long,metadata=gps_acceleration(t,np.column_stack([.5*t,np.zeros(len(t))]))
        self.assertTrue(np.all(np.isnan(lat)) and np.all(np.isnan(long)))
        lat,long,_=gps_acceleration(t,np.column_stack([10*t,np.zeros(len(t))]))
        for values in [lat,long]:
            self.assertTrue(np.all(np.isnan(values[:5])) and np.all(np.isnan(values[-5:])))
            self.assertTrue(np.all(np.isfinite(values[5:-5])))
        self.assertAlmostEqual(metadata["window_span_s"],1.)

    def test_irregular_sampling_and_invalid_windows_rejected(self):
        t=self.time.copy()
        t[10]+=.02
        xy=np.column_stack([10*t,np.zeros(len(t))])
        with self.assertRaisesRegex(ValueError,"uniformly"):
            gps_acceleration(t,xy)
        for window in [0,4,6,1001,11.5]:
            with self.assertRaises(ValueError):
                gps_acceleration(self.time,xy,window_samples=window)

if __name__=="__main__":
    unittest.main()
