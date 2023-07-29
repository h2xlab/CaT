import os
import json
import datetime
import pathlib
import time
import math
import yaml

from collections import deque
from collections import OrderedDict

from omegaconf import OmegaConf
import numpy as np
import cv2
from PIL import Image
import carla

import torch
from torchvision import transforms as T

from leaderboard.autoagents import autonomous_agent

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


from team_code.planner import RoutePlanner
from team_code.autopilot import AutoPilot
from team_code.waypointer import Waypointer

from cat.teacher_planner import TeacherPlanner
from cat.pid import PIDController
from cat.birdview.common.task_vehicle import TaskVehicle
from cat.birdview.chauffeurnet import ObsManager as bev_utils
from cat.birdview.traffic_light import TrafficLightHandler

SAVE_PATH = os.environ.get('SAVE_PATH', None)


def get_entry_point():
    return 'TeacherAgent'

class TeacherAgent(AutoPilot):
    def setup(self, path_to_conf_file, route_index=None):

        super().setup(path_to_conf_file, route_index)


        self.track = autonomous_agent.Track.MAP
        self.alpha = 0.3
        self.status = 0
        self.steer_step = 0
        self.last_moving_status = 0
        self.last_moving_step = -1
        self.last_steers = deque()

        self.config_path = path_to_conf_file
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False

        with open(path_to_conf_file, 'r') as f:
            config = yaml.safe_load(f)

        for key, value in config.items():
            setattr(self, key, value)

        self.device = torch.device('cuda')

        self.teacher_planner = TeacherPlanner(
            pixels_per_meter=self.pixels_per_meter,
            crop_size=self.crop_size,
            feature_x_jitter=self.feature_x_jitter,
            feature_angle_jitter=self.feature_angle_jitter,
            x_offset=0, y_offset=1+self.min_x/((self.max_x-self.min_x)/2),
            num_cmds=self.num_cmds,
            num_plan=self.num_plan,
            num_plan_iter=self.num_plan_iter,
        ).to(self.device)

        print(self.model_dir)

        self.teacher_planner.load_state_dict(torch.load(self.model_dir))
        self.teacher_planner.eval()

        self.turn_controller = PIDController(K_P=self.turn_KP, K_I=self.turn_KI, K_D=self.turn_KD, n=self.turn_n)
        self.speed_controller = PIDController(K_P=self.speed_KP, K_I=self.speed_KI, K_D=self.speed_KD, n=self.speed_n)




        self.takeover = False
        self.stop_time = 0
        self.takeover_time = 0

        self.save_path = None

        self.last_steers = deque()
        if SAVE_PATH is not None:
            now = datetime.datetime.now()
            string = pathlib.Path(os.environ['ROUTES']).stem + '_'
            string += '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))

            print (string)

            self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
            self.save_path.mkdir(parents=True, exist_ok=False)

            (self.save_path / 'rgb').mkdir()
            (self.save_path / 'meta').mkdir()
            (self.save_path / 'bev').mkdir()
            (self.save_path / 'chauffeur').mkdir()

    def _init(self, hd_map):
        super()._init(hd_map)

        self._route_planner = RoutePlanner(3.5, 50.0)
        self._route_planner.set_route(self._global_plan, True)

        self.waypointer = None


        # chauffeur
        self._vehicle = CarlaDataProvider.get_hero_actor()
        self._world = self._vehicle.get_world()
        self._map = self._world.get_map()

        self._obs_configs = OmegaConf.load('config_agent.yaml')['obs_configs']
        snap_shot = self._world.get_snapshot()

        self._timestamp = {
            'step': 0,
            'frame': snap_shot.timestamp.frame,
            'relative_wall_time': 0.0,
            'wall_time': snap_shot.timestamp.platform_timestamp,
            'relative_simulation_time': 0.0,
            'simulation_time': snap_shot.timestamp.elapsed_seconds,
            'start_frame': snap_shot.timestamp.frame,
            'start_wall_time': snap_shot.timestamp.platform_timestamp,
            'start_simulation_time': snap_shot.timestamp.elapsed_seconds
        }

        self.bev_manager = bev_utils(self._obs_configs['birdview'])

        self._spawn_transforms = self._get_spawn_points(self._map)
        target_transforms = CarlaDataProvider._chauffeur_config.chauffeur_trajectory[1:]
        endless = False

        self._parent_vehicle = TaskVehicle(self._vehicle, target_transforms, self._spawn_transforms, endless)

        self.bev_manager.attach_ego_vehicle(self._parent_vehicle)

        TrafficLightHandler.reset(self._world)

        self.initialized = True

    def _get_position(self, tick_data):
        gps = tick_data['gps']
        gps = (gps - self._route_planner.mean) * self._route_planner.scale

        return gps

    def sensors(self):
                return [
                {
                    'type': 'sensor.opendrive_map',
                    'reading_frequency': 1e-6,
                    'id': 'hd_map'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -1.5, 'y': 0.0, 'z':2.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': 900, 'height': 256, 'fov': 100,
                    'id': 'rgb'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.0, 'y': 0.0, 'z': 50.0,
                    'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                    'width': 512, 'height': 512, 'fov': 5 * 10.0,
                    'id': 'bev'
                    },	
                {
                    'type': 'sensor.other.imu',
                    'x': 0.0, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.05,
                    'id': 'imu'
                    },
                {
                    'type': 'sensor.other.gnss',
                    'x': 0.0, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.01,
                    'id': 'gps'
                    },
                {
                    'type': 'sensor.speedometer',
                    'reading_frequency': 20,
                    'id': 'speed'
                    }
                ]

    def tick(self, input_data):
        self.step += 1

        snap_shot = self._world.get_snapshot()
        self._timestamp['step'] = snap_shot.timestamp.frame-self._timestamp['start_frame']
        self._timestamp['frame'] = snap_shot.timestamp.frame
        self._timestamp['wall_time'] = snap_shot.timestamp.platform_timestamp
        self._timestamp['relative_wall_time'] = self._timestamp['wall_time'] - self._timestamp['start_wall_time']
        self._timestamp['simulation_time'] = snap_shot.timestamp.elapsed_seconds
        self._timestamp['relative_simulation_time'] = self._timestamp['simulation_time'] \
            - self._timestamp['start_simulation_time']

        _ = self._parent_vehicle.tick(self._timestamp)

        chauffeur = self.bev_manager.get_observation(self.hazard_source)
        masks = chauffeur['masks']
        rendered = chauffeur['rendered']

        assert masks.shape == (21, 320, 320)
        assert rendered.shape == (320, 320, 3)

        _bev = np.uint8(masks)
        assert _bev.shape == (21, 320, 320)

        cropped_bev = _bev[:, 88:280, 64:256]
        assert cropped_bev.shape == (21, 192, 192)


        rgb = cv2.cvtColor(input_data['rgb'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        bev = cv2.cvtColor(input_data['bev'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        gps = input_data['gps'][1][:2]
        speed = input_data['speed'][1]['speed']
        compass = input_data['imu'][1][-1]

        _gps = input_data['gps'][1]

        if (math.isnan(compass) == True): #It can happen that the compass sends nan for a few frames
            compass = 0.0

        result = {
                'rgb': rgb,
                'gps': gps,
                'speed': speed,
                'compass': compass,
                'bev': bev,
                'chauffeur': cropped_bev, #_bev,
                'chauffeur_vis': rendered
                }
        
        pos = self._get_position(result)
        result['gps'] = pos
        next_wp, next_cmd = self._route_planner.run_step(pos)
        result['next_command'] = next_cmd.value


        if self.waypointer is None:
            self.waypointer = Waypointer(
                self._global_plan, _gps
            )

        _, _, cmd = self.waypointer.tick(_gps)

        result['command'] = cmd.value



        theta = compass + np.pi/2
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
            ])

        local_command_point = np.array([next_wp[0]-pos[0], next_wp[1]-pos[1]])
        local_command_point = R.T.dot(local_command_point)
        result['target_point'] = tuple(local_command_point)

        wx, wy = _rotate(next_wp[0]-pos[0], next_wp[1]-pos[1], -compass+np.pi/2)

        result['nxps'] = np.array([-wx,-wy])



        return result
    @torch.no_grad()
    def run_step(self, input_data, timestamp):

        _ = super().run_step(input_data, timestamp)


        tick_data = self.tick(input_data)
        if self.step < 1:

            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 0.0
            
            return control

        command = tick_data['command']
        if command < 0:
            command = 4
        command -= 1
        assert command in [0, 1, 2, 3, 4, 5]

        nxps = torch.tensor(tick_data['nxps']).float().to(self.device)

        spd = tick_data['speed']
        _spd = torch.tensor(tick_data['speed']).float().to(self.device)



        _bev = torch.tensor(tick_data['chauffeur'][None]).float().to('cuda')


        ego_plan_locs, ego_cast_locs, ego_cast_cmds = self.teacher_planner.infer(_bev, nxps, command, _spd)

        ego_plan_locs = to_numpy(ego_plan_locs)
        ego_cast_locs = to_numpy(ego_cast_locs)

        if self.no_refine:
            ego_plan_locs = ego_cast_locs

        if not np.isnan(ego_plan_locs).any():
            steer, throt, brake, aim, angle, _steer = self.pid_control(ego_plan_locs, spd, command)
        else:
            steer, throt, brake = 0, 0, 0

        if spd * 3.6 > self.max_speed:
            throt = 0

        self.pid_metadata = {}
        self.pid_metadata['steer'] = float(steer)
        self.pid_metadata['throt'] = float(throt)
        self.pid_metadata['brake'] = float(brake)
        self.pid_metadata['spd'] = float(spd)
        self.pid_metadata['command'] = command
        self.pid_metadata['ego_plan_locs'] = ego_plan_locs
        self.pid_metadata['nxps'] = tick_data['nxps']

        control = carla.VehicleControl(steer=steer, throttle=throt, brake=brake)

        if SAVE_PATH is not None and self.step % 10 == 0:
            self.save(tick_data)
        return control

    def pid_control(self, waypoints, speed, cmd):

        waypoints = np.copy(waypoints) * 4
        waypoints[:,1] *= -1

        desired_speed = np.linalg.norm(waypoints[1:]-waypoints[:-1], axis=1).mean()

        aim = waypoints[self.aim_point]
        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90

        if (speed < 0.01):
            angle = 0.0  # When we don't move we don't want the angle error to accumulate in the integral


        brake = desired_speed < self.brake_speed * 4

        if brake:
            angle = 0.0

        _steer = self.turn_controller.step(angle)
        steer = np.clip(_steer, -1.0, 1.0)


        delta = np.clip(desired_speed - speed, 0.0, self.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.max_throttle)
        throttle = throttle if not brake else 0.0

        return float(steer), float(throttle), float(brake), aim, angle, float(_steer)

    def save(self, tick_data):
        frame = self.step // 10

        Image.fromarray(tick_data['rgb']).save(self.save_path / 'rgb' / ('%04d.png' % frame))

        Image.fromarray(tick_data['bev']).save(self.save_path / 'bev' / ('%04d.png' % frame))



        wpts = self.pid_metadata.pop('ego_plan_locs')
        wpts = wpts * self.pixels_per_meter

        wpts[:,0] += 160
        wpts[:,1] += 280

        _map = tick_data['chauffeur_vis']

        for _idx in range(10):
            _x, _y = wpts[_idx]
            cv2.circle(_map, (int(_x), int(_y)), 2, (255,0,255), -1)

        vis_nxp = self.pid_metadata.pop('nxps') * self.pixels_per_meter + [160, 280]
        cv2.circle(_map, (int(vis_nxp[0]), int(vis_nxp[1])), 4, (255,120,120), -1)

        CMD_DICT = {0: 'LEFT', 1: 'RIGHT', 2: 'STRAIGHT', 3: 'LANEFOLLOW', 4: 'CHANGELANELEFT', 5: 'CHANGELANERIGHT'}
        cv2.putText(_map, 'cmd: '+CMD_DICT[self.pid_metadata['command']], (20,40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 1)



        Image.fromarray(_map).save(self.save_path / 'chauffeur' / ('%04d.png' % frame))


        outfile = open(self.save_path / 'meta' / ('%04d.json' % frame), 'w')
        json.dump(self.pid_metadata, outfile, indent=4)
        outfile.close()

    @staticmethod
    def _get_spawn_points(c_map):
        all_spawn_points = c_map.get_spawn_points()

        spawn_transforms = []
        for trans in all_spawn_points:
            wp = c_map.get_waypoint(trans.location)

            if wp.is_junction:
                wp_prev = wp
                while wp_prev.is_junction:
                    wp_prev = wp_prev.previous(1.0)[0]
                spawn_transforms.append([wp_prev.road_id, wp_prev.transform])
                if c_map.name == 'Town03' and (wp_prev.road_id == 44):
                    for _ in range(100):
                        spawn_transforms.append([wp_prev.road_id, wp_prev.transform])

            else:
                spawn_transforms.append([wp.road_id, wp.transform])
                if c_map.name == 'Town03' and (wp.road_id == 44):
                    for _ in range(100):
                        spawn_transforms.append([wp.road_id, wp.transform])

        return spawn_transforms

    def destroy(self):
        del self.teacher_planner
        torch.cuda.empty_cache()


def _rotate(x, y, theta):
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    return R @ [x, y]

def to_numpy(x):
    return x.detach().cpu().numpy()