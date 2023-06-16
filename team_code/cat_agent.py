import os
import json
import datetime
import pathlib
import time
import cv2
import carla
from collections import deque
import math
from collections import OrderedDict
import torch
import carla
import numpy as np
from PIL import Image
from torchvision import transforms as T
from leaderboard.autoagents import autonomous_agent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
import yaml

from team_code.waypointer import Waypointer
from team_code.planner import RoutePlanner
from cat.cat_planner import CatPlanner
from cat.pid import PIDController
from cat.rgb import RGBBrakePredictionModel

SAVE_PATH = None

def get_entry_point():
	return 'CaTAgent'

def _rotate(x, y, theta):
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    return R @ [x, y]


class CaTAgent(autonomous_agent.AutonomousAgent):
	def setup(self, path_to_conf_file, route_index=None):

		self.track = autonomous_agent.Track.SENSORS
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

		self.cat_planner = CatPlanner(
			batch_size=1, 
			num_proj=4, 
			encoder='ResNet34', 
			feature_channels=128, 
			nhead=8, 
			npoints=8, 
			crop_size=self.crop_size, pixels_per_meter=self.pixels_per_meter, 
			num_plan=self.num_plan, num_cmds=self.num_cmds, 
			num_plan_iter=self.num_plan_iter, num_classes=9).to(self.device)

		self.cat_planner.load_state_dict(torch.load(self.model_dir)['persformer_model_state_dict'])
		self.cat_planner.eval()

		self.bra_model = RGBBrakePredictionModel().to(self.device)
		self.bra_model.load_state_dict(torch.load(self.bra_model_dir))
		self.bra_model.eval()

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

		self.cam_config = {
			'width': 480,
			'height': 270,
			'fov': 64
		}

		self.stop_counter = 0
		self.force_move = 0

	def _init(self):

		self._route_planner = RoutePlanner(3.5, 50.0)
		self._route_planner.set_route(self._global_plan, True)

		self.waypointer = None

		self.initialized = True

	def _get_position(self, tick_data):
		gps = tick_data['gps']
		gps = (gps - self._route_planner.mean) * self._route_planner.scale

		return gps

	def sensors(self):
				return [
				{
					'type': 'sensor.camera.rgb',
					'x': -1.5, 'y': 0.0, 'z':2.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'width': self.cam_config['width'], 'height': self.cam_config['height'], 'fov': self.cam_config['fov'],
					'id': 'rgb_front'
				},
				{
					'type': 'sensor.camera.rgb',
					'x': -1.5, 'y': 0.0, 'z':2.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': -60.0,
					'width': self.cam_config['width'], 'height': self.cam_config['height'], 'fov': self.cam_config['fov'],
					'id': 'rgb_left'
				},
				{
					'type': 'sensor.camera.rgb',
					'x': -1.5, 'y': 0.0, 'z':2.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0,
					'width': self.cam_config['width'], 'height': self.cam_config['height'], 'fov': self.cam_config['fov'],
					'id': 'rgb_right'
				},
				{
					'type': 'sensor.camera.rgb', 
					'x': 1.5, 'y': 0.0, 'z': 2.4, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'width': 480, 'height': 288, 'fov': 40, 'id': 'rgb_tel'
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
					},
				]

	def tick(self, input_data):
		self.step += 1

		rgb = []
		for pos in ['left', 'front', 'right']:
			rgb_cam = 'rgb_' + pos
			rgb.append(cv2.cvtColor(input_data[rgb_cam][1][:, :, :3], cv2.COLOR_BGR2RGB))
		rgb = np.concatenate(rgb, axis=1)

		top_crop = 46
		rgb = rgb[top_crop:,:,:]

		_, tel_rgb = input_data.get('rgb_tel')
		tel_rgb = tel_rgb[...,:3][...,::-1].copy()
		tel_rgb = tel_rgb[:-self.crop_tel_bottom]

		gps = input_data['gps'][1][:2]
		speed = input_data['speed'][1]['speed']
		compass = input_data['imu'][1][-1]

		_gps = input_data['gps'][1]

		if (math.isnan(compass) == True): #It can happen that the compass sends nan for a few frames
			compass = 0.0

		result = {
				'rgb': rgb,
				'tel_rgb': tel_rgb,
				'gps': gps,
				'speed': speed,
				'compass': compass,
				}
		
		pos = self._get_position(result)
		result['gps'] = pos
		next_wp, next_cmd = self._route_planner.run_step(pos)

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

		if not self.initialized:
			self._init()

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

		rgb = torch.tensor(tick_data['rgb'][None]).permute(0,3,1,2).float().to(self.device)

		nxps = torch.tensor(tick_data['nxps']).float().to(self.device)

		spd = tick_data['speed']

		if spd < 0.1:
			self.stop_counter += 1
		else:
			self.stop_counter = 0

		tel_rgbs = torch.tensor(tick_data['tel_rgb'][None]).permute(0,3,1,2).float().to(self.device)

		pred_bra = self.bra_model(rgb, tel_rgbs)

		ego_plan_locs, ego_cast_locs, ego_cast_cmds, pred_seg_bev_map = self.cat_planner.infer(rgb, nxps, command)

		ego_plan_locs = to_numpy(ego_plan_locs)
		ego_cast_locs = to_numpy(ego_cast_locs)

		_pred_seg_bev_map = pred_seg_bev_map.squeeze(0).detach().cpu().numpy().argmax(0)

		if self.no_refine:
			ego_plan_locs = ego_cast_locs

		if not np.isnan(ego_plan_locs).any():
			steer, throt, brake, aim, angle, _steer = self.pid_control(ego_plan_locs, spd, command)
		else:
			steer, throt, brake = 0, 0, 0

		_disagree = (throt>0 or pred_bra<=0.1)

		if float(pred_bra) > 0.1:
			throt, brake = 0, 1

		if spd * 3.6 > self.max_speed:
			throt = 0

		if self.stop_counter >= 600:
			self.force_move = 20

		if self.force_move > 0 and _disagree:
			throt, brake = max(0.4, throt), 0
			self.force_move -= 1

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


	@staticmethod
	def _get_spawn_points(c_map):
		all_spawn_points = c_map.get_spawn_points()

		spawn_transforms = []
		for trans in all_spawn_points:
			wp = c_map.get_waypoint(trans.location)

			if wp.is_junction:
				wp_prev = wp
				# wp_next = wp
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

		self.stop_counter = 0
		self.force_move = 0


		del self.cat_planner
		torch.cuda.empty_cache()


def _rotate(x, y, theta):
	R = np.array([
		[np.cos(theta), -np.sin(theta)],
		[np.sin(theta), np.cos(theta)]
	])

	return R @ [x, y]

def to_numpy(x):
	return x.detach().cpu().numpy()
