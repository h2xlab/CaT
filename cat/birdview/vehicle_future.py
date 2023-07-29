
extrapolation_seconds = 2
self.frame_rate = 20

self.vehicle_model = EgoModel(dt=(1.0 / self.frame_rate))
number_of_future_frames = int(extrapolation_seconds * self.frame_rate)
_future_idx = [1, 6, 11, 16]

def _get_vehicle_futures(vehicle_list, criteria, scale=None):

    # actors = self._world.get_actors()
    # vehicles = actors.filter('*vehicle*')


    nearby_vehicles = {}
    for vehicle in vehicles:
        is_within_distance = criteria(vehicle)

        if is_within_distance:
            veh_future_bbs    = []
            traffic_transform = vehicle.get_transform()
            traffic_control   = vehicle.get_control()
            traffic_velocity  = vehicle.get_velocity()
            traffic_speed     = self._get_forward_speed(transform=traffic_transform, velocity=traffic_velocity) # In m/s

            next_loc   = np.array([traffic_transform.location.x, traffic_transform.location.y])
            action     = np.array(np.stack([traffic_control.steer, traffic_control.throttle, traffic_control.brake], axis=-1))
            next_yaw   = np.array([traffic_transform.rotation.yaw / 180.0 * np.pi])
            next_speed = np.array([traffic_speed])
            
            for i in range(self._number_of_future_frames):

                next_loc, next_yaw, next_speed = self._vehicle_model.forward(next_loc, next_yaw, next_speed, action)

                delta_yaws = next_yaw.item() * 180.0 / np.pi

                transform             = carla.Transform(carla.Location(x=next_loc[0].item(), y=next_loc[1].item(), z=traffic_transform.location.z))
                bounding_box          = carla.BoundingBox(transform.location, vehicle.bounding_box.extent)
                bounding_box.rotation = carla.Rotation(pitch=float(traffic_transform.rotation.pitch),
                                                    yaw=float(delta_yaws),
                                                    roll=float(traffic_transform.rotation.roll))

                bb_loc = carla.Location()
                bb_ext = carla.Vector3D(bounding_box.extent)
                if scale is not None:
                    bb_ext = bb_ext * scale
                    bb_ext.x = max(bb_ext.x, 0.8)
                    bb_ext.y = max(bb_ext.y, 0.8)

                veh_future_bbs.append((carla.Transform(bounding_box.location, bounding_box.rotation), bb_loc, bb_ext))
            

            nearby_vehicles[vehicle.id] = veh_future_bbs
    
    vehicle_futures = []
    for idx in self._future_idx:
        idx = np.clip(idx, 1, self._number_of_future_frames)
        future_at_idx = []
        for k, v in nearby_vehicles.items():
            future_at_idx.append(v[idx - 1]) # minus 1 because future is 0-indexed
        vehicle_futures.append((future_at_idx))
    return vehicle_futures

def _get_forward_speed(self, transform, velocity):
    """ Convert the vehicle transform directly to forward speed """
    vel_np = np.array([velocity.x, velocity.y, velocity.z])
    pitch = np.deg2rad(transform.rotation.pitch)
    yaw = np.deg2rad(transform.rotation.yaw)
    orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
    speed = np.dot(vel_np, orientation)
    return speed



def _get_future_masks(self, M_warp):
    vehicle_masks = []
    for future in self._vehicle_futures:
        vehicle_masks.append(self._get_mask_from_actor_list(future, M_warp))
    return vehicle_masks