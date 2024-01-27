from airgym import airsim
import numpy as np
from PIL import Image
import open3d as o3d
import random
import imageio

from gymnasium import spaces
from airgym.envs.airsim_env import AirSimEnv


class AirSimDroneEnv(AirSimEnv):
    def __init__(self):
        super().__init__()

        #env config
        self.step_length = 5
        self.z_val = -15.0
        self.image_shape = (1,84,84)
        self.map_size = 168
        self.map_hist_len = 5
        self.observation_space = spaces.Dict({
            "obs1":spaces.Box(0, 255, shape=self.image_shape, dtype=np.uint8),
            "obs2":spaces.Box(0, 255, shape=(self.map_hist_len, self.map_size,self.map_size), dtype=np.uint8),
        })
        self.action_space = spaces.Discrete(3)

        #airsim
        self.drone = airsim.MultirotorClient()
        self.drone.simPause(False)
        self.image_request = airsim.ImageRequest(0, airsim.ImageType.DepthVis, True)

        #env var
        self.drone_ori = None # 0,90,180,270

        self.global_pcd = None
        self.curr_global_pcd_len = None
        self.prev_global_pcd_len = None
        self.best_pcd_len = 0

        self.not_move = None
        self.num_step = None
        self.map_hist = None

    def __del__(self):
        self.drone.simPause(False)
        self.drone.reset()

    def reset(self, seed=None):
        super().reset(seed=seed)
        self._setup_flight()
        return self._get_obs(), {}

    def set_random_pose(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        pose = self.drone.simGetVehiclePose()
        x_pos = random.randint(0,90)
        y_pos = random.randint(0,120)
        self.drone_ori = random.randint(0,3)*90

        pose.position.x_val += x_pos
        pose.position.y_val += y_pos
        self.drone.simSetVehiclePose(pose, True)
        self.drone.moveToPositionAsync(x_pos, y_pos, self.z_val, 10, timeout_sec=3).join()

        self.drone.moveByVelocityAsync(
            0, 0, 0, 3,
            airsim.DrivetrainType.MaxDegreeOfFreedom,
            airsim.YawMode(False, self.drone_ori)
        ).join()

        success = self.check_collision() is False

        return success

    def _setup_flight(self):

        self.global_pcd = o3d.geometry.PointCloud()
        self.curr_global_pcd_len = 0
        self.prev_global_pcd_len = 0
        self.not_move=0
        self.num_step = 0
        self.map_hist = np.zeros((self.map_hist_len, self.map_size, self.map_size), dtype=np.uint8)

        self.set_random_pose()

    def get_depth_view(self, responses):
        response = responses[0]
        try:
            image = np.array(response.image_data_float, dtype=np.float32)
            image = image.reshape(response.height, response.width)*255
            image[image>255] =255
            image = Image.fromarray(image)
        except:
            return np.zeros(self.image_shape)

        img_size = self.image_shape[1]
        im_final = np.array(image.resize((img_size, img_size)))
        return im_final.reshape(self.image_shape).astype(np.uint8)

    def _get_obs(self):
        self.drone.simPause(True)
        obs = {}

        map2d = np.zeros((self.map_size,self.map_size), dtype=np.uint8)
        #map(drone position)
        drone_state = self.drone.getMultirotorState().kinematics_estimated
        position = np.array([drone_state.position.x_val,drone_state.position.y_val])
        position = position + 21
        position[position<0] = 0
        position[position>=self.map_size] = self.map_size
        position = np.around(position).astype('int')

        try:
            map2d[position[0]-2:position[0]+3,position[1]-2:position[1]+3] = 255
            if self.drone_ori == 0: 
                map2d[position[0]+3,position[1]-1:position[1]+2] = 255
                map2d[position[0]+4,position[1]] = 255
            elif self.drone_ori == 90: 
                map2d[position[0]-1:position[0]+2,position[1]+3] = 255
                map2d[position[0],position[1]+4] = 255
            elif self.drone_ori == 180: 
                map2d[position[0]-3,position[1]-1:position[1]+2] = 255
                map2d[position[0]-4,position[1]] = 255
            elif self.drone_ori == 270: 
                map2d[position[0]-1:position[0]+2,position[1]-3] = 255
                map2d[position[0],position[1]-4] = 255
        except:
            self.drone.simPause(False)
            return obs

        #map(pcd)
        get_pcd = self.get_transformed_lidar_pc()
        self.global_pcd = self.global_pcd + get_pcd
        self.global_pcd = self.global_pcd.voxel_down_sample(voxel_size=2)

        points = np.array(self.global_pcd.points)
        points = points[points[:,2]<(-10)]
        points = points[:,:2]
        self.curr_global_pcd_len = len(points)
        points = points + 21
        points[points<0] = 0
        points[points>=self.map_size] = self.map_size
        points = np.around(points).astype('int')
        map2d[points[:,0], points[:,1]] = 128
        map2d = map2d.reshape((1,self.map_size,self.map_size)).astype(np.uint8)
        self.map_hist = np.concatenate([self.map_hist,map2d],axis=0)
        obs['obs2'] = self.map_hist[-self.map_hist_len:]

		#drone view
        responses = self.drone.simGetImages([self.image_request])
        obs['obs1'] = self.get_depth_view(responses)

        self.drone.simPause(False)

        return obs

    def _do_action(self, action):
        quad_offset = self.interpret_action(action)
        drone_state= self.drone.getMultirotorState().kinematics_estimated
        if quad_offset[0] or quad_offset[1]:
            drone_pos = drone_state.position
            x_pos = drone_pos.x_val + quad_offset[0]
            y_pos = drone_pos.y_val + quad_offset[1]

            self.drone.moveToPositionAsync(
                x_pos,
                y_pos,
                self.z_val,
                5,
                timeout_sec=3
            ).join()
        else:
            new_ori = self.drone_ori + quad_offset[2]
            if new_ori < 0:
                new_ori = new_ori + 360
            self.drone_ori = new_ori%360
            self.drone.moveByVelocityAsync(
                0, 0, 0, 3,
                airsim.DrivetrainType.MaxDegreeOfFreedom,
                airsim.YawMode(False, self.drone_ori)
            ).join()


	
    def check_collision(self):
        z_val= self.drone.getMultirotorState().kinematics_estimated.position.z_val
        col_obj = self.drone.simGetCollisionInfo().object_id
        return (((col_obj != -1) and (col_obj != 148))) or (z_val > -10)


    def _compute_reward(self, action):
        self.num_step += 1
        done = False

        if self.check_collision():
            reward = -1000
            done = True
            print("collision",end=' ')
        else:
            if action == 0:
                self.not_move = 0
            else:
                self.not_move += 1

            if self.not_move >= 6:
                reward = -500
                done = True
                print("not move",end=' ')
            else:
                reward = self.curr_global_pcd_len - self.prev_global_pcd_len
                self.prev_global_pcd_len = self.curr_global_pcd_len

                if self.num_step > 200:
                    print("max step",end=' ')
                    done= True
        if done and (self.curr_global_pcd_len >= self.best_pcd_len):
            self.best_pcd_len = self.curr_global_pcd_len
            print("save gif", "lander_{}.gif".format(self.best_pcd_len))
            imageio.mimsave("gif/lander_{}.gif".format(self.best_pcd_len), self.map_hist[self.map_hist_len:], fps=1)
        print(reward)
        return reward, done

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        if obs:
            reward, done = self._compute_reward(action)
        else:
            reward = -1000
            done = True
            print("collision", reward)
        truncated = False
        info = {}

        return obs, reward, done, truncated, info


    def interpret_action(self, action):
        if action == 0:
            if self.drone_ori == 0:
                quad_offset = (self.step_length,0, 0)
            elif self.drone_ori == 90 :
                quad_offset = (0,self.step_length, 0)
            elif self.drone_ori == 180 :
                quad_offset = (-self.step_length,0, 0)
            elif self.drone_ori == 270 :
                quad_offset = (0,-self.step_length, 0)
        elif action == 1:
            quad_offset = (0, 0, -90)
        elif action == 2:
            quad_offset = (0, 0, 90)
        
        print(quad_offset, end=', ')

        return quad_offset

    def get_transformed_lidar_pc(self):
        # raw lidar data to open3d PointCloud
        lidar_data = self.drone.getLidarData()
        lidar_points = np.array([lidar_data.point_cloud[i:i+3] for i in range(0, len(lidar_data.point_cloud), 3)])
        pcd = o3d.geometry.PointCloud()
        try:
            pcd.points = o3d.utility.Vector3dVector(lidar_points)
        except:
            print("Error!!!")
            print(lidar_data)
            return o3d.geometry.PointCloud()

        # calc rotation matrix
        orientation = lidar_data.pose.orientation
        q0, q1, q2, q3 = orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val
        rotation_matrix = np.array(([1-2*(q2*q2+q3*q3),2*(q1*q2-q3*q0),2*(q1*q3+q2*q0)],
                                    [2*(q1*q2+q3*q0),1-2*(q1*q1+q3*q3),2*(q2*q3-q1*q0)],
                                    [2*(q1*q3-q2*q0),2*(q2*q3+q1*q0),1-2*(q1*q1+q2*q2)]))

        # calc translation vector
        position = lidar_data.pose.position
        x_val, y_val, z_val = position.x_val, position.y_val, position.z_val
        translation_vector = np.array([x_val, y_val, z_val])

        # apply transform
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = translation_vector
        return pcd.transform(transformation_matrix)
