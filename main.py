# Simulate Coders Strike Back from Codingame
from __future__ import annotations
from enum import Enum, auto
import math
import random
from dataclasses import dataclass
import tkinter as tk
import time

import numpy as np
import torch

from dqn import DuellingDDQN_PRBAgent

POD_RADIUS = 400
POD_RADIUS_SQUARED = POD_RADIUS * POD_RADIUS
POD_MAX_STEER_ANGLE_DEG = 18
CHECKPOINT_RADIUS = 600
MAX_THRUST = 200
MAP_WIDTH = 16000
MAP_HEIGHT = 9000
POD_TIMEOUT = 100
FPS = 10


class MoveType(Enum):
    THRUST = 1
    BOOST = 2
    SHIELD = 3


@dataclass
class Vec2D:
    """
    Vector class which holds
    - x and y coordinates

    """
    x: float
    y: float

    def __add__(self, other):
        return Vec2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vec2D(self.x - other.x, self.y - other.y)

    def __mul__(self, other: float):
        return Vec2D(self.x * other, self.y * other)

    def __rmul__(self, other: float):
        return Vec2D(self.x * other, self.y * other)

    def floor(self):
        return Vec2D(math.floor(self.x), math.floor(self.y))

    def normalize(self, max_x: float, max_y: float) -> Vec2D:
        return Vec2D(self.x / max_x, self.y / max_y)

    def __round__(self, n=None) -> Vec2D:
        if n is None:
            return Vec2D(round(self.x), round(self.y))
        else:
            return Vec2D(round(self.x, n), round(self.y, n))

    def dot(self, other: Vec2D) -> float:
        return self.x * other.x + self.y * other.y

    def norm_squared(self):
        return self.x * self.x + self.y * self.y

    def norm(self):
        return math.sqrt(self.norm_squared())

    def distance_squared(self, other):
        return (self.x - other.x) ** 2 + (self.y - other.y) ** 2

    def distance(self, other):
        return math.sqrt(self.distance_squared(other))

    def angle_deg(self, other) -> float:
        return math.degrees(self.angle_rad(other))

    def angle_rad(self, other) -> float:
        """ Returns the angle between two points self and other in radians with respect to the x-axis, counter clock wise """
        return math.atan2(other.y - self.y, other.x - self.x)

    def rotate(self, angle_rad: float):
        """ Rotate the vector by angle_rad radians """
        x = self.x * math.cos(angle_rad) - self.y * math.sin(angle_rad)
        y = self.x * math.sin(angle_rad) + self.y * math.cos(angle_rad)
        return Vec2D(x, y)

    def rotate_deg(self, angle_deg: float):
        """ Rotate the vector by angle_deg degrees """
        return self.rotate(math.radians(angle_deg))

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    @staticmethod
    def from_angle_deg(angle_deg: float) -> Vec2D:
        angle_rad = math.radians(angle_deg)
        return Vec2D(math.cos(angle_rad), math.sin(angle_rad))


@dataclass
class PodMove:
    """
    PodMove class which holds
    - the thrust and target or
    - if boost is used or
    - if shield is used
    """
    type: MoveType
    thrust: int = 0
    target: Vec2D | None = None


@dataclass
class Pod:
    """ contains pod data, e.g. position, velocity, angle, next checkpoint id, etc. """

    id: int
    team_id: int
    pos: Vec2D
    vel: Vec2D
    angle: float
    next_cp_id: int = 0
    shield: int = 0
    boosted: bool = False
    # increases each tick, if no checkpoint is passed for 100 ticks, the pod is destroyed
    timeout: int = 0
    checkpoints_passed: int = 0
    partner_id: int = -1
    destroyed: bool = False

    def __str__(self):
        return f"{self.pos} {self.vel} {self.angle} {self.next_cp_id} {self.shield} {self.boosted} " \
               f"{self.timeout} {self.checkpoints_passed}"

    def activate_shield(self) -> None:
        self.shield = 3

    def activate_boost(self) -> None:
        if self.boosted:
            return
        self.boosted = True
        self.vel = self.vel + \
                   Vec2D(650 * math.cos(self.angle), 650 * math.sin(self.angle))

    def move(self, thrust: int, target: Vec2D) -> None:
        """
        Moves the pod to the target with the given thrust.
        On each turn the pods movements are computed this way:

        Rotation: the pod rotates to face the target point, with a maximum of 18 degrees (except for the 1rst round).
        Acceleration: the pod's facing vector is multiplied by the given thrust value. The result is added to the
                        current speed vector.
        Movement: The speed vector is added to the position of the pod. If a collision occurs at this point, the
                    pods rebound off each other.
        Friction: the current speed vector of each pod is multiplied by 0.85
        The speed's values are truncated and the position's values are rounded to the nearest integer.

        """

        # Rotation
        target_angle = self.pos.angle_deg(target)
        angle_diff = target_angle - self.angle
        if angle_diff > 180:
            angle_diff -= 360
        elif angle_diff < -180:
            angle_diff += 360
        if abs(angle_diff) > 18:
            self.angle += 18 * angle_diff / abs(angle_diff)
        else:
            self.angle = target_angle

        accel = thrust * Vec2D.from_angle_deg(self.angle)
        # Acceleration
        self.vel = self.vel + accel
        # Movement
        self.pos = self.pos + self.vel

    def rebound(self, pods: list[Pod]) -> None:
        """
        rebound with other pod
        Collisions are elastic. The minimum impulse of a collision is 120.
        A boost is in fact an acceleration of 650. The number of boost available is common between pods. If no boost is
        available, the maximum thrust is used.
        A shield multiplies the Pod mass by 10.
        The provided angle is absolute. 0° means facing EAST while 90° means facing SOUTH.
        """

        for pod in pods:
            if pod == self:
                continue
            if pod.destroyed:
                continue
            if self.pos.distance_squared(pod.pos) < POD_RADIUS_SQUARED * 4:
                # Collision
                # Calculate new velocities
                m1 = 1
                m2 = 1
                if self.shield == 3:
                    m1 = 10
                if pod.shield == 3:
                    m2 = 10
                v1 = self.vel
                v2 = pod.vel
                x1 = self.pos
                x2 = pod.pos
                total_mass = m1 + m2
                dv = v1 - v2
                dx = x1 - x2
                v1_new = v1 - 2 * m2 / total_mass * dv.dot(dx) / dx.norm_squared() * dx
                v2_new = v2 - 2 * m1 / total_mass * dv.dot(dx) / dx.norm_squared() * dx
                self.vel = v1_new
                pod.vel = v2_new
                # Move pods
                self.pos = self.pos + self.vel
                pod.pos = pod.pos + pod.vel

    def friction_and_truncation(self) -> None:

        # Friction and truncation
        self.vel = self.vel * 0.85
        self.vel = self.vel.floor()
        self.pos = round(self.pos)

    def update_shield_counter(self):
        self.shield = max(0, self.shield - 1)

    def update_cp(self, checkpoints: list[Vec2D]):
        """Updated timeout and next_cp_id if the pod passed a checkpoint."""
        if self.pos.distance(checkpoints[self.next_cp_id]) < CHECKPOINT_RADIUS:
            self.timeout = 0
            self.next_cp_id = (self.next_cp_id + 1) % len(checkpoints)
        else:
            self.timeout += 1
        if self.timeout >= 100:
            self.destroyed = True

    def finalize_move(self, checkpoints: list[Vec2D]):
        """Updates the pod after a move. Convenience method for move_pods."""
        self.friction_and_truncation()
        self.update_cp(checkpoints)
        self.update_shield_counter()

    def angle_to_target(self, target: Vec2D) -> float:
        """Returns the angle to the target in degrees considering the pod's current angle in deg."""
        angle = self.pos.angle_deg(target)
        angle_diff = angle - self.angle
        if angle_diff > 180:
            angle_diff -= 360
        elif angle_diff < -180:
            angle_diff += 360
        return angle_diff

    def angle_to_target_rad(self, target: Vec2D) -> float:
        """Returns the angle to the target in radians considering the pod's current angle in deg."""
        return math.radians(self.angle_to_target(target))

    def relative_to_pod(self, other: Vec2D) -> Vec2D:
        """Transforms the coordinates of the other vector to the coordinate system of the pod."""
        return Vec2D(other.x - self.pos.x, other.y - self.pos.y).rotate_deg(-self.angle)


def str_to_move(move_str: str) -> PodMove:
    """Converts a string to a PodMove object."""
    if move_str == "BOOST":
        return PodMove(MoveType.BOOST)
    elif move_str == "SHIELD":
        return PodMove(MoveType.SHIELD)
    else:
        thrust, target_x, target_y = map(int, move_str.split())
        return PodMove(MoveType.THRUST, thrust, Vec2D(target_x, target_y))


class GameState:
    """
    GameState class which holds the game state, e.g. the pods, the checkpoints and the number of laps.
    """

    def __init__(self, laps: int, checkpoints: list[Vec2D], pods: list[Pod]):
        self.laps = laps
        self.checkpoints = checkpoints
        self.pods = pods

    def move_pods(self, pod_moves: list[PodMove]):
        """
        Moves the pods according to the given moves.
        """
        for pod, move in zip(self.pods, pod_moves):
            if pod.destroyed:
                continue
            if move.type == MoveType.THRUST:
                pod.move(move.thrust, move.target)
            elif move.type == MoveType.SHIELD:
                pod.activate_shield()
            elif move.type == MoveType.BOOST:
                pod.activate_boost()
            else:
                raise ValueError("Unknown move type")
            # Rebound
        for pod in self.pods:
            if pod.destroyed:
                continue

            pod.rebound(self.pods)
        # Finalize moves
        for pod in self.pods:
            if pod.destroyed:
                continue
            pod.finalize_move(self.checkpoints)

    def check_win_condition(self) -> int:
        """
        Checks if a pod has won the game. Returns the
        index of the winning pod or -1 if no pod has won yet.
        """
        for pod in self.pods:
            if pod.checkpoints_passed == self.laps * len(self.checkpoints):
                print(f"Pod {pod.id} from team {pod.team_id} has won the game.")
                return pod.team_id
        # check if both pods are destroyed from team 0
        if len(self.pods) < 2:
            return -1
        if self.pods[0].destroyed and self.pods[1].destroyed:
            print("Both pods from team 0 are destroyed. Team 1 wins.")
            return 1
        # check if both pods are destroyed from team 1
        if self.pods[2].destroyed and self.pods[3].destroyed:
            print("Both pods from team 1 are destroyed. Team 0 wins.")
            return 0

        return -1


class GameView:
    """ Draw game state using tkinter"""

    def __init__(self, game_state: GameState, xres: int = 800,
                 world_width=MAP_WIDTH, world_height=MAP_HEIGHT) -> None:
        self.root = tk.Tk()
        self.root.title("Codingame Pod Racing")
        self.root.resizable(False, False)
        self.xres = xres
        # scaled to world size
        self.yres = yres = int(round(xres * world_height / world_width))
        self.canvas = tk.Canvas(self.root, width=xres, height=yres, bg="white")
        self.canvas.pack()

        self.world_width = world_width
        self.world_height = world_height
        # create checkpoints
        self.draw_checkpoints(game_state)
        self.pod_id_mapping = {}  # maps pod id to canvas item id
        # create pods
        self.draw_pods(game_state)
        self.last_update = time.time()

    def map_world2screen(self, x: float, y: float) -> tuple[int, int]:
        """Maps world coordinates to screen coordinates."""
        x_screen = int(x / self.world_width * self.xres)
        y_screen = int(y / self.world_height * self.yres)
        return x_screen, y_screen

    def update_game_state(self, game_state: GameState) -> None:
        """Draws the game state."""
        self.update_pods(game_state)
        self.root.update_idletasks()

        self.root.update()
        self.last_update = time.time()

    def sync_fps(self, fps: float) -> None:
        """Syncs the fps to the given value."""
        elapsed = time.time() - self.last_update
        if elapsed < 1 / fps:
            time.sleep(1 / fps - elapsed)

    def draw_checkpoints(self, game_state: GameState) -> None:
        """Draws the checkpoints."""
        for checkpoint in game_state.checkpoints:
            lx, ly = self.map_world2screen(checkpoint.x - CHECKPOINT_RADIUS,
                                           checkpoint.y - CHECKPOINT_RADIUS)
            rx, ry = self.map_world2screen(checkpoint.x + CHECKPOINT_RADIUS,
                                           checkpoint.y + CHECKPOINT_RADIUS)

            self.canvas.create_oval(lx, ly, rx, ry, fill="YELLOW")

    def draw_pods(self, game_state: GameState) -> None:
        """Draws the pods."""
        for pod in game_state.pods:
            lx, ly = self.map_world2screen(
                pod.pos.x - POD_RADIUS, pod.pos.y - POD_RADIUS)
            rx, ry = self.map_world2screen(
                pod.pos.x + POD_RADIUS, pod.pos.y + POD_RADIUS)
            color = "RED" if pod.team_id == 0 else "BLUE"
            pod_item_id = self.canvas.create_oval(
                lx, ly, rx, ry, fill=color)
            px, py = self.map_world2screen(
                pod.pos.x, pod.pos.y)

            # calc target by using pod pos and current angle
            target = pod.pos + Vec2D.from_angle_deg(pod.angle) * POD_RADIUS

            tx, ty = self.map_world2screen(
                target.x, target.y)

            p_line_item = self.canvas.create_line(
                px, py, tx, ty, fill="BLACK")
            self.pod_id_mapping[pod.id] = pod_item_id, p_line_item

    def update_pods(self, game_state: GameState) -> None:
        """Updates the pods."""
        for pod in game_state.pods:
            if pod.id not in self.pod_id_mapping:
                continue
            pod_canvas_id, p_line_item = self.pod_id_mapping[pod.id]
            if pod.destroyed:
                # delete pod from canvas
                self.pod_id_mapping.pop(pod.id)
                self.canvas.delete(pod_canvas_id)
                self.canvas.delete(p_line_item)
                continue
            lx, ly = self.map_world2screen(
                pod.pos.x - POD_RADIUS, pod.pos.y - POD_RADIUS)
            self.canvas.moveto(pod_canvas_id, lx, ly)

            px, py = self.map_world2screen(
                pod.pos.x, pod.pos.y)

            # calc target by using pod pos and current angle
            target = pod.pos + Vec2D.from_angle_deg(pod.angle) * POD_RADIUS

            tx, ty = self.map_world2screen(
                target.x, target.y)
            self.canvas.coords(p_line_item, px, py, tx, ty)


class Game:

    def __init__(self, game_state: GameState) -> None:
        self.game_state = game_state
        self.counter = 0

    def tick(self, pod_moves: list[PodMove]) -> None:
        """
        Performs a game tick.
        """
        self.counter += 1
        self.game_state.move_pods(pod_moves)
        winner_id = self.game_state.check_win_condition()
        if winner_id >= 0:
            exit(0)


# env state contains next_state, next_reward, done, info =
EnvResponse = tuple[np.array, float, bool, dict]

"""
    ACTION_LIST = [thrust 200 ; rotate left ]
[thrust 200 ; no rotation ]
[thrust 200 ; rotate right]
[no thrust  ; rotate left ]
[no thrust  ; no rotation ]
[no thrust  ; rotate right]

"""


class ActionSpace:
    def __init__(self, n: int) -> None:
        self.n = n

    def sample(self) -> int:
        return random.randint(0, self.n - 1)


class GameEnvSingleRunner:
    """ Wrapper for training in CSB environment, provides state reset, and rewards"""
    ROTATE_MAPPING = {0: 0, 1: 90, 2: -90}
    action_space = ActionSpace(6)
    observation_space = np.zeros((7,))

    def __init__(self):
        self.game_state = create_random_start_game_state(num_pods=1)
        self.counter = 0

    def reset(self):
        self.game_state = create_random_start_game_state(num_pods=1)
        self.counter = 0
        return self.get_state_as_numpy()

    def get_state_as_numpy(self) -> np.array:
        """ Returns state as numpy array, by placing the racer at 0,0 and rotating other elements around it"""
        racer = self.game_state.pods[0]
        # get the next two checkpoints
        checkpoints = self.game_state.checkpoints
        cp1 = checkpoints[racer.next_cp_id]
        cp2 = checkpoints[(racer.next_cp_id + 1) % len(checkpoints)]
        # transorm the checkpoints to be relative to the racer

        cp1 = cp1 - racer.pos
        cp2 = cp2 - racer.pos
        # normalize angle
        angle = racer.angle / 360
        # normalize speed

        return np.array([racer.vel.x, racer.vel.y, angle, cp1.x, cp1.y, cp2.x, cp2.y])

    def calc_reward(self) -> float:
        """ Calculates the reward for the current state"""
        # give reward when checkpoint was reached
        racer = self.game_state.pods[0]
        # if timeout is zero, we reached a checkpoint except for start of game
        # calc distance to next cp
        checkpoints = self.game_state.checkpoints
        cp1 = checkpoints[racer.next_cp_id]
        cp2 = checkpoints[(racer.next_cp_id + 1) % len(checkpoints)]
        dist_to_cp1 = racer.pos.distance(cp1) / MAP_WIDTH
        # the reward is the negative distance to the next checkpoint
        reward = -dist_to_cp1
        if racer.timeout == 0 and self.counter > 0:
            return 100
        if racer.timeout == 100:
            return -1000
        return reward

    def step(self, action: int) -> EnvResponse:
        # pick action and translate to world action
        thrust = 200 if action <= 2 else 0
        rotate_id = action % 3
        rot_angle = self.ROTATE_MAPPING[rotate_id]
        racer = self.game_state.pods[0]
        target = racer.pos + Vec2D.from_angle_deg(self.game_state.pods[0].angle + rot_angle) * 1000
        pod_move = PodMove(MoveType.THRUST, thrust, target)
        self.game_state.move_pods([pod_move])
        self.counter += 1
        # env state contains next_state, next_reward, done, info
        return self.get_state_as_numpy(), self.calc_reward(), racer.destroyed, {}


def main_test_game_env(agent) -> None:
    env = GameEnvSingleRunner()
    env.reset()
    game_view = GameView(env.game_state)

    game_view.update_game_state(env.game_state)
    state = env.get_state_as_numpy()

    for _ in range(100):
        # get action from agent
        state = torch.from_numpy(state).to(torch.device("cpu"), torch.float)
        action, agent_info = agent(state)
        state, reward, done, info = env.step(action)

        game_view.update_game_state(env.game_state)
        game_view.sync_fps(30)
        print(state, reward, done, info)


def main_local():
    # create initial game state
    game_state = create_random_start_game_state(num_pods=1)
    game_view = GameView(game_state)
    game = Game(game_state)
    while True:
        # create moves
        start = time.time()
        pod_moves = create_moves_to_id(game_state)
        # perform game tick
        game.tick(pod_moves)
        # print debug info / next id of pod
        pod1 = game_state.pods[0]
        print(f"tick {game.counter} cp_id: {pod1.next_cp_id} t:{pod1.timeout}")

        print(f"speed {pod1.vel.norm()}")
        # draw game state
        elapsed = time.time() - start
        frame_time = 1 / FPS
        wait_time = frame_time - elapsed
        if wait_time > 0:
            time.sleep(wait_time)
        game_view.update_game_state(game_state)


def create_debug_game_state():
    # create 3 checkpoints
    checkpoints = [
        Vec2D(2000, 2000),
        Vec2D(2000, MAP_HEIGHT - 700),
        Vec2D(MAP_WIDTH - 700, MAP_HEIGHT - 700),
    ]
    # create 1 pods
    pods = [
        Pod(0, 0, Vec2D(MAP_WIDTH // 2, MAP_HEIGHT // 2), Vec2D(0, 0), angle=0)
    ]
    return GameState(3, checkpoints, pods)


def create_random_start_game_state(num_pods=4):
    laps = 3
    num_checkpoints = random.randint(2, 8)
    checkpoints = []
    for i in range(num_checkpoints):
        too_close = True
        new_cp = random_checkpoint()

        while too_close:
            new_cp = random_checkpoint()
            too_close = False
            for checkpoint in checkpoints:
                if new_cp.distance(checkpoint) < 2 * CHECKPOINT_RADIUS:
                    too_close = True
                    break

        checkpoints.append(new_cp)
    # create pods
    pods = []
    for i in range(num_pods):
        team_id = 0 if i < 2 else 1
        rand_x = random.randint(0, MAP_WIDTH)
        rand_y = random.randint(0, MAP_HEIGHT)
        pods.append(Pod(i, team_id, Vec2D(rand_x, rand_y), Vec2D(0, 0), i))
    game_state = GameState(laps, checkpoints, pods)
    return game_state


def random_checkpoint():
    rand_x = random.randint(
        CHECKPOINT_RADIUS, MAP_WIDTH - 1 - CHECKPOINT_RADIUS)
    rand_y = random.randint(
        CHECKPOINT_RADIUS, MAP_HEIGHT - 1 - CHECKPOINT_RADIUS)
    # check if checkpoint is too close to another checkpoint
    new_cp = Vec2D(rand_x, rand_y)
    return new_cp


def create_moves_to_id(game_state: GameState) -> list[PodMove]:
    """ Creates moves to the next checkpoint"""
    pod_moves = []
    for pod in game_state.pods:
        # get next checkpoint
        next_cp = game_state.checkpoints[pod.next_cp_id]
        # create move to next checkpoint
        pod_move = PodMove(MoveType.THRUST, 100, next_cp)
        pod_moves.append(pod_move)
    return pod_moves


def create_random_moves(game_state) -> list[PodMove]:
    pod_moves = []
    for pod in game_state.pods:
        if pod.destroyed:
            continue
        move_type = random.choice(
            [MoveType.THRUST, MoveType.SHIELD, MoveType.BOOST])
        if move_type == MoveType.THRUST:
            thrust = random.randint(0, 200)
            target = Vec2D(random.randint(0, 16000), random.randint(0, 9000))
            pod_moves.append(PodMove(MoveType.THRUST, thrust, target))
        else:
            pod_moves.append(PodMove(move_type))
    return pod_moves


if __name__ == "__main__":
    # create initial game state
    # main_local()
    from agent import agent

    agent = agent.to(torch.device("cpu"), torch.float)
    agent.load_weights("weights/weights_final.pt")
    main_test_game_env(agent)
