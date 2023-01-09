from typing import Callable
import time

from main import GameState, GameView, Vec2D, create_debug_game_state, PodMove, MoveType


FPS = 10


def vizualize_prop(func: Callable) -> None:
    state = create_debug_game_state()
    view = GameView(state)
    while True:
        start = time.time()
        func(state)
        # wait to keep the FPS
        elapsed = time.time() - start
        frame_time = 1 / FPS
        wait_time = frame_time - elapsed
        if wait_time > 0:
            time.sleep(wait_time)
        view.update_game_state(state)


def rotate_pod(game_state: GameState) -> None:
    pod = game_state.pods[0]
    pod.angle += 1


def move_pod(game_state: GameState) -> None:
    pod = game_state.pods[0]
    pod.pos.y += 10


def move_pod_by_target(game_state: GameState) -> None:
    pod = game_state.pods[0]
    target = pod.pos + Vec2D(-100, 0)
    pod.move(100, target)
    pod.friction_and_truncation()


if __name__ == '__main__':
    vizualize_prop(move_pod_by_target)
