import sys
import math
import numpy as np
from numpy.linalg import norm
import itertools
import time
from copy import deepcopy, copy

import multiprocessing as mp
from collections import defaultdict

# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.


force_radius = 400
check_radius = 600
field_w = 16000
field_h = 9000
lookahead = 15
max_duration = lookahead // 2
max_time_init = 1000
time_per_tick = 70
num_checkpoints = None  # will be initialized by game loop, evil, but easy
laps = None

time_dict = defaultdict(list)


def eval_times(total_time_global):
    for key, values in sorted(time_dict.items(), key= lambda x: sum(x[1]), reverse=True):
        total_time = sum(values)
        hits = len(values)
        time_per_hit = total_time / hits
        percentage = total_time / total_time_global
        print(f"{key : <24}: {percentage * 100:4.1f} % {total_time * 1000:7.3f} {time_per_hit * 1000:7.3f} {hits:4}",
              file=sys.stderr)


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        dur = time.time() - start
        # print(f"{func.__name__}: {dur*1000:.3f} ms", file=sys.stderr)
        time_dict[func.__name__].append(dur)
        return ret

    return wrapper


#@timeit
def closest_point(a, b, p0):
    da = b[1] - a[1]
    db = a[0] - b[0]
    c1 = da * a[0] + db * a[1]
    c2 = -db * p0[0] + da * p0[1]
    det = da * da + db * db
    cx = 0
    cy = 0

    if (det != 0):
        cx = (da * c1 - db * c2) / det
        cy = (da * c2 + db * c1) / det
    else:
        # The point is already on the line
        cx = p0[0];
        cy = p0[1];

    return np.array([cx, cy]);


def distance2(p1, p2):
    dist = [(a - b) ** 2 for a, b in zip(p1, p2)] # seems to be fasted implementation
    # dist2 = np.sum((p1 - p2) * (p1 - p2))
    return sum(dist)


def distance(p1, p2):
    return math.sqrt(distance2(p1, p2))


def mean_pos(*positions):
    avg_x = 0
    avg_y = 0
    N = len(positions)
    for pos in positions:
        avg_x += pos[0]
        avg_y += pos[1]
    return (avg_x / N, avg_y / N)


class PodTeam(object):

    def __init__(self, pod1, pod2):
        self.pod1 = pod1
        self.pod2 = pod2
        self.timeout = 100


class CP(object):
    def __init__(self, pos):
        self.pos = pos
        self.vel = np.array([0, 0])


class Pod(object):

    def __init__(self, pod):
        self.pos = np.array(pod[:2])
        self.angle = pod[4]
        self.vel = np.array(pod[2:4])
        self.next_target_id = pod[5]
        self.shield = 0
        self.boosted = False
        self.timeout = 100
        self.checked = 0
        self.partner = None
        self.behaviour = ''
        self.save()

    #@timeit
    def save(self):
        self.last_pos = copy(self.pos)
        self.last_vel = copy(self.vel)
        self.last_checked = copy(self.checked)
        self.last_timeout = copy(self.timeout)
        self.last_shield = copy(self.shield)
        self.last_next_target_id = copy(self.next_target_id)
        self.last_angle = copy(self.angle)
        self.last_behaviour = copy(self.behaviour)

    #@timeit
    def load(self):
        self.pos = copy(self.last_pos)
        self.vel = copy(self.last_vel)
        self.checked = copy(self.last_checked)
        self.timeout = copy(self.last_timeout)
        self.shield = copy(self.last_shield)
        self.next_target_id = copy(self.last_next_target_id)
        self.angle = copy(self.last_angle)
        self.behaviour = copy(self.last_behaviour)

    def set_partner(self, partner):
        self.partner = partner

    def update(self, pod):
        next_target_id = pod[5]
        if (self.shield > 0):
            self.shield -= 1
        if (next_target_id != self.next_target_id):
            self.timeout = 100
            self.partner.timeout = 100
            self.checked += 1
        else:
            self.timeout -= 1
        self.pos = np.array(pod[:2])
        self.angle = pod[4]
        self.vel = np.array(pod[2:4])
        self.next_target_id = pod[5]
        self.save()

    def distance2(self, p):
        dist = [(a - b) ** 2 for a, b in zip(self.pos, p)]
        return sum(dist)

    def distance(self, p):
        return math.sqrt(self.distance2(p))

    def rotate_by_angle(self, angle):
        a = angle + self.angle
        if (a >= 360.0):
            a = a - 360.0
        elif (a < 0.0):
            a += 360.0
        self.angle = a

    def move_angle2point(self, move_angle):
        a = self.angle + move_angle

        if (a >= 360.0):
            a = a - 360.0
        elif (a < 0.0):
            a += 360.0
        a = a * np.pi / 180.0
        dp = np.array([np.cos(a), np.sin(a)])
        pos = self.pos + dp * 4000.0
        return pos

    #@timeit
    def runner_score(self, check_point_list):

        next_p = check_point_list[self.next_target_id]
        nnext_p = check_point_list[(self.next_target_id + 1) % num_checkpoints]

        dist = self.distance(next_p)
        # nn_dist = self.distance(nnext_p)
        score = self.checked * 50000 - dist
        # ~ score += self.timeout * 1e4
        # ~ if dist2 < 2000**2:
        # ~ pref_entr_vec = nnext_p - next_p
        # ~ pref_entr_vec = pref_entr_vec/np.linalg.norm(pref_entr_vec)
        # ~ vel = self.vel / np.linalg.norm(self.vel)
        # ~ score += np.dot(self.vel, vel) * 1000
        # score += self.timeout * 10
        return score * 10

    #@timeit
    def hunter_score(self, er, eh, check_point_list):
        er_next_p = check_point_list[er.next_target_id]
        # er_nnext_p = check_point_list[(er.next_target_id+1)%num_checkpoints]
        # enem_cp_dist = er.distance( er_next_p  )
        # reh_dist = eh.distance(self.partner.pos)
        score = 0
        # score -= np.dot(er.pos - next_p, er.vel)
        # score -= self.distance2(er_next_p) * 5
        # score -= self.diffAngle(er.pos) * 5
        # score -= er.runner_score(check_point_list)
        dist_ernc = self.distance(er_next_p)
        # dist_er = self.distance(er.pos)
        score -= self.diffAngle(er.pos) * 10
        score -= dist_ernc
        # if dist_ernc < 800:
        score -= self.distance(er.pos)

        # ~ if dist_er * 3 < dist_ernc:
        # ~ score -= self.distance(er.pos)*2
        # ~ else:
        # ~ score -= self.distance(er_next_p)

        # score += reh_dist*3
        # ~ if self.timeout < 20:
        # ~ dist = self.distance(next_p)
        # ~ score = self.checked * 50000 - dist

        # score += enem_cp_dist * 10
        # score += reh_dist *5

        # score += er.distance(next_p) * 10
        return score

    #@timeit
    def collide_pod(self, other):
        dist2 = self.distance2(other.pos)
        sr = 640000
        if dist2 < sr:
            return other.pos, 0  # place and time for collision

        ac = other.pos - self.pos
        d = np.dot(ac, other.vel - self.vel)
        normal_dist2 = np.sum((d - other.pos) ** 2)
        if normal_dist2 < sr:
            return other.pos, math.sqrt(dist2) / norm(self.vel)
        return None

    #@timeit
    def bounce(self, other):
        m1 = 1
        m2 = 1
        if self.shield:
            m1 = 10
        if other.shield:
            m2 = 10

        mcoeff = (m1 + m2) / (m1 * m2)

        n = self.pos - other.pos

        # Square of the distance between the 2 pods. This value could be hardcoded because it is always 800²
        nxnysquare = 800 ** 2

        dv = self.vel - other.vel

        # fx and fy are the components of the impact vector. product is just there for optimisation purposes
        product = np.sum(n * dv)
        f = (n * product) / (nxnysquare * mcoeff)

        # We apply the impact vector once
        self.vel = self.vel - f / m1
        other.vel = other.vel + f / m2

        # If the norm of the impact vector is less than 120, we normalize it to 120
        impulse = norm(f)

        if (0 < impulse < 120.0):
            f = f * 120.0 / impulse
            self.vel = self.vel - f / m1
            other.vel = other.vel + f / m2

    #@timeit
    def get_angle(self, p):

        # cos_alpha = np.dot(self.pos, p) / (np.linalg.norm(self.pos) * np.linalg.norm(p) )
        # a = np.arccos(cos_alpha) * 180.0 / np.pi
        # a = cos_alpha
        # old code check for correctness
        # dist = [(a - b) ** 2 for a, b in zip(self.pos, p)]
        # dist = sum(dist)
        # d = math.sqrt(dist)
        d = distance(self.pos, p)
        diff = (p - self.pos) / d
        # Simple trigonometry. We multiply by 180.0 / PI to convert radiants to degrees.
        a = math.acos(diff[0]) * 180.0 / np.pi
        # If the point I want is below me, I have to shift the angle for it to be correct
        if (diff[1] < 0):
            a = 360.0 - a
        return a

    #@timeit
    def diffAngle(self, p):
        a = self.get_angle(p)

        # To know whether we should turn clockwise or not we look at the two ways and keep the smallest
        # The ternary operators replace the use of a modulo operator which would be slower
        # right = this.angle <= a ? a - this.angle : 360.0 - this.angle + a;
        right = a - self.angle if self.angle <= a else 360.0 - self.angle + a
        # left = this.angle >= a ? this.angle - a : this.angle + 360.0 - a;
        left = self.angle - a if self.angle >= a else self.angle + 360.0 - a
        if (right < left):
            return right
        else:
            # We return a negative angle if we must rotate to left
            return -left

    #@timeit
    def rotate(self, p):
        a = self.diffAngle(p)
        # print("Rotate: (angle, diffangle): ({:.3f},{:.3f}".format(self.angle, a), file = sys.stderr)
        ## Can't turn by more than 18° in one turn
        a = np.clip(a, -18.0, 18.0)
        self.angle += a;

        # The % operator is slow. If we can avoid it, it's better.
        if (self.angle >= 360.0):
            self.angle = self.angle - 360.0
        if self.angle < 0.0:
            self.angle += 360.0

    #@timeit
    def boost(self, thrust):

        if thrust == 'SHIELD':
            self.shield = 3

        if self.shield > 0:
            thrust = 0

        if thrust == 0:
            return

        if thrust == 'BOOST':
            thrust = 200
            if not self.boosted:
                thrust = 650
                self.boosted = True

        rad = self.angle / 180.0 * np.pi
        self.vel = self.vel + np.array([np.cos(rad), np.sin(rad)]) * thrust

    def predict_move(self, t=1.0):
        # self.rotate(target)

        pos = self.pos + self.vel * t
        return pos
        # self.speed *= 0.85
        # self.pos = self.pos.astype(np.int)

    def move(self, t=1.0):
        # self.rotate(target)

        self.pos = self.pos + self.vel * t
        # self.speed *= 0.85
        # self.pos = self.pos.astype(np.int)

    def end(self):
        self.pos = np.round(self.pos)
        self.vel = (self.vel * 0.85).astype(int)

        self.timeout -= 1
        if (self.shield > 0):
            self.shield -= 1

    def hit_check(self):
        self.checked += 1
        self.timeout = 100
        self.partner.timeout = 100
        self.next_target_id = (self.next_target_id + 1) % num_checkpoints

    #@timeit
    def collide_check(self, check_point_list):
        cp = check_point_list[self.next_target_id]
        dist2 = self.distance2(cp)
        sr = 359000

        if dist2 < sr:
            return cp, 0  # place and time for collision

        ac = cp - self.pos
        d = np.dot(ac, self.vel)
        normal_dist2 = np.sum((d - cp) ** 2)
        if normal_dist2 < sr:
            return cp, math.sqrt(dist2) / norm(self.vel)
        return None

    # def play_until(self, p, thrust, t):
    #     self.rotate(p)
    #     self.boost(thrust)
    #     self.move(t)
    #
    # def play(self, p, thrust):
    #     self.rotate(p)
    #     self.boost(thrust)
    #     self.move(1.0)
    #     self.end()

    def __str__(self):
        return "Pos: {} Vel:{} Angle: {} ".format(self.pos, self.vel, self.angle)

    def __repr__(self):
        return self.__str__()


class Move(object):

    def __init__(self, target, thrust, duration):
        self.target = target
        self.thrust = thrust
        self.duration = duration

    def mv_str(self):
        tx = round(int(self.target[0]))
        ty = round(int(self.target[1]))
        return f"{tx} {ty} {self.thrust}"

    def __str__(self):
        return "Target: {} Thrust:{}, Duration:{}".format(self.target, self.thrust, self.duration)

    def __repr__(self):
        return self.__str__()


def angle(vec1, vec2):
    dp = np.sum(vec1 * vec2) / (norm(vec1) * norm(vec2))
    # print("dp:", dp, vec1, vec2, file=sys.stderr)

    if dp == 1.0:
        return 0.0
    return np.arccos(dp) / np.pi * 180


#@timeit
def get_move_racer(pod, check_point_list, duration=5):
    next_id = pod.next_target_id
    next_next_id = (next_id + 1) % len(check_point_list)
    next_check = np.array(check_point_list[next_id])
    nnext_check = np.array(check_point_list[next_next_id])
    next_checkpoint_angle = pod.diffAngle(next_check)
    next_checkpoint_dist = pod.distance(next_check)
    thrust = 200
    # print("target angle: ", next_checkpoint_angle, file = sys.stderr)
    if np.abs(next_checkpoint_angle) > 90:
        thrust = 0
    # tx, ty = next_checkpoint_x + anti_target[0], next_checkpoint_y + anti_target[1]
    target = next_check
    if next_checkpoint_dist < 4000:
        nn_tvec = (nnext_check - next_check)
        nn_norm = norm(nn_tvec)
        if nn_norm > 0:
            nn_tvec = nn_tvec / norm(nn_tvec)

        target = target + nn_tvec * (check_radius - 50)
    if next_checkpoint_dist > 4000 and abs(next_checkpoint_angle) < 15:  # and opponent_dist < next_checkpoint_dist:
        thrust = 'BOOST'
        used_boost = True
    # print("coll_friend: ", coll_friend, file = sys.stderr)
    # print("coll_enem_at: ", coll_enem_at, file = sys.stderr)
    # print("coll_enem_run: ", coll_enem_run, file = sys.stderr)
    return Move(target, thrust, duration)


#@timeit
def get_move_defender(pod, er, check_point_list, duration=5):
    # move to closest target of er, which can be reacher
    # if there keep runner of
    DEFENCE_DIST = 4 * check_radius
    enem_curr_target = check_point_list[er.next_target_id]
    if pod.distance(enem_curr_target) < 2000:
        target = er.pos
    else:
        e_id = (er.next_target_id + 1) % len(check_point_list)
        target_cp = check_point_list[e_id]
        t_vec = er.pos - target_cp
        dist = np.linalg.norm(t_vec)
        target = t_vec * DEFENCE_DIST / dist

    target_angle = pod.diffAngle(target)
    dist = pod.distance(target)
    # print("target angle: ", target_angle, file = sys.stderr)
    thrust = 200
    if abs(target_angle) > 90:
        thrust = 0
    if abs(target_angle) < 5 and dist > 2000:
        thrust = 'BOOST'

    return Move(target, thrust, duration)


def estimate_roles(pods):
    pod1, pod2, enem_pod1, enem_pod2 = pods
    er = enem_pod1
    ea = enem_pod2
    sr = pod1
    sa = pod2
    if ea.checked > er.checked:
        er, ea = ea, er

    if sa.checked > sr.checked:
        sr, sa = sa, sr
    sr.behaviour = 'runner'
    er.behaviour = 'runner'
    sa.behaviour = 'attacker'
    ea.behaviour = 'attacker'

    return sr, sa, er, ea


#@timeit
def simulate(pods, pod_moves, check_point_list):
    score = 0
    for turn in range(lookahead):
        ea, er, sa, sr = simulate_turn(check_point_list, pod_moves, pods, turn)
        # score += overall_score([sr, sa, er, ea], check_point_list)
        score += sr.runner_score(check_point_list) - er.runner_score(check_point_list)
        score += sa.hunter_score(er, ea, check_point_list) - ea.hunter_score(sr, sa, check_point_list)
        # score += sa.runner_score( check_point_list) - ea.runner_score( check_point_list)
        # score += sa.runner_score(check_point_list)
        # score -= er.runner_score(check_point_list)
        # score -= ea.hunter_score(sr, sa, check_point_list)
    return score

#@timeit
def simulate_turn(check_point_list, pod_moves, pods, turn):
    sr, sa, er, ea = estimate_roles(pods)
    k = -1
    for pod, moves in zip(pods, pod_moves):
        k += 1
        if not moves:  # simple runner pod
            if pod.behaviour == 'runner':
                move = get_move_racer(pod, check_point_list, duration=1)
            elif pod.behaviour == 'attacker':
                move = get_move_defender(pod, er, check_point_list, duration=1)
            else:
                raise NotImplementedError("Missing behaviour.")

        else:
            move = turn2move(moves, turn)
        pod.rotate(move.target)
        pod.boost(move.thrust)
        if pod.collide_check(check_point_list):
            pod.hit_check()
    for i, j in itertools.combinations(range(4), 2):
        col = pods[i].collide_pod(pods[j])
        if col and col[1] < 1.0:
            pods[i].bounce(pods[j])
    for pod in pods:
        pod.move()
        pod.end()
    return ea, er, sa, sr


#@timeit
def overall_score(pods_roles, check_point_list):
    sr, sa, er, ea = pods_roles
    score = 0
    if sr.timeout < 20:
        self_run_dist = sr.distance(check_point_list[sr.next_target_id])
        self_hunt_dist = sa.distance(check_point_list[sa.next_target_id])
        score -= (self_run_dist + self_hunt_dist) * 10

    if er.timeout < 20:
        er_dist = er.distance(check_point_list[er.next_target_id])
        ea_dist = ea.distance(check_point_list[ea.next_target_id])
        score += (er_dist + ea_dist) * 10
    er_dist = er.distance(check_point_list[er.next_target_id])
    sr_dist = sr.distance(check_point_list[sr.next_target_id])
    run_diff = (sr.checked - er.checked) * 500000 + (er_dist - sr_dist)
    score += run_diff
    time_e = er_dist / math.sqrt(np.sum(er.vel ** 2))
    hunter_enem_check_dist2 = sa.distance2(check_point_list[er.next_target_id])
    time_h = hunter_enem_check_dist2 / math.sqrt(np.sum(sa.vel ** 2))
    if time_e > time_h:
        score -= hunter_enem_check_dist2
    else:  # go to second next cp
        hunter_enem_check_dist2 = sa.distance2(check_point_list[(er.next_target_id + 1) % num_checkpoints])
        score -= hunter_enem_check_dist2
    return score


class Solution(object):

    def __init__(self, pods):
        self.moves1 = generate_random_moves()
        self.moves2 = generate_random_moves()

        self.pods = pods
        self.score = -np.inf
        # self.total_random()

    def create_direct_bot_solution(self, check_point_list):
        sr, sa, er, ea = estimate_roles(self.pods)
        run_move = get_move_racer(sr, check_point_list)
        attack_move = get_move_defender(sa, er, check_point_list)
        if self.pods[0].behaviour == 'runner':
            self.moves1[0] = run_move
            self.moves2[0] = attack_move
        else:
            self.moves1[0] = attack_move
            self.moves2[0] = run_move
        self.score = -np.inf
        self.grow_moves()

    def reset(self):
        for pod in self.pods:
            pod.load()

    def save(self):
        for pod in self.pods:
            pod.save()

    #@timeit
    def get_score(self, check_point_list):
        # self.save()

        moves = [self.moves1,
                 self.moves2, None, None]
        self.score = simulate(self.pods, moves, check_point_list)
        self.reset()
        return self.score

    def shift(self):

        self.moves1[0].duration -= 1
        self.moves2[0].duration -= 1
        if self.moves1[0].duration <= 0:
            self.moves1.pop(0)
        if self.moves2[0].duration <= 0:
            self.moves2.pop(0)
        self.grow_moves()
        self.score = -np.inf

    def grow_moves(self):
        while moves2turns(self.moves1) < lookahead:
            self.moves1.append(generate_random_move())
        while moves2turns(self.moves2) < lookahead:
            self.moves2.append(generate_random_move())

    def create_child(self):
        child = Solution(self.pods)
        child.moves1 = deepcopy(self.moves1)
        child.moves2 = deepcopy(self.moves2)
        child.mutate()
        return child

    def mutate(self, alpha=0.5):
        # alpha mutation degree 0.0: nothing, 1.0: all
        for move in (self.moves1 + self.moves2):
            if np.random.random() < alpha:
                move.target = generate_random_target()
            if np.random.random() < alpha:
                move.thrust = generate_random_thrust()
            if np.random.random() < alpha:
                move.duration = np.random.randint(1, max_duration)
        self.grow_moves()
        self.score = -np.inf

    def __str__(self):
        string = str(self.score) + ' \n'
        string += "{}\n" * len(self.moves1 + self.moves2)
        return string.format(*(self.moves1 + self.moves2))

    def __repr__(self):
        return self.__str__()

#@timeit
def turn2move(moves, turn):
    dur = 0
    for m in moves:
        dur += m.duration
        if dur >= turn:
            return m


def moves2turns(moves):
    turns = 0
    for move in moves:
        turns += move.duration
    return turns


def generate_random_moves():
    turns = 0
    moves = []
    while (turns < lookahead):
        move = generate_random_move()
        turns += move.duration
        moves.append(move)
    return moves


def generate_random_move():
    target = generate_random_target()
    thrust = generate_random_thrust()
    duration = np.random.randint(1, max_duration)
    return Move(target, thrust, duration)


def generate_random_thrust():
    thrust = np.random.random() * 400.0 - 100.0
    thrust = int(np.clip(thrust, 0, 200))
    shild_dice = np.random.random()
    boost_dice = np.random.random()
    if shild_dice > 0.75:
        thrust = 'SHIELD'
    if boost_dice > 0.75:
        thrust = 'BOOST'
    return thrust


def generate_random_angle():
    a = np.random.random() * 80.0 - 40.0
    a = np.clip(a, -18.0, 18.0)
    return a


def generate_random_target():
    x = np.random.randint(field_w)
    y = np.random.randint(field_h)
    return np.array([x, y])


def update_solution(best_sols, child_sols):
    all_sols = best_sols + child_sols
    all_sols.sort(key=lambda x: x.score, reverse=True)
    # print(all_sols, file = sys.stderr)

    return all_sols[:len(best_sols)]


#@timeit
def run_simul(best_solution, pods, check_point_list, max_time_ms=68, with_seed=True):
    # print(best_solution, file = sys.stderr)
    start = time.time()
    # child = best_solution.create_child()
    if with_seed:
        best_solution.shift()
        best_score = best_solution.get_score(check_point_list)
    else:
        # best_solution.total_random()
        best_solution.moves1[0].thrust = 'BOOST'
        best_solution.moves2[0].thrust = 'BOOST'
        best_solution.moves1[0].target = check_point_list[pods[0].next_target_id]
        best_solution.moves2[0].target = check_point_list[pods[1].next_target_id]
        best_score = best_solution.get_score(check_point_list)
    stupid_sol = Solution(pods)
    stupid_sol.create_direct_bot_solution(check_point_list)
    stupid_score = stupid_sol.get_score(check_point_list)
    if stupid_score > best_score:
        print("Direct Bot Solution Taken!", file=sys.stderr)
        best_solution = stupid_sol

    # print("0.0",best_solution, file = sys.stderr)

    # child = Solution(pods, best_solution.depth )
    num_solutions = 0
    curtime = (time.time() - start) * 1000.0
    # print(f"Init simultime: {curtime:.1} ms", file = sys.stderr)

    while (curtime < max_time_ms):
        is_sol_random = False
        num_solutions += 1
        if num_solutions % 3 == 0:
            child = Solution(pods)
            is_sol_random = True
        else:
            child = best_solution.create_child()

        score = child.get_score(check_point_list)
        if score > best_score:
            if is_sol_random:
                print("Random Solution Taken!", file=sys.stderr)
            best_score = score
            best_solution = child
        curtime = (time.time() - start) * 1000.0
        # print(f"Loop simultime: {curtime:.1} ms", file = sys.stderr)

    # print("0.1",best_solution, file = sys.stderr)

    # print("Cur Time / Solutions / Best Score :", curtime, num_solutions, best_score, file = sys.stderr)
    print("Best Score: {:.4e}".format(best_solution.score), file=sys.stderr)
    print("Solutions:", num_solutions, file=sys.stderr)
    return best_solution


def calc_output(pod1, pod2, enem_pod1, enem_pod2, check_point_list):
    pods = pod1, pod2, enem_pod1, enem_pod2

    sr, sa, er, ea = estimate_roles(pods)

    target1, thrust1 = get_move_racer(sr, sa, er, ea, check_point_list)
    target2, thrust2 = get_move_defender(sr, sa, er, ea, check_point_list)
    pod1_moves = target1, thrust1
    pod2_moves = target2, thrust2

    return pod1_moves, pod2_moves


def run(local=False):
    # game loop
    frame = 0
    check_point_list = []
    global laps, num_checkpoints
    if not local:
        laps = int(input())
        checkpoint_count = int(input())
    else:
        laps = 3
        checkpoint_count = 5
    num_checkpoints = checkpoint_count
    # print("laps: ", laps, file = sys.stderr)
    # print("check_count: ", checkpoint_count, file = sys.stderr)
    if not local:
        for l in range(checkpoint_count):
            check_point = [int(i) for i in input().split()]
            check_point_list.append(np.array(check_point))
    else:
        check_points = np.random.randint(12000, size=(checkpoint_count, 2))
        for x in range(check_points.shape[0]):
            check_point_list.append(check_points[x, :])

    check_point_list = tuple(check_point_list)
    while True:
        start = time.time()

        time_dict.clear()

        # next_checkpoint_x: x position of the next check point
        # next_checkpoint_y: y position of the next check point
        # next_checkpoint_dist: distance to the next checkpoint
        # next_checkpoint_angle: angle between your pod orientation and the direction of the next checkpoint
        frame += 1

        if frame > 10 and local:
            break
        print("Frame: (start)", frame, file=sys.stderr)

        start = time.time()
        if not local:
            pod1 = [int(i) for i in input().split()]
            pod2 = [int(i) for i in input().split()]
            enem_pod1 = [int(i) for i in input().split()]
            enem_pod2 = [int(i) for i in input().split()]
        else:
            pod1 = [12, 12, 12, 12, 40, 1]
            pod2 = [1200, 1200, 12, 12, 40, 1]
            enem_pod1 = [2200, 1200, 12, 12, 40, 1]
            enem_pod2 = [2200, 2200, 12, 12, 40, 1]

        pod_vals = [pod1, pod2, enem_pod1, enem_pod2]

        if frame == 1:
            pods = [Pod(pod1), Pod(pod2),
                    Pod(enem_pod1), Pod(enem_pod2)]
            pods[0].set_partner(pods[1])
            pods[1].set_partner(pods[0])
            pods[2].set_partner(pods[3])
            pods[3].set_partner(pods[2])
            best_solution = Solution(pods)
            # print(best_solutio, file = sys.stderr)
            # print("0",best_solution, file = sys.stderr)
            best_solution = run_simul(best_solution, pods, check_point_list, max_time_ms=950, with_seed=False)
            # print("1",best_solution, file = sys.stderr)

        if frame > 1:
            # print("SimulDiff Pod1 Pos, Vel, Ang:",  np.array(pod1[:2]) - pods[0].pos, np.array(pod1[2:4]) - pods[0].vel, pod1[4] - pods[0].angle, file = sys.stderr)
            # print("SimulDiff Pod2 Pos, Vel, Ang:",  np.array(pod2[:2]) - pods[1].pos, np.array(pod2[2:4]) - pods[1].vel, pod2[4] - pods[1].angle,file = sys.stderr)
            # print("Frame: (mid1)", frame, file = sys.stderr)
            # curtime = (time.time() - start)*1000.0
            # print("Cur Time :", curtime, file = sys.stderr)
            # print("Angles1: cur_real, cur_bot, steer ({:.1f},{:.1f}, {:.1f})".format(pod1[4], pods[0].angle, m1.angle), file = sys.stderr)

            for pod_val, pod in zip(pod_vals, pods):
                pod.update(pod_val)
            # print("Angles2: cur_real, cur_bot, steer ({:.1f},{:.1f}, {:.1f})".format(pod1[4], pods[0].angle, m1.angle), file = sys.stderr)

            best_solution = run_simul(best_solution, pods, check_point_list)
            # print("2",best_solution, file = sys.stderr)
        # print("Frame (CalcFin): ", frame, file = sys.stderr)
        # curtime = (time.time() - start)*1000.0
        # print("Cur Time :", curtime, file = sys.stderr)

        # print("Before Sim, frame:", frame, file = sys.stderr)
        # for pod in pods:
        #    print(pod, file = sys.stderr)

        # progress = update_progress(pods, check_point_list, progress)

        # pod1_moves, pod2_moves, progress = calc_output(pod1, pod2, enem_pod1, enem_pod2,
        #                            check_point_list, progress )

        m1 = best_solution.moves1[0]
        m2 = best_solution.moves2[0]
        total_time = time.time() - start
        # eval_times(total_time)
        print(m1.mv_str())
        print(m2.mv_str())


if __name__ == "__main__":
    import socket


    is_local = socket.gethostname() == 'ostrich'
    if is_local:

        from line_profiler import LineProfiler

        lprofiler = LineProfiler()

        profile_functions = [simulate_turn, Pod.get_angle, distance, distance2]
        for pf in profile_functions:
            lprofiler.add_function(pf)
        lp_wrapper = lprofiler(run)
        lp_wrapper(local=is_local)
        lprofiler.print_stats()
        # Process profile content: generate a cachegrind file and send it to user.

        # You can also write the result to the console:
    else:
        run(local=is_local)


