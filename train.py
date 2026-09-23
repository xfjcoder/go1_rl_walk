"""
Train Go1 to walk with PPO (Stable-Baselines3): flat terrain first, then a
0.2-1.0 m/s speed range, then rough terrain (a random heightfield up to
+-12 cm). Each stage --resume's from the previous stage's checkpoint.

    python train.py --run-name my_run --timesteps 6000000 --n-envs 8 \\
        --target-speed 0.3 --kp 80 --kd 2 --log-std-init -1.0
    python train.py --run-name my_run_2 \\
        --resume runs/my_run/checkpoints/go1_flat_final_<steps>.zip

The full staged recipe (exact flags for each stage, and why) that produced
the current best checkpoint (runs/k_hardmine) is in README.md -- start
there rather than guessing flag combinations from scratch. Every run also
writes its own resolved args.json / env_kwargs.json to runs/<name>/, so any
past run's exact settings are always recoverable from that run's directory.

    tensorboard --logdir runs                              # monitor progress
    python eval_policy.py --run-dir runs/my_run             # fall rate / speed / drift
    python gait_stats.py  --run-dir runs/my_run             # per-foot step rate / duty / clearance
    python play.py --run-dir runs/my_run --record out.gif   # watch it
"""
import argparse
import os
import re

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.utils import set_random_seed, get_schedule_fn

from envs.go1_env import Go1FlatEnv

CKPT_DIR = "checkpoints"
LOG_DIR = "logs"


def linear_schedule(initial_value: float):
    """
    Linearly decay from initial_value down to ~0 over the course of a run.

    Justification: every training log so far shows approx_kl and clip_fraction
    climbing sharply late in training (up to approx_kl=0.90, clip_fraction=0.69
    in one run) while std stays low and bounded -- consistent with a fixed
    learning_rate becoming too large once the policy's action std has shrunk
    (for a Gaussian policy, a fixed-size parameter update produces proportionally
    larger KL divergence as std shrinks). Decaying the learning rate counteracts
    this instead of just tolerating it.
    """
    def _schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return _schedule


class StdGuardCallback(BaseCallback):
    """
    Safety net against PPO's continuous-action entropy blowup: the entropy
    bonus (ent_coef * entropy) for a Gaussian policy is unbounded above
    (entropy grows with log(std)), so if the policy-gradient signal ever
    goes quiet -- e.g. right after a reward-function change destabilizes
    the value function -- ent_coef can dominate the loss and drive std
    upward without limit, degrading the policy into near-random actions.
    This happened during flat-terrain tuning: std climbed from ~0.45 to
    ~3.0 over an 8M-step resume that combined ent_coef=0.01 with a reward
    change, and the resulting policy only moved one side of the robot.

    This callback stops training automatically if std exceeds a threshold,
    so a bad hyperparameter combination costs you a few thousand steps of
    compute instead of an entire multi-million-step run.
    """
    def __init__(self, max_std: float = 1.5, check_every: int = 4096, verbose: int = 1):
        super().__init__(verbose)
        self.max_std = max_std
        self.check_every = check_every

    def _on_step(self) -> bool:
        if self.n_calls % self.check_every == 0:
            std = float(self.model.policy.log_std.detach().exp().mean().item())
            if std > self.max_std:
                print(f"\nSTOPPING TRAINING: action std = {std:.2f} exceeded safety "
                      f"threshold {self.max_std} at step {self.num_timesteps}.\n"
                      f"This is entropy blowup, not a normal plateau. Likely cause: "
                      f"ent_coef is too high, or was combined with a reward-function "
                      f"change mid-training. Lower --ent-coef (0.0 is safest for "
                      f"continuous control) and resume from the last good checkpoint "
                      f"in checkpoints/, not from this run's final save.")
                return False
        return True


class RewardWarmupCallback(BaseCallback):
    """
    Linearly ramps a reward-shaping attribute from 0 up to its target value
    over warmup_steps, instead of imposing it at full strength from step 0.

    This matters for rigid, schedule-based shaping like phase_match_weight:
    imposing the full-strength diagonal-trot timing constraint immediately
    forces the policy to attempt precisely-timed leg lifts before it has
    learned basic balance. This is exactly what happened when
    phase_match_weight=0.4 was applied from step 0 on an untrained policy --
    ep_len_mean dropped from a consistent ~1000 to ~800, i.e. it started
    falling far more often, even though the attempted gait looked more like
    real stepping (briefly) before each fall. Ramping the weight in gives the
    policy time to learn "don't fall" first, then "match this rhythm" second.

    Uses set_attr on the vec env, which SB3's VecEnvWrapper chain (VecNormalize
    -> VecMonitor -> SubprocVecEnv) forwards down to each worker's actual
    Go1FlatEnv instance. Throttled via check_every since SubprocVecEnv's
    set_attr is an IPC round-trip to every worker process.
    """
    def __init__(self, attr_name: str, target_value: float, warmup_steps: int,
                 check_every: int = 4096, verbose: int = 1):
        super().__init__(verbose)
        self.attr_name = attr_name
        self.target_value = target_value
        self.warmup_steps = warmup_steps
        self.check_every = check_every
        self._last_printed_progress = -1

    def _on_step(self) -> bool:
        if self.n_calls % self.check_every == 0:
            progress = min(1.0, self.num_timesteps / max(self.warmup_steps, 1))
            current_value = progress * self.target_value
            self.training_env.set_attr(self.attr_name, current_value)
            self.logger.record(f"warmup/{self.attr_name}", current_value)
            progress_pct = int(progress * 100)
            if progress_pct != self._last_printed_progress and progress_pct % 10 == 0:
                print(f"\n[warmup] {self.attr_name} = {current_value:.3f} "
                      f"({progress_pct}% of ramp, step {self.num_timesteps})")
                self._last_printed_progress = progress_pct
        return True


class RampCallback(BaseCallback):
    """
    Linearly ramp an env attribute from `v_start` to `v_final` over `ramp_steps`, counted from
    the start of THIS run (so it also works on --resume), and log it as curriculum/<log_name>.
    Used for the upper end of the sampled command-speed range (speed_max_current) and of the
    sampled terrain amplitude (terrain_amp_max_current); envs sample U(range_min, current) at each reset.
    """
    def __init__(self, attr: str, log_name: str, v_start: float, v_final: float, ramp_steps: int,
                 check_every: int = 4096):
        super().__init__(0)
        self.attr, self.log_name = attr, log_name
        self.v_start, self.v_final = v_start, v_final
        self.ramp_steps, self.check_every = max(ramp_steps, 1), check_every
        self._t0 = 0

    def _on_training_start(self) -> None:
        self._t0 = self.num_timesteps

    def _on_step(self) -> bool:
        if self.n_calls % self.check_every == 0:
            progress = min(1.0, (self.num_timesteps - self._t0) / self.ramp_steps)
            v = self.v_start + progress * (self.v_final - self.v_start)
            self.training_env.set_attr(self.attr, v)
            self.logger.record(f"curriculum/{self.log_name}", v)
        return True


def vecnormalize_path_for(checkpoint_zip: str):
    """Matching VecNormalize stats file for a checkpoint zip saved by this script."""
    d, base = os.path.split(checkpoint_zip)
    m = re.match(r"go1_flat_(\d+)_steps\.zip$", base)
    if m:
        return os.path.join(d, f"go1_flat_vecnormalize_{m.group(1)}_steps.pkl")
    m = re.match(r"go1_flat_final_(\d+)\.zip$", base)
    if m:
        return os.path.join(d, f"vecnormalize_{m.group(1)}.pkl")
    if base == "go1_flat_final.zip":
        return os.path.join(d, "vecnormalize_final.pkl")
    return None


class RewardComponentLoggingCallback(BaseCallback):
    """
    Logs the mean of each individual reward term (velocity, torque, gait,
    foot_duty, phase_match, trot_symmetry, heading, etc. -- everything in
    Go1FlatEnv._compute_reward's weights_applied dict, returned each step via
    info["reward_components"]) to TensorBoard separately, instead of only the
    summed total.

    This matters because the summed total hides exactly the information
    needed to diagnose a run: e.g. "reward dipped during steps 5-9M, then
    recovered 10-20M" could mean the value function was adjusting to the
    phase_match warmup ramp, or that foot_duty penalties spiked as the policy
    tried and failed at real stepping, or several other things -- all
    indistinguishable from the aggregate number alone, but immediately obvious
    once each term has its own TensorBoard curve.
    """
    def __init__(self, log_every: int = 2048, verbose: int = 0):
        super().__init__(verbose)
        self.log_every = log_every
        self._sums = {}
        self._count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            components = info.get("reward_components")
            if components is None:
                continue
            for key, value in components.items():
                self._sums[key] = self._sums.get(key, 0.0) + value
            self._count += 1

        if self._count >= self.log_every:
            for key, total in self._sums.items():
                self.logger.record(f"reward_components/{key}", total / self._count)
            self._sums = {}
            self._count = 0
        return True


def make_env(rank: int, seed: int, env_kwargs: dict):
    """Factory for one training env. env_kwargs are Go1FlatEnv kwargs (also saved to the run dir
    as env_kwargs.json so play.py / eval_policy.py rebuild the exact same env)."""
    def _init():
        env = Go1FlatEnv(render_mode=None, domain_randomize=True, **env_kwargs)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


def main():
    parser = argparse.ArgumentParser()

    # ---- run bookkeeping ----
    g = parser.add_argument_group("run bookkeeping")
    g.add_argument("--run-name", type=str, default=None,
                    help="Write everything for this run to runs/<name>/{checkpoints,logs} plus "
                         "args.json and env_kwargs.json, so runs never overwrite each other and "
                         "play.py / eval_policy.py / gait_stats.py can rebuild the exact env with "
                         "--run-dir. Without it, the legacy shared checkpoints/ and logs/ are used.")
    g.add_argument("--resume", type=str, default=None, help="Path to a .zip checkpoint to resume from")
    g.add_argument("--resume-log-std", type=float, default=None,
                    help="On --resume, reset the policy's log action std to this value (re-opens "
                         "exploration when the task changes, e.g. -1.6 = std 0.2).")
    g.add_argument("--seed", type=int, default=0)

    # ---- PPO / exploration ----
    g = parser.add_argument_group("PPO / exploration")
    g.add_argument("--timesteps", type=int, default=20_000_000,
                    help="Total env steps for THIS run (on top of any --resume checkpoint's steps).")
    g.add_argument("--n-envs", type=int, default=16, help="Parallel MuJoCo envs")
    g.add_argument("--n-steps", type=int, default=1024,
                    help="Rollout steps PER ENV before each PPO update. "
                         "Rollout buffer size = n_steps * n_envs; batch_size must divide it evenly.")
    g.add_argument("--batch-size", type=int, default=4096,
                    help="Must evenly divide n_steps * n_envs (checked below).")
    g.add_argument("--learning-rate", type=float, default=3e-4,
                    help="Peak learning rate. Decays linearly to ~0 over the run (see "
                         "linear_schedule) to counteract the approx_kl/clip_fraction blowup "
                         "seen late in every run so far once action std shrinks.")
    g.add_argument("--ent-coef", type=float, default=0.0,
                    help="Entropy bonus coefficient. 0.0 (default) lets action noise collapse "
                         "quickly; 0.005-0.01 keeps exploration alive longer. CAUTION: unbounded "
                         "above for continuous actions -- if the policy-gradient signal goes quiet "
                         "(e.g. right after changing a reward-shaping weight mid-run), entropy can "
                         "dominate and std can blow up uncontrolled. Change ent_coef and reward "
                         "weights in SEPARATE runs, not the same resume. StdGuardCallback below "
                         "auto-stops training if this happens, but it's better to avoid it.")
    g.add_argument("--use-sde", action=argparse.BooleanOptionalAction, default=False,
                    # Measured on an UNTRAINED policy: gSDE at log_std_init=0 knocked the robot over
                    # in ~0.7s (its effective noise is exp(log_std) x the 128-dim latent features,
                    # far larger than the logged std), while plain Gaussian noise at std 0.22 stayed
                    # up ~19s. Pilots with gSDE only learned to lunge forward and fall. Turn it back
                    # on only with a much lower --log-std-init (about -3).
                    help="Generalized State-Dependent Exploration: samples noise once per "
                         "sde-sample-freq steps as a function of state, producing temporally-"
                         "correlated exploration instead of independent per-step jitter.")
    g.add_argument("--sde-sample-freq", type=int, default=4,
                    help="Resample SDE noise every N control steps. Only used if --use-sde.")
    g.add_argument("--log-std-init", type=float, default=-1.5,
                    help="Initial log of the action std (std = exp(value)). The robot stands stably "
                         "with zero action but is fragile to noise: untrained-policy survival was "
                         "1.1s at std 1.0, 4s at 0.37, 19s at 0.22 (no gSDE). -1.5 lets the policy "
                         "start near the standing pose; PPO then widens/narrows it as needed.")

    # ---- physics / control ----
    g = parser.add_argument_group("physics / control")
    g.add_argument("--kp", type=float, default=40.0,
                    help="PD position gain, N*m/rad. A scripted-walking test showed 80.0 fixes "
                         "open-loop tracking error, but transferring that straight to an untrained "
                         "RL policy caused catastrophic tumbling -- a stiffer gain amplifies noisy "
                         "actions into violent torque. 80 is fine once a gait is already learned "
                         "(every runs/* checkpoint uses --kp 80 --kd 2); keep the low default for a "
                         "from-scratch run.")
    g.add_argument("--kd", type=float, default=1.0, help="PD velocity gain, N*m*s/rad.")

    # ---- command speed ----
    g = parser.add_argument_group("command speed")
    g.add_argument("--target-speed", type=float, default=1.0,
                    help="Fixed target forward speed (m/s). Ignored if --speed-range-min is set.")
    g.add_argument("--speed-range-min", type=float, default=None,
                    help="Enable per-episode command sampling: target_speed ~ U(min, speed_max_current). "
                         "Needs --speed-range-max. Default: fixed --target-speed.")
    g.add_argument("--speed-range-max", type=float, default=1.0,
                    help="Final upper end of the sampled command range (m/s).")
    g.add_argument("--speed-curriculum-start", type=float, default=0.3,
                    help="Upper end of the command range at the start of the run; ramps linearly to "
                         "--speed-range-max over --speed-curriculum-steps.")
    g.add_argument("--speed-curriculum-steps", type=int, default=8_000_000)

    # ---- gait clock (prescribed diagonal-trot timing) ----
    g = parser.add_argument_group("gait clock")
    g.add_argument("--gait-period", type=float, default=0.7,
                    help="Seconds per full stride cycle for the prescribed gait-phase clock. "
                         "Only relevant if --phase-match-weight > 0.")
    g.add_argument("--gait-period-fast", type=float, default=None,
                    help="Trot-clock period (s) at 1.0 m/s; the period interpolates linearly from "
                         "--gait-period (at 0.3 m/s) to this, so faster commands get a faster step "
                         "rate. Default: fixed period regardless of speed.")
    g.add_argument("--gait-style", type=str, default="trot", choices=["trot", "bound"],
                    help="'trot': diagonal pairs together (FR+RL, FL+RR), the default and the only "
                         "style used in any working run so far. 'bound' (front pair / rear pair "
                         "alternating) is wired through phase_match and trot_symmetry but untested.")
    g.add_argument("--phase-match-weight", type=float, default=0.0,
                    help="Reward for matching the prescribed diagonal-trot timing above. This is "
                         "what actually fixed the front/rear step-rate mismatch (see README) -- "
                         "0.5 with an 8M-step warmup is the proven setting. Ramp it in rather than "
                         "applying it at full strength from step 0 (see --phase-match-warmup-steps): "
                         "on an untrained policy, full strength immediately made it fall more, not "
                         "less, since it demands precisely-timed lifts before the policy can balance.")
    g.add_argument("--phase-match-warmup-steps", type=int, default=8_000_000,
                    help="Linearly ramp phase-match-weight from 0 up to its target value over "
                         "this many steps. Only relevant if --phase-match-weight > 0.")
    g.add_argument("--phase-match-stair-relax", type=float, default=0.0,
                    help="Metres: linearly relax phase_match_weight to 0 as the current episode's "
                         "stair height goes from 0 to this value, freeing the policy from the rigid "
                         "2-2 diagonal trot rhythm when a stair is tall enough to need a different "
                         "support pattern. 0 = off (always full strength). Only affects stair "
                         "episodes.")
    g.add_argument("--static-stability-weight", type=float, default=0.0,
                    help="Reward weight for having MORE than 2 feet down, scaled by how tall the "
                         "current stair is (0 at stair_h=0, full weight at "
                         "--static-stability-ref-height). A direct incentive toward a static, "
                         "weight-shifting stance specifically when a stair demands it. 0 = off.")
    g.add_argument("--static-stability-ref-height", type=float, default=0.12)

    # ---- reward shaping: gait quality ----
    g = parser.add_argument_group("reward shaping: gait quality")
    g.add_argument("--air-time-weight", type=float, default=1.0,
                    help="Reward, on touchdown, for how long that foot was airborne relative to "
                         "--target-air-time. Lets the policy discover its own step frequency.")
    g.add_argument("--air-time-cap", action=argparse.BooleanOptionalAction, default=False,
                    help="Cap the touchdown air-time credit at --target-air-time so a single long "
                         "lift can't out-earn several normal steps (this is what fixed one run's "
                         "front-leg double-stepping: an uncapped credit let a single leg hover for "
                         "457ms and out-earn multiple short, correct steps elsewhere).")
    g.add_argument("--target-air-time", type=float, default=0.2,
                    help="Seconds. Touchdowns faster than this are penalized (discourages "
                         "shuffling), slower are rewarded up to this point.")
    g.add_argument("--foot-clearance-weight", type=float, default=0.08,
                    help="Weight of the swing-foot lift bonus. Without this, tiny/fast low-clearance "
                         "shuffling scores as well as a bold, visible stride.")
    g.add_argument("--target-clearance", type=float, default=0.04,
                    help="Base swing-foot lift target (m); the reward caps its benefit here, so a foot "
                         "has NO incentive to lift any higher than this even when the terrain needs it. "
                         "The env raises the EFFECTIVE per-episode target to stair_height+0.03 when "
                         "stairs are enabled, so this only sets the floor (flat/slope/bump episodes). "
                         "Confirmed: leaving this at the 4cm default while training 12cm stairs left a "
                         "policy physically stuck at the first riser (median swing height 6.6cm).")
    g.add_argument("--trot-weight", type=float, default=0.15,
                    help="Weight of the diagonal trot-symmetry reward bonus. CAUTION: a heavier "
                         "weight (tried at 0.3 alongside body-frame velocity / yaw-rate shaping) "
                         "was gamed into a hobble on a single diagonal pair, not a real trot -- "
                         "the working recipe uses --phase-match-weight for gait timing and leaves "
                         "this at 0.")
    g.add_argument("--max-foot-duty-cycle", type=float, default=0.75,
                    help="A foot averaging more ground-contact time than this fraction gets "
                         "penalized regardless of other gait terms (stops a foot from permanently "
                         "opting out by dragging).")
    g.add_argument("--min-foot-duty-cycle", type=float, default=0.2,
                    help="Symmetric to --max-foot-duty-cycle: penalizes a foot that stays "
                         "permanently lifted, never touching down.")
    g.add_argument("--foot-duty-weight", type=float, default=0.3,
                    help="Weight of the per-foot duty-cycle penalty above. The working recipe "
                         "leaves this at 0 (--phase-match-weight handles gait timing instead).")

    # ---- reward shaping: heading / drift ----
    g = parser.add_argument_group("reward shaping: heading / drift")
    g.add_argument("--body-frame-velocity", action=argparse.BooleanOptionalAction, default=False,
                    help="Track forward velocity / penalize sideways velocity in the body frame "
                         "instead of the world frame (stops rewarding off-axis drift).")
    g.add_argument("--lateral-tracking-weight", type=float, default=0.0,
                    help="Reward exp(-(v_side/sigma)^2) for zero sideways velocity. This is what "
                         "fixed a run that walked straight (good yaw) but crabbed sideways at "
                         "0.046 m/s -- pair with --body-frame-velocity.")
    g.add_argument("--lateral-tracking-sigma", type=float, default=0.1)
    g.add_argument("--yaw-rate-weight", type=float, default=0.0,
                    help="Penalty weight on yaw rate squared (discourages turning/veering).")
    g.add_argument("--heading-weight", type=float, default=0.5,
                    help="Penalizes yaw deviation from straight-ahead. Without this, a slow "
                         "constant yaw drift costs almost nothing per step and compounds into "
                         "visible curving over a long episode.")
    g.add_argument("--lateral-position-weight", type=float, default=0.3,
                    help="Penalizes y-position drift from the start line, on top of the lateral-"
                         "velocity penalty (which only discourages instantaneous sideways speed, "
                         "not accumulated drift).")

    # ---- rough terrain ----
    g = parser.add_argument_group("rough terrain")
    g.add_argument("--terrain-amp-max", type=float, default=None,
                    help="Enable rough terrain: each episode samples a heightfield amplitude "
                         "(peak-to-peak, m) ~ U(terrain-amp-min, current max). Final max amplitude, "
                         "e.g. 0.12. Unset = flat ground (a single plane, as before terrain support).")
    g.add_argument("--terrain-amp-min", type=float, default=0.0,
                    help="Lower end of the sampled terrain-amplitude range (m). Raise this (with "
                         "--terrain-curriculum-steps 1, i.e. no ramp) to hard-mine training toward "
                         "rougher terrain once the full 0..max curriculum has already run once -- "
                         "sampling from 0 even late in a curriculum under-trains the hardest corner "
                         "(large bumps + high speed together), since it's still a small slice of "
                         "what a uniform-from-0 distribution actually shows the policy.")
    g.add_argument("--terrain-curriculum-start", type=float, default=0.0,
                    help="Terrain amplitude max at the start of the run (m); ramps to --terrain-amp-max.")
    g.add_argument("--terrain-curriculum-steps", type=int, default=10_000_000)
    g.add_argument("--slope-min-deg", type=float, default=0.0,
                    help="Lower end of the sampled slope range (deg). Raise for hard-mining, same idea "
                         "as --terrain-amp-min.")
    g.add_argument("--slope-max-deg", type=float, default=None,
                    help="Enable slopes: each episode samples a slope angle (deg) ~ U(slope-min-deg, "
                         "current max) and a random uphill/downhill direction -- flat pad, then a "
                         "constant grade over --ramp-length, then a flat plateau. Unset = no slope "
                         "(independent of --terrain-amp-max; the two combine if both are set).")
    g.add_argument("--slope-curriculum-start", type=float, default=0.0,
                    help="Slope angle max at the start of the run (deg); ramps to --slope-max-deg.")
    g.add_argument("--slope-curriculum-steps", type=int, default=10_000_000)
    g.add_argument("--ramp-length", type=float, default=8.0,
                    help="Horizontal distance (m) a slope climbs/descends over before leveling into a "
                         "plateau. Longer = gentler effective grade at the same angle.")
    g.add_argument("--stair-height-min", type=float, default=0.0,
                    help="Lower end of the sampled stair riser height range (m). Raise for hard-mining, "
                         "same idea as --terrain-amp-min.")
    g.add_argument("--stair-height-max", type=float, default=None,
                    help="Enable stairs: each episode samples a riser height (m) ~ U(stair-height-min, "
                         "current max) and a random ascending/descending direction -- flat pad, then "
                         "--num-stairs steps, then a flat plateau. Unset = no stairs (independent of "
                         "--terrain-amp-max/--slope-max-deg; all three combine if set together).")
    g.add_argument("--stair-curriculum-start", type=float, default=0.0,
                    help="Stair riser height max at the start of the run (m); ramps to --stair-height-max.")
    g.add_argument("--stair-curriculum-steps", type=int, default=10_000_000)
    g.add_argument("--stair-depth", type=float, default=0.25,
                    help="Tread depth (m) of each stair. Kept a multiple of the 5cm heightfield cell "
                         "size for a crisp riser edge.")
    g.add_argument("--num-stairs", type=int, default=8, help="Number of steps before leveling into a plateau.")
    g.add_argument("--obstacle-height-min", type=float, default=0.0,
                    help="Lower end of the sampled obstacle height range (m). Raise for hard-mining, "
                         "same idea as --terrain-amp-min.")
    g.add_argument("--obstacle-height-max", type=float, default=None,
                    help="Enable discrete obstacles: each episode samples a height (m) ~ U(min, current "
                         "max) and scatters --num-obstacles isolated round bumps at random positions on "
                         "otherwise-flat ground (unlike --terrain-amp-max's continuous noise everywhere). "
                         "Unset = no obstacles; combines with terrain/slope/stairs if set together.")
    g.add_argument("--obstacle-curriculum-start", type=float, default=0.0,
                    help="Obstacle height max at the start of the run (m); ramps to --obstacle-height-max.")
    g.add_argument("--obstacle-curriculum-steps", type=int, default=10_000_000)
    g.add_argument("--obstacle-radius", type=float, default=0.10,
                    help="Base radius (m) of each obstacle's footprint (actual radius/height randomized "
                         "+-30% per obstacle).")
    g.add_argument("--num-obstacles", type=int, default=6, help="Obstacles scattered per episode.")
    g.add_argument("--use-terrain-heightmap", action=argparse.BooleanOptionalAction, default=False,
                    help="Add a 3x3 local heightmap (9 dims) to the observation -- see "
                         "Go1FlatEnv.__init__ for the exact layout. Changes obs_dim 51->60, breaking "
                         "--resume with any pre-existing (blind) checkpoint; use "
                         "expand_obs_checkpoint.py to warm-start one instead of retraining from "
                         "scratch. Default False = every prior run's exact behavior, unchanged.")
    g.add_argument("--obstacle-lane-half-width", type=float, default=0.4,
                    help="Metres either side of y=0 that obstacles are placed within. Narrower than the "
                         "full course width so obstacles actually land in the robot's walking path -- "
                         "placing them across the full +-3m course made them almost never cross it.")
    g.add_argument("--gait-period-stair-stretch", type=float, default=0.0,
                    help="Extra seconds of gait period per metre of the current episode's stair riser "
                         "height (0 = off). Gives a tall step's swing phase more real time to complete "
                         "a big lift instead of being rushed by a clock tuned for flat/bump/slope "
                         "terrain. Added because a fixed-tempo clock left a policy physically stuck at "
                         "the first riser on 12cm stairs even after fixing the foot-clearance reward's "
                         "cap -- e.g. 2.5 adds +0.3s of period (roughly +0.15s of swing time) at 12cm.")

    # ---- domain randomization ----
    g = parser.add_argument_group("domain randomization")
    g.add_argument("--friction-range", type=float, nargs=2, default=(0.6, 1.1), metavar=("LO", "HI"),
                    help="Floor/terrain (and, on terrain runs, foot) friction sampled each episode.")
    g.add_argument("--mass-scale-range", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="Trunk mass scale sampled each episode, e.g. 0.9 1.1.")
    g.add_argument("--push-velocity", type=float, default=0.0,
                    help="Random horizontal velocity kick (+-m/s) applied to the trunk every 3-6s.")

    args = parser.parse_args()

    global CKPT_DIR, LOG_DIR
    run_dir = None
    if args.run_name:
        run_dir = os.path.join("runs", args.run_name)
        if os.path.exists(run_dir) and not args.resume:
            raise SystemExit(f"{run_dir} already exists -- pick a new --run-name (or use --resume).")
        CKPT_DIR = os.path.join(run_dir, "checkpoints")
        LOG_DIR = os.path.join(run_dir, "logs")

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    buffer_size = args.n_steps * args.n_envs
    if buffer_size % args.batch_size != 0:
        # Fall back to the largest divisor of buffer_size that's <= requested batch_size,
        # so PPO never silently truncates a minibatch.
        candidates = [d for d in range(args.batch_size, 0, -1) if buffer_size % d == 0]
        fixed_batch_size = candidates[0] if candidates else buffer_size
        print(f"WARNING: batch_size={args.batch_size} does not divide rollout buffer "
              f"size ({args.n_steps} x {args.n_envs} = {buffer_size}). "
              f"Using batch_size={fixed_batch_size} instead.")
        args.batch_size = fixed_batch_size

    try:
        import tensorboard  # noqa: F401
        tb_log_dir = LOG_DIR
    except ImportError:
        print("WARNING: tensorboard is not installed in this environment "
              "(pip install tensorboard) -- continuing without TensorBoard logging.")
        tb_log_dir = None

    # If warming up, start phase_match_weight at 0 -- RewardWarmupCallback below
    # ramps it up to the target over phase_match_warmup_steps. On a --resume,
    # num_timesteps is already large, so the warmup progress calculation will
    # correctly jump straight to (or near) full strength rather than resetting.
    initial_phase_match_weight = 0.0 if args.phase_match_warmup_steps > 0 else args.phase_match_weight

    env_kwargs = dict(
        target_speed=args.target_speed, trot_symmetry_weight=args.trot_weight,
        foot_clearance_weight=args.foot_clearance_weight, target_clearance=args.target_clearance,
        heading_weight=args.heading_weight,
        lateral_position_weight=args.lateral_position_weight,
        body_frame_velocity=args.body_frame_velocity, yaw_rate_weight=args.yaw_rate_weight,
        air_time_cap=args.air_time_cap,
        command_speed_range=([args.speed_range_min, args.speed_range_max]
                             if args.speed_range_min is not None else None),
        gait_period_fast=args.gait_period_fast,
        terrain_amplitude_range=([args.terrain_amp_min, args.terrain_amp_max] if args.terrain_amp_max is not None else None),
        slope_range=([args.slope_min_deg, args.slope_max_deg] if args.slope_max_deg is not None else None),
        ramp_length=args.ramp_length,
        stair_height_range=([args.stair_height_min, args.stair_height_max] if args.stair_height_max is not None else None),
        stair_depth=args.stair_depth, num_stairs=args.num_stairs,
        gait_period_stair_stretch=args.gait_period_stair_stretch,
        obstacle_height_range=([args.obstacle_height_min, args.obstacle_height_max] if args.obstacle_height_max is not None else None),
        obstacle_radius=args.obstacle_radius, num_obstacles=args.num_obstacles,
        obstacle_lane_half_width=args.obstacle_lane_half_width,
        use_terrain_heightmap=args.use_terrain_heightmap,
        friction_range=list(args.friction_range),
        mass_scale_range=list(args.mass_scale_range) if args.mass_scale_range else None,
        push_velocity=args.push_velocity,
        lateral_tracking_weight=args.lateral_tracking_weight,
        lateral_tracking_sigma=args.lateral_tracking_sigma,
        max_foot_duty_cycle=args.max_foot_duty_cycle, min_foot_duty_cycle=args.min_foot_duty_cycle,
        foot_duty_weight=args.foot_duty_weight, gait_period=args.gait_period, gait_style=args.gait_style,
        phase_match_weight=initial_phase_match_weight, air_time_weight=args.air_time_weight,
        phase_match_stair_relax=args.phase_match_stair_relax,
        static_stability_weight=args.static_stability_weight,
        static_stability_ref_height=args.static_stability_ref_height,
        target_air_time=args.target_air_time, kp=args.kp, kd=args.kd,
    )
    if run_dir:
        import json
        with open(os.path.join(run_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        with open(os.path.join(run_dir, "env_kwargs.json"), "w") as f:
            json.dump(env_kwargs, f, indent=2)
    env = SubprocVecEnv([make_env(i, args.seed, env_kwargs) for i in range(args.n_envs)])
    if args.speed_range_min is not None:
        env.set_attr("speed_max_current", args.speed_curriculum_start)
    if args.terrain_amp_max is not None:
        env.set_attr("terrain_amp_max_current", args.terrain_curriculum_start)
    if args.slope_max_deg is not None:
        env.set_attr("slope_deg_max_current", args.slope_curriculum_start)
    if args.stair_height_max is not None:
        env.set_attr("stair_height_max_current", args.stair_curriculum_start)
    if args.obstacle_height_max is not None:
        env.set_attr("obstacle_height_max_current", args.obstacle_curriculum_start)
    env = VecMonitor(env)
    vec_path = vecnormalize_path_for(args.resume) if args.resume else None
    if vec_path and os.path.exists(vec_path):
        # Keep the observation/reward normalization the policy was trained with; a fresh VecNormalize
        # would feed a resumed policy differently scaled inputs until its running stats caught up.
        env = VecNormalize.load(vec_path, env)
        env.training, env.norm_obs, env.norm_reward = True, True, True
        print(f"Loaded VecNormalize stats from {vec_path}")
    else:
        if args.resume:
            print("WARNING: no VecNormalize stats found next to the resume checkpoint; starting fresh.")
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

    policy_kwargs = dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
                         log_std_init=args.log_std_init)

    if args.resume:
        model = PPO.load(args.resume, env=env, tensorboard_log=tb_log_dir)
        model.ent_coef = args.ent_coef  # allow overriding exploration on resume, e.g. to fix a
                                         # gait that converged asymmetrically due to ent_coef=0.0
        model.lr_schedule = get_schedule_fn(linear_schedule(args.learning_rate))
        if args.resume_log_std is not None:
            model.policy.log_std.data.fill_(args.resume_log_std)
            print(f"Reset policy log_std to {args.resume_log_std} (std {np.exp(args.resume_log_std):.2f})")
        print(f"Resumed from {args.resume} (ent_coef={args.ent_coef}, "
              f"learning_rate decaying from {args.learning_rate} over this resume's steps). "
              f"NOTE: --use-sde is ignored on resume -- SDE on/off is part of the loaded "
              f"policy's structure and can't be toggled after the fact.")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=linear_schedule(args.learning_rate),
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=args.ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_sde=args.use_sde,
            sde_sample_freq=args.sde_sample_freq if args.use_sde else -1,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tb_log_dir,
            verbose=1,
            seed=args.seed,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1_000_000 // args.n_envs, 1),
        save_path=CKPT_DIR,
        name_prefix="go1_flat",
        save_vecnormalize=True,
    )
    std_guard = StdGuardCallback(max_std=1.5)
    reward_logger = RewardComponentLoggingCallback()
    callbacks = [checkpoint_callback, std_guard, reward_logger]
    if args.speed_range_min is not None:
        callbacks.append(RampCallback("speed_max_current", "speed_max", args.speed_curriculum_start,
                                      args.speed_range_max, args.speed_curriculum_steps))
    if args.terrain_amp_max is not None:
        callbacks.append(RampCallback("terrain_amp_max_current", "terrain_amp_max", args.terrain_curriculum_start,
                                      args.terrain_amp_max, args.terrain_curriculum_steps))
    if args.slope_max_deg is not None:
        callbacks.append(RampCallback("slope_deg_max_current", "slope_max_deg", args.slope_curriculum_start,
                                      args.slope_max_deg, args.slope_curriculum_steps))
    if args.stair_height_max is not None:
        callbacks.append(RampCallback("stair_height_max_current", "stair_height_max", args.stair_curriculum_start,
                                      args.stair_height_max, args.stair_curriculum_steps))
    if args.obstacle_height_max is not None:
        callbacks.append(RampCallback("obstacle_height_max_current", "obstacle_height_max", args.obstacle_curriculum_start,
                                      args.obstacle_height_max, args.obstacle_curriculum_steps))
    if args.phase_match_warmup_steps > 0:
        callbacks.append(RewardWarmupCallback(
            attr_name="phase_match_weight",
            target_value=args.phase_match_weight,
            warmup_steps=args.phase_match_warmup_steps,
        ))

    model.learn(
        total_timesteps=args.timesteps,
        callback=CallbackList(callbacks),
        tb_log_name="go1_flat_ppo",
        reset_num_timesteps=args.resume is None,
    )

    model.save(os.path.join(CKPT_DIR, "go1_flat_final"))
    env.save(os.path.join(CKPT_DIR, "vecnormalize_final.pkl"))
    # Also save a uniquely-named copy so a later run can never silently overwrite
    # a good policy under the shared "go1_flat_final" name (this is exactly what
    # happened during trot-weight tuning: a degraded resume clobbered a healthy one).
    stamped_name = f"go1_flat_final_{model.num_timesteps}"
    model.save(os.path.join(CKPT_DIR, stamped_name))
    env.save(os.path.join(CKPT_DIR, f"vecnormalize_{model.num_timesteps}.pkl"))
    print(f"Training complete. Saved to {CKPT_DIR}/go1_flat_final.zip "
          f"(and {CKPT_DIR}/{stamped_name}.zip, which won't be overwritten by future runs)")


if __name__ == "__main__":
    main()
