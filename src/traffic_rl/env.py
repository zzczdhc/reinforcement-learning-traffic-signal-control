"""Adaptive traffic signal control environment."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Mapping, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

DIRECTIONS = ("N", "S", "E", "W")
NS_DIRECTIONS = ("N", "S")
EW_DIRECTIONS = ("E", "W")

KEEP_ACTION = 0
SWITCH_ACTION = 1

OBSERVATION_VARIANTS = {"full", "minimal"}


@dataclass(frozen=True)
class ScheduleSegment:
    """Arrival-rate segment active until a given time step."""

    until_step: int
    rates: Dict[str, float]


@dataclass
class TrafficMetrics:
    """Episode-level metrics tracked by the environment."""

    total_reward: float = 0.0
    total_departed: int = 0
    total_wait_time_steps: float = 0.0
    switch_count: int = 0  # backward-compatible alias for applied switches
    switch_requested_count: int = 0
    switch_applied_count: int = 0
    invalid_switch_count: int = 0
    queue_history: list[int] = field(default_factory=list)
    reward_history: list[float] = field(default_factory=list)
    phase_history: list[int] = field(default_factory=list)
    throughput_history: list[int] = field(default_factory=list)
    switch_requested_history: list[int] = field(default_factory=list)
    switch_applied_history: list[int] = field(default_factory=list)
    invalid_switch_history: list[int] = field(default_factory=list)

    def summary(self, step_seconds: int) -> dict[str, float]:
        """Aggregate metrics for reporting."""
        avg_queue = float(np.mean(self.queue_history)) if self.queue_history else 0.0
        max_queue = int(max(self.queue_history)) if self.queue_history else 0
        avg_wait_steps = (
            self.total_wait_time_steps / self.total_departed if self.total_departed else 0.0
        )
        throughput_per_step = (
            self.total_departed / len(self.queue_history) if self.queue_history else 0.0
        )

        return {
            "total_reward": float(self.total_reward),
            "average_queue_length": avg_queue,
            "maximum_queue_length": float(max_queue),
            "throughput_per_step": float(throughput_per_step),
            "total_departed": float(self.total_departed),
            "average_wait_time_steps": float(avg_wait_steps),
            "average_wait_time_seconds": float(avg_wait_steps * step_seconds),
            "switch_count": float(self.switch_count),
            "switch_requested_count": float(self.switch_requested_count),
            "switch_applied_count": float(self.switch_applied_count),
            "invalid_switch_count": float(self.invalid_switch_count),
        }


class AdaptiveTrafficSignalEnv(gym.Env):
    """Single-intersection Gymnasium environment for traffic signal control."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        arrival_schedule: Sequence[Mapping[str, Any]],
        episode_length: int = 200,
        step_seconds: int = 3,
        min_green_time: int = 2,
        yellow_time: int = 1,
        max_departures_per_step: int = 4,
        recent_arrival_window: int = 5,
        reward_mode: str = "queue",
        switch_penalty: float = 2.0,
        observation_variant: str = "full",
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        if reward_mode not in {"queue", "waiting"}:
            raise ValueError("reward_mode must be either 'queue' or 'waiting'")
        if episode_length <= 0:
            raise ValueError("episode_length must be > 0")
        if step_seconds <= 0:
            raise ValueError("step_seconds must be > 0")
        if min_green_time < 0:
            raise ValueError("min_green_time must be >= 0")
        if yellow_time < 0:
            raise ValueError("yellow_time must be >= 0")
        if max_departures_per_step <= 0:
            raise ValueError("max_departures_per_step must be > 0")
        if recent_arrival_window <= 0:
            raise ValueError("recent_arrival_window must be > 0")
        if observation_variant not in OBSERVATION_VARIANTS:
            raise ValueError(
                f"observation_variant must be one of {sorted(OBSERVATION_VARIANTS)}"
            )
        if render_mode not in {None, "human"}:
            raise ValueError("render_mode must be None or 'human'")

        super().__init__()
        self.arrival_schedule = self._normalize_schedule(arrival_schedule)
        self.episode_length = int(episode_length)
        self.step_seconds = int(step_seconds)
        self.min_green_time = int(min_green_time)
        self.yellow_time = int(yellow_time)
        self.max_departures_per_step = int(max_departures_per_step)
        self.recent_arrival_window = int(recent_arrival_window)
        self.reward_mode = reward_mode
        self.switch_penalty = float(switch_penalty)
        self.observation_variant = observation_variant
        self.render_mode = render_mode
        self.rng = np.random.default_rng(seed)

        self.observation_dim = 13 if self.observation_variant == "full" else 6
        self.action_dim = 2
        self.action_space = spaces.Discrete(self.action_dim)
        self.observation_space = self._build_observation_space()

        self.step_count = 0
        self.current_phase = 0
        self.phase_duration = 0
        self.yellow_remaining = 0
        self.pending_phase: int | None = None
        self.queues: Dict[str, Deque[int]] = {}
        self.recent_arrivals: Deque[np.ndarray] = deque(maxlen=self.recent_arrival_window)
        self.metrics = TrafficMetrics()

    @staticmethod
    def _normalize_schedule(
        schedule: Sequence[Mapping[str, Any]],
    ) -> list[ScheduleSegment]:
        if not schedule:
            raise ValueError("arrival_schedule must contain at least one segment")

        normalized: list[ScheduleSegment] = []
        previous_until = -1

        for segment in schedule:
            until_step = int(segment["until_step"])
            rates = {
                direction: float(segment["rates"].get(direction, 0.0))
                for direction in DIRECTIONS
            }

            if until_step <= previous_until:
                raise ValueError("arrival_schedule must be sorted by increasing until_step")
            if until_step < 0:
                raise ValueError("arrival_schedule until_step values must be >= 0")
            for direction, rate in rates.items():
                if not np.isfinite(rate) or rate < 0.0:
                    raise ValueError(f"arrival rate for {direction} must be finite and >= 0")

            previous_until = until_step
            normalized.append(ScheduleSegment(until_step=until_step, rates=rates))

        return normalized

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the episode state and return the initial observation."""
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        options = {} if options is None else dict(options)
        initial_phase = int(options.get("initial_phase", 0))
        if initial_phase not in {0, 1}:
            raise ValueError("initial_phase must be 0 or 1")

        self.step_count = 0
        self.current_phase = initial_phase
        self.phase_duration = 0
        self.yellow_remaining = 0
        self.pending_phase = None
        self.queues = {direction: deque() for direction in DIRECTIONS}
        self.recent_arrivals = deque(maxlen=self.recent_arrival_window)
        self.metrics = TrafficMetrics()

        initial_queues = options.get("initial_queues")
        if initial_queues is not None:
            if len(initial_queues) != len(DIRECTIONS):
                raise ValueError("initial_queues must have length 4")
            for direction, count in zip(DIRECTIONS, initial_queues):
                count_int = int(count)
                if count_int < 0:
                    raise ValueError("initial queue lengths must be >= 0")
                self.queues[direction].extend([0] * count_int)

        observation = self._get_observation()
        switch_allowed = self._is_switch_allowed()
        info = {
            "queue_length": self._total_queue_length(),
            "arrival_rates": self._current_arrival_rates(),
            "phase": self.current_phase,
            "switch_allowed": switch_allowed,
            "next_switch_allowed": switch_allowed,
            "yellow_remaining": self.yellow_remaining,
            "observation_variant": self.observation_variant,
        }
        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance the simulator by one step."""
        if action not in {KEEP_ACTION, SWITCH_ACTION}:
            raise ValueError("action must be 0 (keep) or 1 (switch)")
        if self.step_count >= self.episode_length:
            raise RuntimeError("episode is done; call reset() before calling step() again")

        switch_requested = action == SWITCH_ACTION
        switch_allowed = self._is_switch_allowed()
        switch_applied = False
        invalid_switch = bool(switch_requested and not switch_allowed)
        in_yellow = False

        waiting_increment = self._total_queue_length()
        self._age_queued_vehicles()

        arrivals = self._sample_arrivals()
        arrival_vector = np.asarray([arrivals[direction] for direction in DIRECTIONS], dtype=np.float32)
        self.recent_arrivals.append(arrival_vector)

        departures = {direction: 0 for direction in DIRECTIONS}

        if self.yellow_remaining > 0:
            in_yellow = True
            self._consume_yellow_step()
        elif switch_requested and switch_allowed:
            switch_applied = True
            self.metrics.switch_count += 1
            if self.yellow_time > 0:
                self.pending_phase = 1 - self.current_phase
                self.yellow_remaining = self.yellow_time
                in_yellow = True
                self._consume_yellow_step()
            else:
                self.current_phase = 1 - self.current_phase
                self.phase_duration = 0
                departures = self._serve_current_phase()
                self.phase_duration += 1
        else:
            departures = self._serve_current_phase()
            self.phase_duration += 1

        for direction, count in arrivals.items():
            self.queues[direction].extend([0] * count)

        queue_length = self._total_queue_length()
        reward_signal = -queue_length if self.reward_mode == "queue" else -waiting_increment
        reward = float(reward_signal - (self.switch_penalty if switch_applied else 0.0))

        self.metrics.total_reward += reward
        self.metrics.switch_requested_count += int(switch_requested)
        self.metrics.switch_applied_count += int(switch_applied)
        self.metrics.invalid_switch_count += int(invalid_switch)
        self.metrics.queue_history.append(queue_length)
        self.metrics.reward_history.append(reward)
        self.metrics.phase_history.append(self.current_phase)
        self.metrics.throughput_history.append(sum(departures.values()))
        self.metrics.switch_requested_history.append(int(switch_requested))
        self.metrics.switch_applied_history.append(int(switch_applied))
        self.metrics.invalid_switch_history.append(int(invalid_switch))

        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self.episode_length

        observation = self._get_observation()
        next_switch_allowed = self._is_switch_allowed()
        info = {
            "step": self.step_count,
            "queue_length": queue_length,
            "arrivals": arrivals,
            "departures": departures,
            "phase": self.current_phase,
            "phase_duration": self.phase_duration,
            "switch_requested": bool(switch_requested),
            "switch_allowed": bool(switch_allowed),
            "next_switch_allowed": bool(next_switch_allowed),
            "switch_applied": bool(switch_applied),
            "invalid_switch": bool(invalid_switch),
            "in_yellow": bool(in_yellow),
            "yellow_remaining": int(self.yellow_remaining),
            "pending_phase": self.pending_phase,
            "switch_cooldown": int(self.yellow_remaining),
            "observation_variant": self.observation_variant,
        }
        return observation, reward, terminated, truncated, info

    def render(self) -> None:
        """Print a compact text representation of the current state."""
        print(
            "[AdaptiveTrafficSignalEnv] "
            f"step={self.step_count} "
            f"phase={self.current_phase} "
            f"queues={[len(self.queues.get(d, [])) for d in DIRECTIONS]} "
            f"phase_duration={self.phase_duration} "
            f"switch_allowed={self._is_switch_allowed()} "
            f"yellow_remaining={self.yellow_remaining}"
        )

    def summarize(self) -> dict[str, float]:
        """Return a summary of the current episode metrics."""
        return self.metrics.summary(step_seconds=self.step_seconds)

    def _is_switch_allowed(self) -> bool:
        return bool(
            self.yellow_remaining == 0
            and self.phase_duration >= self.min_green_time
        )

    def _consume_yellow_step(self) -> None:
        if self.yellow_remaining <= 0:
            raise RuntimeError("_consume_yellow_step called when not in yellow")

        self.yellow_remaining -= 1
        if self.yellow_remaining == 0:
            if self.pending_phase is None:
                raise RuntimeError("yellow ended but pending_phase is missing")
            self.current_phase = self.pending_phase
            self.pending_phase = None
            self.phase_duration = 0

    def _serve_current_phase(self) -> dict[str, int]:
        departures = {direction: 0 for direction in DIRECTIONS}
        active_directions = NS_DIRECTIONS if self.current_phase == 0 else EW_DIRECTIONS
        for direction in active_directions:
            departures[direction] = min(len(self.queues[direction]), self.max_departures_per_step)
            for _ in range(departures[direction]):
                self.metrics.total_wait_time_steps += self.queues[direction].popleft()
                self.metrics.total_departed += 1
        return departures

    def _total_queue_length(self) -> int:
        return sum(len(queue) for queue in self.queues.values())

    def _current_arrival_rates(self) -> dict[str, float]:
        for segment in self.arrival_schedule:
            if self.step_count < segment.until_step:
                return dict(segment.rates)
        return dict(self.arrival_schedule[-1].rates)

    def _sample_arrivals(self) -> dict[str, int]:
        rates = self._current_arrival_rates()
        return {
            direction: int(self.rng.poisson(rates[direction]))
            for direction in DIRECTIONS
        }

    def _age_queued_vehicles(self) -> None:
        for direction in DIRECTIONS:
            self.queues[direction] = deque(age + 1 for age in self.queues[direction])

    def _recent_arrival_means(self) -> np.ndarray:
        if not self.recent_arrivals:
            return np.zeros(4, dtype=np.float32)
        stacked = np.stack(list(self.recent_arrivals), axis=0)
        return np.mean(stacked, axis=0).astype(np.float32)

    def _get_observation(self) -> np.ndarray:
        values = [
            len(self.queues.get("N", [])),
            len(self.queues.get("S", [])),
            len(self.queues.get("E", [])),
            len(self.queues.get("W", [])),
            self.current_phase,
            self.phase_duration,
        ]
        if self.observation_variant == "full":
            normalized_step = float(self.step_count) / float(self.episode_length)
            normalized_step = float(min(max(normalized_step, 0.0), 1.0))
            recent_arrivals = self._recent_arrival_means()
            values.extend(
                [
                    float(self._is_switch_allowed()),
                    self.yellow_remaining,
                    normalized_step,
                    recent_arrivals[0],
                    recent_arrivals[1],
                    recent_arrivals[2],
                    recent_arrivals[3],
                ]
            )
        return np.asarray(values, dtype=np.float32)

    def _build_observation_space(self) -> spaces.Box:
        if self.observation_variant == "minimal":
            high = np.asarray(
                [
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    1.0,
                    np.inf,
                ],
                dtype=np.float32,
            )
        else:
            high = np.asarray(
                [
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    1.0,
                    np.inf,
                    1.0,
                    float(self.yellow_time),
                    1.0,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                ],
                dtype=np.float32,
            )
        return spaces.Box(
            low=np.zeros(self.observation_dim, dtype=np.float32),
            high=high,
            dtype=np.float32,
        )
