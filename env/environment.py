"""
Pandemic RL Environment — OpenEnv Compliant
=============================================
3-city SIR pandemic simulation with quarantine and vaccination actions.

OpenEnv API:
  - reset() -> observation
  - step(action) -> (observation, reward, done, info)
  - state() -> full state dict
"""

import random


class PandemicEnv:
    """
    3-city pandemic simulation.

    Observation (12 floats): Per city [susceptible, infected, recovered, dead], normalized /1000.
    Actions (7 discrete):
        0 = do nothing
        1-3 = quarantine city 0/1/2
        4-6 = vaccinate city 0/1/2
    """

    NUM_CITIES = 3
    NUM_ACTIONS = 7
    OBS_SIZE = 12

    def __init__(self, config=None, seed=42):
        cfg = config or {}
        self.max_steps = cfg.get("max_steps", 50)
        self.infection_rate = cfg.get("infection_rate", 0.15)
        self.recovery_rate = cfg.get("recovery_rate", 0.08)
        self.death_rate = cfg.get("death_rate", 0.01)
        self.travel_rate = cfg.get("travel_rate", 0.02)
        self.initial_infected = cfg.get("initial_infected", 50)
        self.vaccination_rate = cfg.get("vaccination_rate", 0.05)
        self.quarantine_effectiveness = cfg.get("quarantine_effectiveness", 0.3)

        self.seed_value = seed
        self.rng = random.Random(seed)
        self.cities = None
        self.step_count = 0
        self.done = False
        self.quarantined = [False, False, False]
        self.vaccinated = [False, False, False]
        self.trajectory = []
        self.reset()

    def reset(self):
        """Reset environment. Returns initial observation."""
        self.rng = random.Random(self.seed_value)
        self.cities = [
            [1000 - self.initial_infected, self.initial_infected, 0, 0],
            [1000, 0, 0, 0],
            [1000, 0, 0, 0],
        ]
        self.step_count = 0
        self.done = False
        self.quarantined = [False, False, False]
        self.vaccinated = [False, False, False]
        self.trajectory = []
        obs = self._obs()
        self.trajectory.append({
            "step": 0, "observation": obs[:], "action": None,
            "reward": 0.0, "info": self._info(),
        })
        return obs

    def step(self, action):
        """Take one step. Returns (observation, reward, done, info)."""
        if self.done:
            return self._obs(), 0.0, True, self._info()

        self.step_count += 1
        action = max(0, min(int(action), self.NUM_ACTIONS - 1))

        if 1 <= action <= 3:
            self.quarantined[action - 1] = True
        elif 4 <= action <= 6:
            self.vaccinated[action - 4] = True

        new_cities = [c[:] for c in self.cities]

        for i in range(self.NUM_CITIES):
            s, inf, r, d = self.cities[i]
            pop = max(s + inf + r, 1)

            if self.vaccinated[i]:
                vacc = min(int(s * self.vaccination_rate), s)
                s -= vacc
                r += vacc

            new_inf = int(s * inf / pop * self.infection_rate)
            if self.quarantined[i]:
                new_inf = int(new_inf * self.quarantine_effectiveness)
            new_inf = min(new_inf, s)

            new_rec = min(int(inf * self.recovery_rate), inf)
            new_dead = min(int(inf * self.death_rate), inf - new_rec)

            new_cities[i][0] = s - new_inf
            new_cities[i][1] = inf + new_inf - new_rec - new_dead
            new_cities[i][2] = r + new_rec
            new_cities[i][3] = d + new_dead

        for i in range(self.NUM_CITIES):
            if self.quarantined[i]:
                continue
            for j in range(self.NUM_CITIES):
                if i == j or self.quarantined[j]:
                    continue
                travelers = int(new_cities[i][1] * self.travel_rate)
                if travelers > 0 and new_cities[j][0] > 0:
                    spread = min(travelers, new_cities[j][0])
                    new_cities[j][0] -= spread
                    new_cities[j][1] += spread

        self.cities = new_cities

        total_infected = sum(c[1] for c in self.cities)
        total_dead = sum(c[3] for c in self.cities)
        total_alive = sum(c[0] + c[2] for c in self.cities)
        total_population = max(total_alive + total_infected + total_dead, 1)
        reward = (
           (total_alive / total_population)
         - (total_infected / total_population) * 1.5
         - (total_dead / total_population) * 2.5
        )

        if self.step_count >= self.max_steps or total_infected == 0:
            self.done = True

        obs = self._obs()
        info = self._info()
        self.trajectory.append({
            "step": self.step_count, "observation": obs[:],
            "action": action, "reward": reward, "info": info,
        })
        return obs, reward, self.done, info

    def state(self):
        """Return full environment state (OpenEnv spec)."""
        info = self._info()
        return {
            "cities": [c[:] for c in self.cities],
            "step": self.step_count,
            "done": self.done,
            "quarantined": self.quarantined[:],
            "vaccinated": self.vaccinated[:],
            **info,
        }

    def get_trajectory(self):
        """Return full trajectory for grading."""
        return self.trajectory[:]

    def _obs(self):
        flat = []
        for c in self.cities:
            for v in c:
                flat.append(float(v) / 1000.0)
        return flat

    def _info(self):
        total_s = sum(c[0] for c in self.cities)
        total_i = sum(c[1] for c in self.cities)
        total_r = sum(c[2] for c in self.cities)
        total_d = sum(c[3] for c in self.cities)
        total_pop = total_s + total_i + total_r + total_d
        return {
            "susceptible": total_s,
            "infected": total_i,
            "recovered": total_r,
            "dead": total_d,
            "total_population": total_pop,
            "survival_rate": round((total_s + total_r) / max(total_pop, 1), 4),
            "step": self.step_count,
        }
