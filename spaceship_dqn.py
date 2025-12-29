"""
Space Dodge – Reinforcement Learning Demo (DQN)

Ein kleines 1D-Weltraumspiel:
- Ein Raumschiff bewegt sich horizontal durch Spuren (lanes)
- Hindernisse fallen von oben nach unten
- Ziel: möglichst lange überleben und Hindernissen ausweichen

Technik:
- Custom Environment (ähnlich OpenAI Gym)
- Deep Q-Network (DQN) mit PyTorch
- Optionales Rendering mit pygame
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Optionales Rendering
try:
    import pygame  # type: ignore

    HAS_PYGAME = True
except Exception:
    HAS_PYGAME = False


# ============================================================
# Environment Configuration
# ============================================================


@dataclass
class EnvConfig:
    """
    Konfigurationsklasse für das Space-Dodge-Environment.

    Attributes
    ----------
    lanes : int
        Anzahl der horizontalen Spuren.
    ship_y : int
        Y-Position des Raumschiffs im Grid.
    grid_h : int
        Höhe des Spielfeldes.
    spawn_prob : float
        Wahrscheinlichkeit, pro Schritt ein neues Hindernis zu erzeugen.
    max_obstacles : int
        Maximale Anzahl gleichzeitiger Hindernisse.
    speed : int
        Fallgeschwindigkeit der Hindernisse (Zellen pro Schritt).
    step_penalty : float
        Kleine Strafe pro Schritt.
    survive_reward : float
        Belohnung für jeden überlebten Schritt.
    dodge_reward : float
        Zusatzbelohnung für Hindernisse, die am Schiff vorbei sind.
    crash_penalty : float
        Strafe bei Kollision.
    """

    lanes: int = 7
    ship_y: int = 9
    grid_h: int = 10
    spawn_prob: float = 0.35
    max_obstacles: int = 6
    speed: int = 1
    step_penalty: float = -0.01
    survive_reward: float = 0.05
    dodge_reward: float = 0.2
    crash_penalty: float = -1.0


# ============================================================
# Environment
# ============================================================


class SpaceDodgeEnv:
    """
    Einfaches Reinforcement-Learning-Environment für ein Ausweichspiel.

    Aktionen:
      0 = links, 1 = bleiben, 2 = rechts

    Observation (float32):
      [ ship_x_norm,
        for k in 0..2: (obs_x_norm, obs_y_norm, dx_norm) ]

    Es werden die 3 nächsten Hindernisse (nach Nähe zur Schiff-Y) verwendet.
    """

    def __init__(self, cfg: EnvConfig = EnvConfig(), seed: int = 0):
        """
        Initialisiert das Environment.

        Parameters
        ----------
        cfg : EnvConfig
            Konfiguration.
        seed : int
            Seed für reproduzierbare Zufallszahlen.
        """
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.ship_x = cfg.lanes // 2
        self.obstacles: List[List[int]] = []  # [ [x,y], ... ]
        self.t = 0
        self.done = False
        

    def reset(self) -> np.ndarray:
        """
        Setzt das Environment zurück.

        Returns
        -------
        np.ndarray
            Anfangsbeobachtung.
        """
        self.ship_x = self.cfg.lanes // 2
        self.obstacles = []
        self.t = 0
        self.done = False
        return self._obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Führt einen Schritt aus.

        Parameters
        ----------
        action : int
            0=links, 1=bleiben, 2=rechts

        Returns
        -------
        obs : np.ndarray
            Neue Beobachtung.
        reward : float
            Reward.
        done : bool
            Episode beendet?
        info : dict
            Zusatzinfos (leer).
        """
        if self.done:
            raise RuntimeError("Episode beendet – bitte reset() aufrufen.")

        # Schiff bewegen
        if action == 0:
            self.ship_x = max(0, self.ship_x - 1)
        elif action == 2:
            self.ship_x = min(self.cfg.lanes - 1, self.ship_x + 1)

        reward = self.cfg.step_penalty

        # Hindernis spawnen
        if len(self.obstacles) < self.cfg.max_obstacles and self.rng.random() < self.cfg.spawn_prob:
            self.obstacles.append([self.rng.randrange(self.cfg.lanes), 0])

        # Hindernisse bewegen
        for ob in self.obstacles:
            ob[1] += self.cfg.speed

        # Hindernisse, die am Schiff vorbei sind -> dodge_reward
        remaining: List[List[int]] = []
        dodged = 0
        for ox, oy in self.obstacles:
            if oy > self.cfg.ship_y:
                dodged += 1
            elif oy < self.cfg.grid_h:
                remaining.append([ox, oy])
        self.obstacles = remaining
        reward += dodged * self.cfg.dodge_reward

        # Kollision prüfen
        crashed = any((ox == self.ship_x and oy == self.cfg.ship_y) for ox, oy in self.obstacles)
        if crashed:
            reward += self.cfg.crash_penalty
            self.done = True
        else:
            reward += self.cfg.survive_reward

        self.t += 1
        if self.t >= 500:
            self.done = True

        return self._obs(), float(reward), self.done, {}

    def _obs(self) -> np.ndarray:
        """
        Erzeugt den Beobachtungsvektor.

        Returns
        -------
        np.ndarray
            Zustandsvektor (float32).
        """
        ship_x_norm = self.ship_x / (self.cfg.lanes - 1)

        obs_sorted = sorted(self.obstacles, key=lambda ob: abs(self.cfg.ship_y - ob[1]))

        features = [ship_x_norm]
        for i in range(3):
            if i < len(obs_sorted):
                ox, oy = obs_sorted[i]
                features.extend(
                    [
                        ox / (self.cfg.lanes - 1),
                        oy / (self.cfg.grid_h - 1),
                        (ox - self.ship_x) / (self.cfg.lanes - 1),
                    ]
                )
            else:
                features.extend([0.0, 0.0, 0.0])

        return np.array(features, dtype=np.float32)

    @property
    def obs_size(self) -> int:
        """Dimension des Beobachtungsvektors."""
        return 1 + 3 * 3  # 10

    @property
    def n_actions(self) -> int:
        """Anzahl der Aktionen."""
        return 3


# ============================================================
# Deep Q-Network
# ============================================================


class QNet(nn.Module):
    """Kleines MLP zur Approximation der Q-Werte."""

    def __init__(self, in_dim: int, out_dim: int):
        """
        Parameters
        ----------
        in_dim : int
            Zustandsdimension.
        out_dim : int
            Anzahl Aktionen.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Vorwärtsdurchlauf: liefert Q(s,·)."""
        return self.net(x)


# ============================================================
# Replay Buffer
# ============================================================


class ReplayBuffer:
    """Experience Replay Buffer für DQN."""

    def __init__(self, capacity: int = 50_000):
        """Erstellt einen ReplayBuffer mit begrenzter Kapazität."""
        self.buf: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def push(self, s: np.ndarray, a: int, r: float, ns: np.ndarray, d: bool) -> None:
        """Speichert einen Übergang (s,a,r,ns,done)."""
        self.buf.append((s, a, r, ns, d))

    def sample(self, batch_size: int):
        """
        Zieht zufällig ein Batch.

        Returns
        -------
        states, actions, rewards, next_states, dones : np.ndarray
        """
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = map(np.array, zip(*batch))
        return s, a, r, ns, d

    def __len__(self) -> int:
        """Aktuelle Anzahl gespeicherter Übergänge."""
        return len(self.buf)


# ============================================================
# Optional pygame renderer
# ============================================================


class PygameRenderer:
    """Sehr einfacher Renderer für das Environment (pygame)."""

    def __init__(self, env: SpaceDodgeEnv, cell: int = 50):
        if not HAS_PYGAME:
            raise RuntimeError("pygame nicht installiert. `pip install pygame`")
        pygame.init()
        self.env = env
        self.cell = cell
        w = env.cfg.lanes * cell
        h = env.cfg.grid_h * cell
        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("Space Dodge (DQN)")

    def draw(self) -> None:
        """Zeichnet das aktuelle Spielfeld und verarbeitet Quit-Events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit

        self.screen.fill((10, 10, 20))
        self.clock = pygame.time.Clock()
        # Grid-Linien
        for x in range(self.env.cfg.lanes):
            pygame.draw.line(self.screen, (25, 25, 40), (x * self.cell, 0), (x * self.cell, self.env.cfg.grid_h * self.cell), 1)
        for y in range(self.env.cfg.grid_h):
            pygame.draw.line(self.screen, (25, 25, 40), (0, y * self.cell), (self.env.cfg.lanes * self.cell, y * self.cell), 1)

        # Ship
        sx = self.env.ship_x * self.cell
        sy = self.env.cfg.ship_y * self.cell
        pygame.draw.rect(self.screen, (80, 200, 255), (sx + 8, sy + 8, self.cell - 16, self.cell - 16))

        # Obstacles
        for ox, oy in self.env.obstacles:
            if 0 <= oy < self.env.cfg.grid_h:
                x = ox * self.cell
                y = oy * self.cell
                pygame.draw.rect(self.screen, (255, 140, 80), (x + 10, y + 10, self.cell - 20, self.cell - 20))

        pygame.display.flip()
        self.clock.tick(10)


# ============================================================
# Training + Watching
# ============================================================


def train_dqn(
    episodes: int = 600,
    gamma: float = 0.99,
    lr: float = 1e-3,
    batch_size: int = 128,
    start_learning: int = 2000,
    target_update: int = 500,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_steps: int = 80_000,
    render_every: int = 0,
    seed: int = 0,
):
    """
    Trainiert ein DQN auf dem SpaceDodgeEnv.

    Parameters
    ----------
    episodes : int
        Anzahl Episoden.
    render_every : int
        Wenn >0, wird jede n-te Episode gerendert (pygame nötig).

    Returns
    -------
    q : QNet
        Trainiertes Q-Netz.
    env : SpaceDodgeEnv
        Environment-Instanz.
    """
    env = SpaceDodgeEnv(seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    q = QNet(env.obs_size, env.n_actions).to(device)
    qt = QNet(env.obs_size, env.n_actions).to(device)
    qt.load_state_dict(q.state_dict())

    opt = optim.Adam(q.parameters(), lr=lr)
    rb = ReplayBuffer()

    steps = 0
    renderer = None

    def epsilon() -> float:
        frac = min(1.0, steps / eps_decay_steps)
        return eps_start + frac * (eps_end - eps_start)

    returns: List[float] = []

    for ep in range(1, episodes + 1):
        s = env.reset()
        done = False
        ep_ret = 0.0

        do_render = render_every > 0 and ep % render_every == 0
        if do_render and HAS_PYGAME:
            renderer = renderer or PygameRenderer(env)

        while not done:
            steps += 1

            # Action selection (epsilon-greedy)
            if random.random() < epsilon():
                a = random.randrange(env.n_actions)
            else:
                with torch.no_grad():
                    qs = q(torch.tensor(s, device=device).unsqueeze(0))
                    a = int(torch.argmax(qs, dim=1).item())

            ns, r, done, _ = env.step(a)
            rb.push(s, a, r, ns, done)
            s = ns
            ep_ret += r

            # Learning
            if len(rb) >= max(batch_size, start_learning):
                bs, ba, br, bns, bd = rb.sample(batch_size)

                bs_t = torch.tensor(bs, device=device)
                ba_t = torch.tensor(ba, device=device, dtype=torch.int64).unsqueeze(1)
                br_t = torch.tensor(br, device=device).unsqueeze(1)
                bns_t = torch.tensor(bns, device=device)
                bd_t = torch.tensor(bd.astype(np.float32), device=device).unsqueeze(1)

                q_sa = q(bs_t).gather(1, ba_t)

                with torch.no_grad():
                    # Double DQN
                    next_a = torch.argmax(q(bns_t), dim=1, keepdim=True)
                    next_q = qt(bns_t).gather(1, next_a)
                    target = br_t + gamma * (1.0 - bd_t) * next_q

                loss = nn.functional.smooth_l1_loss(q_sa, target)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 10.0)
                opt.step()

                if steps % target_update == 0:
                    qt.load_state_dict(q.state_dict())

            # Render
            if do_render and renderer:
                renderer.draw()

        returns.append(ep_ret)
        if ep % 20 == 0:
            avg = float(np.mean(returns[-20:]))
            print(f"Episode {ep:4d} | avg return (last 20): {avg: .3f} | eps: {epsilon():.3f}")
    pygame.time.delay(120)
    return q, env


def watch(q: QNet, episodes: int = 5, seed: int = 42) -> None:
    """
    Lässt einen trainierten Agenten spielen (pygame-Fenster).

    Parameters
    ----------
    q : QNet
        Trainiertes Modell.
    episodes : int
        Anzahl Episoden zum Zuschauen.
    seed : int
        Seed für reproduzierbares Hindernisverhalten.
    """
    if not HAS_PYGAME:
        raise RuntimeError("pygame nicht installiert. `pip install pygame`")

    env = SpaceDodgeEnv(seed=seed)
    device = next(q.parameters()).device
    renderer = PygameRenderer(env)

    for _ in range(episodes):
        s = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                a = int(torch.argmax(q(torch.tensor(s, device=device).unsqueeze(0)), dim=1).item())
            s, _, done, _ = env.step(a)
            renderer.draw()


if __name__ == "__main__":
    # Training starten:
    # - render_every=0 => kein Fenster während Training
    # - render_every=50 => alle 50 Episoden ein Fenster (langsamer)
    q, _env = train_dqn(episodes=600, render_every=100, seed=0)

    # Danach zuschauen (öffnet pygame-Fenster)
    if HAS_PYGAME:
        watch(q, episodes=5, seed=42)
    else:
        print("Zum Zuschauen: `pip install pygame`")
