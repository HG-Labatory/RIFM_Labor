"""
Klassisches Conway's Game of Life (minimal, pygame).

Controls:
- Linksklick: Zelle umschalten (alive/dead)
- SPACE: Pause/Play
- C: Clear (alles tot)
- ESC / Fenster schließen: Ende

Du kannst selbst ergänzen:
- Random seed (R)
- Step-by-step (N)
- Wrap-around vs. harte Kanten
- Patterns (Glider etc.)
- UI (Gridgröße, Geschwindigkeit)
"""

import numpy as np
import pygame


# -----------------------
# Konfiguration (leicht anpassen)
# -----------------------
CELL_SIZE = 16  # Pixel pro Zelle
GRID_W = 50  # Zellen in x
GRID_H = 35  # Zellen in y
FPS = 10  # Simulationsgeschwindigkeit (kleiner = langsamer)


def count_neighbors(grid: np.ndarray, x: int, y: int) -> int:
    """
    Zählt lebende Nachbarn (8er-Nachbarschaft) für (x,y).

    Aktuell: harte Kanten (außerhalb = tot).
    (Wenn du Wrap-around willst: hier anpassen.)
    """
    n = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                n += int(grid[ny, nx])
    return n


def step(grid: np.ndarray) -> np.ndarray:
    """
    Berechnet das nächste Grid nach den Game-of-Life-Regeln.
    """
    new_grid = np.zeros_like(grid)
    for y in range(GRID_H):
        for x in range(GRID_W):
            alive = grid[y, x] == 1
            neighbors = count_neighbors(grid, x, y)

            # Conway-Regeln:
            # 1) alive & (2|3) -> alive
            # 2) alive & (<2 or >3) -> dead
            # 3) dead & (3) -> alive
            if alive and neighbors in (2, 3):
                new_grid[y, x] = 1
            elif (not alive) and neighbors == 3:
                new_grid[y, x] = 1

    return new_grid


def cell_from_mouse(pos) -> tuple[int, int]:
    """
    Wandelt Mausposition (Pixel) in Zellkoordinaten (x,y) um.
    """
    mx, my = pos
    return mx // CELL_SIZE, my // CELL_SIZE


def draw(screen, grid: np.ndarray) -> None:
    """
    Zeichnet Grid. (Absichtlich simpel gehalten.)
    """
    screen.fill((20, 20, 25))
    for y in range(GRID_H):
        for x in range(GRID_W):
            if grid[y, x] == 1:
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, (220, 220, 230), rect)

    # feines Gitter (optional)
    for x in range(GRID_W + 1):
        pygame.draw.line(screen, (40, 40, 50), (x * CELL_SIZE, 0), (x * CELL_SIZE, GRID_H * CELL_SIZE), 1)
    for y in range(GRID_H + 1):
        pygame.draw.line(screen, (40, 40, 50), (0, y * CELL_SIZE), (GRID_W * CELL_SIZE, y * CELL_SIZE), 1)


def main() -> None:
    """
    Startet das pygame-Fenster und die Event-Loop.
    """
    pygame.init()
    screen = pygame.display.set_mode((GRID_W * CELL_SIZE, GRID_H * CELL_SIZE))
    pygame.display.set_caption("Game of Life (minimal)")
    clock = pygame.time.Clock()

    grid = np.zeros((GRID_H, GRID_W), dtype=np.uint8)
    running = True
    paused = True  # starte pausiert, damit du zuerst klicken kannst

    while running:
        clock.tick(FPS)

        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_c:
                    grid[:] = 0

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                x, y = cell_from_mouse(event.pos)
                if 0 <= x < GRID_W and 0 <= y < GRID_H:
                    grid[y, x] = 0 if grid[y, x] == 1 else 1

        # Simulation
        if not paused:
            grid = step(grid)

        # Render
        draw(screen, grid)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
