import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import heapq, math, json, time, os, random


GRID_ROWS = 15
GRID_COLS = 15
TILE_SIZE = 25
WINDOW_TITLE = "Planetary Exploration Rover"

BATTERY_FULL = 200.0

FLAT, SANDY, ROCK = "flat", "sandy", "rocky"
CLIFF, TRAP, HAZARDOUS = "cliff", "trap", "hazardous"

TERRAIN_LIST = [FLAT, SANDY, ROCK, CLIFF, TRAP, HAZARDOUS]
TERRAIN_COST = {
    FLAT: 5.0, 
    SANDY: 10.0, 
    ROCK: 1e6, 
    CLIFF: 1e6,
    TRAP: 15.0, 
    HAZARDOUS: 15.0 
}

TERRAIN_UTILITY = {
    FLAT:       1,     # low utility
    SANDY:      2,     # medium utility
    TRAP:      -5,     # negative (dangerous)
    HAZARDOUS: 10,     # high scientific value
    ROCK:    -999,     # impossible to explore
    CLIFF:   -999      # impossible
}


TERRAIN_COLOR = {
    FLAT: "#dff7df", 
    SANDY: "#f6e0a9", 
    ROCK: "#6b6b6b", 
    CLIFF: "#a30000",      
    TRAP: "#8b4513",        
    HAZARDOUS: "#7b68ee"    
}

DIAGONAL_MULT = math.sqrt(2)

ANIM_STEPS = 8
ANIM_DELAY_MS = 28

# Heuristics 
def manhattan(a, b, grid=None): return abs(a[0]-b[0]) + abs(a[1]-b[1])

def euclidean(a, b, grid=None): return math.hypot(a[0]-b[0], a[1]-b[1])

def terrain_aggressive(a,b,grid):
    (r1,c1),(r2,c2)=a,b
    d=math.hypot(r1-r2,c1-c2)
    steps=max(int(math.ceil(d*2)),1)
    tot, cnt = 0.0, 0
    for t in range(steps+1):
        alpha=t/steps
        rr=int(round(r1*(1-alpha)+r2*alpha))
        cc=int(round(c1*(1-alpha)+c2*alpha))
        if 0<=rr<len(grid) and 0<=cc<len(grid[0]):
            tot += min(TERRAIN_COST.get(grid[rr][cc],TERRAIN_COST[FLAT]), 1000)
            cnt += 1
    avg = (tot/cnt) if cnt else TERRAIN_COST[FLAT]
    return d * avg


def adaptive_cost_manhattan(current, goal, grid=None):
    r_n, c_n = current
    r_g, c_g = goal
    COST_FACTOR = 5.0
    distance = abs(r_n - r_g) + abs(c_n - c_g)
    
    return COST_FACTOR * distance

def obstacle_aversion(current, goal, grid):
    r_n, c_n = current
    h_base = adaptive_cost_manhattan(current, goal)

    min_dist_to_rock_sq = float('inf')
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            # Only check truly impassable terrains (ROCK, CLIFF) for aversion penalty
            if grid[r][c] == ROCK or grid[r][c] == CLIFF:
                dist_sq = (r_n - r)**2 + (c_n - c)**2
                if dist_sq < min_dist_to_rock_sq:
                    min_dist_to_rock_sq = dist_sq

    PENALTY_FACTOR = 10.0
    
    if min_dist_to_rock_sq == float('inf'):
        penalty = 0.0
    else:
        # minimum distance of 0.5^2 = 0.25 to prevent division by zero near rocks
        min_dist_sq = max(min_dist_to_rock_sq, 0.25) 
        
        penalty = PENALTY_FACTOR * (1.0 / min_dist_sq)

    return h_base + penalty

HEURISTICS = {
    "Manhattan": manhattan,
    "Euclidean": euclidean,
    "Terrain (aggressive)": terrain_aggressive,
    "Adaptive Cost (H1)": adaptive_cost_manhattan,
    "Obstacle Aversion (H2)": obstacle_aversion
}

# === Battery helpers ===
def percent_to_abs(pct: float) -> float:
    return (float(pct) / 100.0) * BATTERY_FULL

def abs_to_percent(abs_val: float) -> float:
    if BATTERY_FULL <= 0:
        return 0.0
    return (float(abs_val) / BATTERY_FULL) * 100.0

def in_bounds(node):
    r,c = node
    return 0 <= r < GRID_ROWS and 0 <= c < GRID_COLS

# only 4 directional i.e; up, down, left, right
def neighbors_4(node):
    r, c = node
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        mult = 1.0 
        yield (nr, nc), mult

def astar(grid, start, goal, h_fn, battery=None, battery_threshold=None, prohibited_nodes=None):
    if start == goal:
        return [start], 0.0, 0

    prohibited_nodes = prohibited_nodes if prohibited_nodes is not None else set()

    open_heap = []
    heapq.heappush(open_heap, (h_fn(start, goal, grid), 0.0, start))
    came_from = {}
    gscore = {start: 0.0}
    closed = set()
    explored = 0

    while open_heap:
        f, g, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)
        explored += 1

        if current == goal:
            # Reconstruct path
            path = [current]
            while path[-1] in came_from:
                path.append(came_from[path[-1]])
            path.reverse()
            return path, gscore[current], explored


        for nb, mult in neighbors_4(current):
            if not in_bounds(nb):
                continue

            terrain_next = grid[nb[0]][nb[1]]
            if terrain_next == ROCK or terrain_next == CLIFF or nb in prohibited_nodes:
                continue

            move_cost = TERRAIN_COST[terrain_next] * mult
            # FOR UTILITY BASED ;
            # base_cost = TERRAIN_COST[terrain_next] * mult
            # utility_gain = get_utility(terrain_next)
            # move_cost = base_cost - utility_gain

            tentative = gscore[current] + move_cost

            if battery is not None and tentative > battery:
                continue

            if tentative < gscore.get(nb, float("inf")):
                gscore[nb] = tentative
                came_from[nb] = current
                heapq.heappush(open_heap, (tentative + h_fn(nb, goal, grid), tentative, nb))

    return None, float("inf"), explored

def plan_with_recharges(grid, start, final_destination, battery, h_fn, recharge_sites, prohibited_nodes=None):
    
    prohibited_nodes = prohibited_nodes if prohibited_nodes is not None else set()

    # first it will try direct path
    path, path_cost, _ = astar(grid, start, final_destination, h_fn, battery=battery, prohibited_nodes=prohibited_nodes)
    if path is not None and path_cost <= battery:
        return path, path_cost, "direct_to_goal"

    # chaining via recharge stations
    visited_recharges = set()
    current_pos = start
    remaining_battery = battery
    full_path = []
    total_cost = 0.0
    max_segments = 8  

    for _ in range(max_segments):
        # try from current_pos to final destination with remaining battery
        path_to_goal, cost_to_goal, _ = astar(grid, current_pos, final_destination, h_fn, battery=remaining_battery, prohibited_nodes=prohibited_nodes)
        if path_to_goal is not None and cost_to_goal <= remaining_battery:
            # append path_to_goal 
            if full_path:
                full_path.extend(path_to_goal[1:])
            else:
                full_path.extend(path_to_goal)
            total_cost += cost_to_goal
            return full_path, total_cost, "chained_to_goal"

        # finding reachable recharge stations from current_pos
        reachable_recharges = []
        for rc in recharge_sites:
            if rc in visited_recharges:
                continue
            p_rc, c_rc, _ = astar(grid, current_pos, rc, h_fn, battery=remaining_battery, prohibited_nodes=prohibited_nodes)
            if p_rc is not None and c_rc <= remaining_battery:
                # estimating cost from rc to goal with full battery 
                _, est_cost_rc_to_goal, _ = astar(grid, rc, final_destination, h_fn, battery=BATTERY_FULL, prohibited_nodes=prohibited_nodes)
                # if est_cost_rc_to_goal is inf, it's means it is not reachable from rc
                if est_cost_rc_to_goal == float("inf"):
                    est_cost_rc_to_goal = float("inf")
                reachable_recharges.append({
                    'pos': rc,
                    'path': p_rc,
                    'cost': c_rc,
                    'est_total_cost': total_cost + c_rc + est_cost_rc_to_goal
                })

        if not reachable_recharges:
            if remaining_battery < BATTERY_FULL * 0.20:
                return None, total_cost, "battery_too_low_no_recharge"
            return None, total_cost, "no_recharge_reachable"

        # here we are choosing best recharge (lowest estimated total cost)
        reachable_recharges.sort(key=lambda x: x['est_total_cost'])
        best_rc = reachable_recharges[0]

        if full_path:
            full_path.extend(best_rc['path'][1:])
        else:
            full_path.extend(best_rc['path'])
        total_cost += best_rc['cost']
        current_pos = best_rc['pos']
        remaining_battery = BATTERY_FULL  
        visited_recharges.add(current_pos)

        if current_pos == final_destination:
            return full_path, total_cost, "recharge_at_goal"

    return None, total_cost, "chain_limit_reached"

def get_utility(terrain):
    return TERRAIN_UTILITY.get(terrain, 0)

# GUI App(we have taken this gui code from our 1st year project; just change formatting and css here & there)
class RoverApp:
    def __init__(self, root):
        self.root = root
        root.title(WINDOW_TITLE)
        self.canvas = tk.Canvas(root, width=GRID_COLS*TILE_SIZE, height=GRID_ROWS*TILE_SIZE, bg="white")
        self.canvas.grid(row=0, column=0, sticky="nw", padx=6, pady=6)
        self.side_frame = ttk.Frame(root)
        self.side_frame.grid(row=0, column=1, sticky="ne", padx=6, pady=6)
        self.status_var = tk.StringVar(value="Place Start and Goal, choose heuristic, then Plan.")
        self.status = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w")
        self.status.grid(row=2, column=0, columnspan=2, sticky="we", padx=6, pady=(4,6))
        self.grid = [[FLAT for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
        self.start = None; self.goal = None; self.pos = None
        self.recharges = set()
        self.planned_path = []
        self.traveled = []
        self.battery = BATTERY_FULL
        self.running=False; self.anim_after=None
        self.rect_ids = [[None]*GRID_COLS for _ in range(GRID_ROWS)]
        self.recharge_ids = {}
        self.start_id = None; self.goal_id = None; self.rover_id=None
        self.path_line_ids=[]; self.traveled_line_ids=[]
        self.total_cost_used = 0.0
        self.permanently_prohibited = set() 
        self.last_hazard_hit = None 
        
        self.build_side_panel()
        self.draw_grid()
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<Motion>", self.on_motion)

    def draw_grid(self):
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                x0=c*TILE_SIZE; y0=r*TILE_SIZE; x1=x0+TILE_SIZE; y1=y0+TILE_SIZE
                rect = self.canvas.create_rectangle(x0,y0,x1,y1, fill=TERRAIN_COLOR[self.grid[r][c]], outline="#aaaaaa")
                self.rect_ids[r][c] = rect

    def redraw_tile(self,r,c):
        self.canvas.itemconfig(self.rect_ids[r][c], fill=TERRAIN_COLOR[self.grid[r][c]])

    def place_start(self,r,c):
        if self.grid[r][c] == ROCK or self.grid[r][c] == CLIFF:
            messagebox.showinfo("Invalid","Cannot place Start on impassable terrain (ROCK/CLIFF).")
            return
        
        if self.start_id: self.canvas.delete(self.start_id)
        self.start=(r,c); self.pos=(r,c)
        x0=c*TILE_SIZE+8; y0=r*TILE_SIZE+8; x1=x0+TILE_SIZE-16; y1=y0+TILE_SIZE-16
        self.start_id = self.canvas.create_rectangle(x0,y0,x1,y1, fill="#1b9e77", outline="#083")
        self.place_rover(self.pos)

    def place_goal(self,r,c):
        if self.grid[r][c] == ROCK or self.grid[r][c] == CLIFF:
            messagebox.showinfo("Invalid","Cannot place Goal on impassable terrain (ROCK/CLIFF).")
            return
        
        if self.goal_id: self.canvas.delete(self.goal_id)
        self.goal=(r,c)
        x0=c*TILE_SIZE+8; y0=r*TILE_SIZE+8; x1=x0+TILE_SIZE-16; y1=y0+TILE_SIZE-16
        self.goal_id = self.canvas.create_rectangle(x0,y0,x1,y1, fill="#d95f02", outline="#b24a00")

    def toggle_recharge(self,r,c):
        key=(r,c)
        if key in self.recharges:
            self.recharges.remove(key)
            if key in self.recharge_ids:
                self.canvas.delete(self.recharge_ids[key]); del self.recharge_ids[key]
        else:
            if self.grid[r][c]==ROCK or self.grid[r][c]==CLIFF:
                messagebox.showinfo("Invalid","Cannot place recharge on impassable tile (ROCK/CLIFF).")
                return
            
            oval = self.canvas.create_oval(c*TILE_SIZE+10, r*TILE_SIZE+10,
                                           c*TILE_SIZE+TILE_SIZE-10, r*TILE_SIZE+TILE_SIZE-10,
                                           fill="#61b3ff", outline="#1a7fb3", width=2)
            self.recharge_ids[key]=oval
            self.recharges.add(key)

    def place_rover(self,pos):
        if self.rover_id: self.canvas.delete(self.rover_id)
        r,c = pos
        x0=c*TILE_SIZE+12; y0=r*TILE_SIZE+12; x1=x0+TILE_SIZE-24; y1=y0+TILE_SIZE-24
        self.rover_id = self.canvas.create_rectangle(x0,y0,x1,y1, fill="#ff7f0e", outline="#a04f00")
        self.draw_side_battery()

    def draw_side_battery(self):
        pct = max(0.0, min(100.0, self.battery / (BATTERY_FULL/100.0))) 
        fill_h = int((pct/100.0) * 160) 
        self.bar_canvas.delete("all")
        self.bar_canvas.create_rectangle(6,6,46,166, outline="#333", width=2)
        self.bar_canvas.create_rectangle(8,8,44,164, fill="#ddd", outline="")
        color = "#2ca02c" if pct>60 else "#ffbb33" if pct>25 else "#d62728"
        self.bar_canvas.create_rectangle(8, 164-fill_h, 44, 164, fill=color, outline="")
        self.bar_canvas.itemconfig(self.batt_text, text=f"{self.battery:.0f}/{BATTERY_FULL:.0f}") 

    def draw_planned_path(self, path):
        for lid in self.path_line_ids: self.canvas.delete(lid)
        self.path_line_ids.clear()
        if not path: return
        for a,b in zip(path[:-1], path[1:]):
            x0=a[1]*TILE_SIZE+TILE_SIZE//2; y0=a[0]*TILE_SIZE+TILE_SIZE//2
            x1=b[1]*TILE_SIZE+TILE_SIZE//2; y1=b[0]*TILE_SIZE+TILE_SIZE//2
            lid = self.canvas.create_line(x0,y0,x1,y1, width=3, fill="#1f77b4", dash=(4,3))
            self.path_line_ids.append(lid)

    def draw_traveled_segment(self,a,b):
        x0=a[1]*TILE_SIZE+TILE_SIZE//2; y0=a[0]*TILE_SIZE+TILE_SIZE//2
        x1=b[1]*TILE_SIZE+TILE_SIZE//2; y1=b[0]*TILE_SIZE+TILE_SIZE//2
        lid = self.canvas.create_line(x0,y0,x1,y1, width=3, fill="#2ca02c")
        self.traveled_line_ids.append(lid)

    def clear_paths(self):
        for lid in self.path_line_ids: self.canvas.delete(lid)
        for lid in self.traveled_line_ids: self.canvas.delete(lid)
        self.path_line_ids.clear(); self.traveled_line_ids.clear()

    # Side Panel
    def build_side_panel(self):
        ttk.Label(self.side_frame, text="Mode").grid(row=0, column=0, pady=(2,6))
        self.mode_var = tk.StringVar(value="Terrain")
        modes = ["Terrain","Start","Goal","Recharge"]
        for i,m in enumerate(modes):
            ttk.Radiobutton(self.side_frame, text=m, variable=self.mode_var, value=m).grid(row=1+i, column=0, sticky="w")
        
        ttk.Label(self.side_frame, text="Heuristic").grid(row=5, column=0, pady=(8,2))
        self.heur_var = tk.StringVar(value="Manhattan")
        ttk.OptionMenu(self.side_frame, self.heur_var, self.heur_var.get(), *HEURISTICS.keys()).grid(row=6, column=0, sticky="we")
        
        ttk.Label(self.side_frame, text=f"Start Battery (max {BATTERY_FULL:.0f})").grid(row=7, column=0, pady=(8,2))
        self.batt_scale = ttk.Scale(self.side_frame, from_=10, to=BATTERY_FULL, orient="horizontal")
        self.batt_scale.set(BATTERY_FULL); self.batt_scale.grid(row=8, column=0, sticky="we")
        
        ttk.Button(self.side_frame, text="Plan (A*)", command=self.on_plan).grid(row=9, column=0, pady=(8,3), sticky="we")
        ttk.Button(self.side_frame, text="Step", command=self.on_step).grid(row=10, column=0, pady=(2,3), sticky="we")
        self.run_btn = ttk.Button(self.side_frame, text="Run", command=self.on_run); self.run_btn.grid(row=11, column=0, pady=(2,3), sticky="we")
        ttk.Button(self.side_frame, text="Stop", command=self.on_stop).grid(row=12, column=0, pady=(2,6), sticky="we")
        
        ttk.Button(self.side_frame, text="Reset Traversal", command=self.on_reset_traversal).grid(row=13, column=0, pady=(2,3), sticky="we")
        ttk.Button(self.side_frame, text="Global Reset", command=self.on_global_reset).grid(row=14, column=0, pady=(2,6), sticky="we")

        ttk.Button(self.side_frame, text="Randomize", command=self.on_random).grid(row=15, column=0, pady=(2,6), sticky="we")
        ttk.Label(self.side_frame, text="Battery").grid(row=0, column=1, padx=(8,4))
        self.bar_canvas = tk.Canvas(self.side_frame, width=52, height=172, bg="#ffffff", highlightthickness=0)
        self.bar_canvas.grid(row=1, column=1, rowspan=6, padx=(8,4))
        self.batt_text = self.bar_canvas.create_text(26, 178, text=f"{self.battery:.0f}/{BATTERY_FULL:.0f}", anchor="n")
        
        ttk.Label(self.side_frame, text="Current Tile:").grid(row=7, column=1, sticky="w", padx=(8,4))
        self.curr_tile_var = tk.StringVar(value="(-,-)")
        ttk.Label(self.side_frame, textvariable=self.curr_tile_var).grid(row=8, column=1, sticky="w", padx=(8,4))
        ttk.Label(self.side_frame, text="Terrain:").grid(row=9, column=1, sticky="w", padx=(8,4))
        self.curr_terrain_var = tk.StringVar(value="-")
        ttk.Label(self.side_frame, textvariable=self.curr_terrain_var).grid(row=10, column=1, sticky="w", padx=(8,4))
        ttk.Label(self.side_frame, text="Step Cost:").grid(row=11, column=1, sticky="w", padx=(8,4))
        self.step_cost_var = tk.StringVar(value="0.0")
        ttk.Label(self.side_frame, textvariable=self.step_cost_var).grid(row=12, column=1, sticky="w", padx=(8,4))
        ttk.Label(self.side_frame, text="Total Cost used:").grid(row=13, column=1, sticky="w", padx=(8,4))
        self.total_cost_var = tk.StringVar(value="0.0")
        ttk.Label(self.side_frame, textvariable=self.total_cost_var).grid(row=14, column=1, sticky="w", padx=(8,4))

    # Events 
    def on_motion(self, event):
        c = event.x // TILE_SIZE; r = event.y // TILE_SIZE
        if not (0 <= r < GRID_ROWS and 0 <= c < GRID_COLS):
            return
        cost_value = TERRAIN_COST[self.grid[r][c]]
        if cost_value >= 1e6:
            cost_str = "Inf"
        else:
            cost_str = f"{cost_value:.0f}"
            
        txt = f"({r},{c}) {self.grid[r][c]} cost={cost_str}"
        self.status_var.set(txt)

    def on_click(self, event):
        c = event.x // TILE_SIZE; r = event.y // TILE_SIZE
        if not in_bounds((r,c)): return
        mode = self.mode_var.get()
        if mode == "Terrain":
            cur = self.grid[r][c]
            idx = TERRAIN_LIST.index(cur)
            nxt_idx = (idx + 1) % len(TERRAIN_LIST)
            nxt = TERRAIN_LIST[nxt_idx]
            self.grid[r][c] = nxt
            self.redraw_tile(r,c)
        elif mode == "Start":
            self.place_start(r,c)
            start_pct = float(self.batt_scale.get())
            self.battery = percent_to_abs(start_pct) 
            self.total_cost_used = 0.0; self.update_side_status()
            self.clear_paths()
        elif mode == "Goal":
            self.place_goal(r,c); self.clear_paths()
        elif mode == "Recharge":
            self.toggle_recharge(r,c)
            self.clear_paths()

    def on_right_click(self, event):
        c = event.x // TILE_SIZE; r = event.y // TILE_SIZE
        if not in_bounds((r,c)): return
        self.toggle_recharge(r,c)
        self.clear_paths()

    # Actions 
    def on_plan(self, preserve_cost=False):
        if not self.start or not self.goal:
            messagebox.showerror("Missing", "Set Start and Goal first.")
            return
        
        current_battery = self.battery if preserve_cost else float(self.batt_scale.get())
        
        if not preserve_cost:
            self.total_cost_used = 0.0
            self.battery = current_battery
            if self.start: self.pos = self.start
            self.permanently_prohibited = set() 
            self.last_hazard_hit = None 
        
        if self.pos is None and self.start: self.pos = self.start

        hname = self.heur_var.get(); hfn = HEURISTICS[hname]
   
        prohibited = self.permanently_prohibited
        
        path, cost, info = plan_with_recharges(
            self.grid, 
            self.pos or self.start, 
            self.goal, 
            current_battery, 
            hfn, 
            list(self.recharges),
            prohibited_nodes=prohibited
        )
        
        if path:
            self.planned_path = path
            self.clear_paths()
            self.draw_planned_path(path)
            self.status_var.set(f"Planned path: {info}, est. cost={cost:.1f}. Steps={len(path)-1}")
        else:
            self.planned_path=[]; self.clear_paths()
            self.status_var.set(f"Planning failed: {info}")
            
        self.last_hazard_hit = None 
        self.update_side_status()
        
        return path

    def on_step(self):
        if not self.planned_path:
            self.on_plan()
            if not self.planned_path: return
        if self.pos is None:
            self.place_start(*self.start)
            
        try:
            cur_idx = self.planned_path.index(self.pos)
        except ValueError:
            cur_idx = -1
            self.status_var.set("Rover off planned path. Re-plan to continue.")
            return
            
        next_idx = cur_idx + 1
        if next_idx >= len(self.planned_path):
            self.status_var.set("At goal or end of planned path.")
            return
        
        next_cell = self.planned_path[next_idx]
        terrain = self.grid[next_cell[0]][next_cell[1]]
        
        if terrain == ROCK or terrain == CLIFF: 
            self.status_var.set(f"Path leads to impassable terrain: {terrain}. Stopping.")
            return
            
        step_cost = TERRAIN_COST[terrain]
        self.step_cost_var.set(f"{step_cost:.1f}")
        
        if step_cost > self.battery:
            self.status_var.set("Not enough battery for next tile. Cannot proceed.")
            return
            
        self.battery -= step_cost; self.total_cost_used += step_cost
        self.update_side_status()
        self.animate_move(self.pos, next_cell, on_complete=lambda: self._after_step(next_cell))
        
    def _after_step(self, newpos):
        prev = self.pos
        
        landed_terrain = self.grid[newpos[0]][newpos[1]]
        if landed_terrain == TRAP or landed_terrain == HAZARDOUS:
            self.status_var.set(f"Hazard detected at {newpos} ({landed_terrain})! Backtracking to safe cell.")
            self.pos = prev
            self.permanently_prohibited.add(newpos) 
            self.place_rover(self.pos)
            self.update_side_status()
            self.last_hazard_hit = newpos 
            return

        self.pos = newpos
        
        if prev != newpos:
            self.draw_traveled_segment(prev, newpos)
            
        if self.pos in self.recharges:
            self.battery = BATTERY_FULL
            self.status_var.set("Arrived at recharge — battery refilled. Re-planning.")
            self.update_side_status()
            self.on_plan(preserve_cost=True) 
            
        if self.pos == self.goal:
            self.status_var.set("Goal reached!")
            self.update_side_status()
            
    def _deliberate_and_replan(self):
        
        current_pos = self.pos or self.start
        
        try:
            current_index = self.planned_path.index(current_pos)
            if current_index >= len(self.planned_path) - 1 and current_pos != self.goal:
                self.status_var.set("Path segment completed/exhausted. Forcing full re-plan.")
                self.on_plan(preserve_cost=True)
                return True
        except ValueError:
            if self.last_hazard_hit is None:
                 self.status_var.set("Rover off path. Forcing full re-plan.")
                 self.on_plan(preserve_cost=True)
                 return True

        if self.battery < BATTERY_FULL * 0.20:
            self.status_var.set("Battery <20% — URGENT OVERRIDE: Re-planning whole path via recharges.")
            self.on_plan(preserve_cost=True) 
            if not self.planned_path:
                self.status_var.set("Battery <20% but no reachable path; stopping.")
                self.running = False
            return True

        elif BATTERY_FULL * 0.20 <= self.battery <= BATTERY_FULL * 0.25:
            nearby = [
                rc for rc in self.recharges
                if math.hypot((current_pos[0] - rc[0]), (current_pos[1] - rc[1])) <= 2.0
            ]
            if nearby:
                nearest_rc = min(nearby, key=lambda rc: math.hypot(current_pos[0]-rc[0], current_pos[1]-rc[1]))
                
                try:
                    target_index = self.planned_path.index(nearest_rc)
                    if target_index == self.planned_path.index(current_pos) + 1:
                        pass
                    else:
                        self.status_var.set(f"Battery {self.battery:.0f}% — nearby recharge detected, overriding full plan.")
                        self.on_plan(preserve_cost=True) 
                        return True
                except ValueError:
                    self.status_var.set(f"Battery {self.battery:.0f}% — nearby recharge detected, overriding full plan.")
                    self.on_plan(preserve_cost=True) 
                    return True
                        
        return False 

    def execute_next_step(self):
        
        if not self.running: return

        if self.pos is None:
            self.place_start(*self.start)

        replan_occurred = self._deliberate_and_replan()
        
        if replan_occurred and not self.running: 
            return
        if replan_occurred:
             self.root.after(360, self.execute_next_step) 
             return
            
        if not self.planned_path:
            self.status_var.set("No path available after deliberation. Stopping.")
            self.running = False
            return
            
        try:
            cur_idx = self.planned_path.index(self.pos)
        except ValueError:
            self.status_var.set("Rover off path (Execute phase error). Re-planning.")
            self.on_plan(preserve_cost=True)
            self.root.after(360, self.execute_next_step) 
            return

        next_idx = cur_idx + 1
        if next_idx >= len(self.planned_path):
            self.status_var.set("Path fully traversed or Goal reached. Stopping.")
            self.running = False
            return

        next_cell = self.planned_path[next_idx]
        terrain = self.grid[next_cell[0]][next_cell[1]]
        
        if terrain == ROCK or terrain == CLIFF:
            self.status_var.set(f"Critical error: Planned path includes impassable terrain: {terrain}. Stopping.")
            self.running = False
            return
            
        step_cost = TERRAIN_COST[terrain]
        self.step_cost_var.set(f"{step_cost:.1f}")

        if step_cost > self.battery:
            self.status_var.set("Battery insufficient for next step. Stopping (deliberation failed).")
            self.running = False
            return

        self.battery -= step_cost
        self.total_cost_used += step_cost
        self.update_side_status()
        
        self.animate_move(
            self.pos,
            next_cell,
            on_complete=lambda: self._after_execution_step(next_cell, self.pos) # Pass current position as 'prev'
        )
        
    def _after_execution_step(self, newpos, prev_pos):
        
        landed_terrain = self.grid[newpos[0]][newpos[1]]
        if landed_terrain == TRAP or landed_terrain == HAZARDOUS:
            self.status_var.set(f"Hazard detected at {newpos} ({landed_terrain})! Backtracking, re-planning, and resuming.")
            
            # setting the prohibited node in the permanent set
            self.permanently_prohibited.add(newpos) 
            self.last_hazard_hit = newpos 
            
            self.pos = prev_pos 
            self.place_rover(self.pos) 
            self.update_side_status()
            
            self.draw_traveled_segment(prev_pos, newpos)
            
            self.on_plan(preserve_cost=True) 
            
            self.root.after(360, self.execute_next_step)
            return
        
        self.pos = newpos
        if prev_pos != newpos:
            self.draw_traveled_segment(prev_pos, newpos)
            
        self.last_hazard_hit = None 
            
        is_recharged = False
        if self.pos in self.recharges:
            self.battery = BATTERY_FULL
            is_recharged = True
            self.status_var.set("Arrived at recharge (auto) - battery refilled. Re-planning to goal.")
            self.update_side_status()
            self.on_plan(preserve_cost=True)
            
        if self.pos == self.goal:
            self.status_var.set("Goal reached!")
            self.running=False; self.update_side_status(); return
        
        if self.running:
            delay = 360 if is_recharged else 10 
            self.root.after(delay, self.execute_next_step)
        
    def on_run(self):
        if not self.start or not self.goal:
            messagebox.showerror("Missing", "Set Start and Goal first.")
            return
        if self.running: return
        
        if not self.planned_path or self.planned_path[-1] != self.goal or self.planned_path[0] != self.pos:
            self.on_plan() 
            if not self.planned_path: return
            
        self.running = True
        self.execute_next_step() 

    def on_stop(self):
        self.running=False
        if self.anim_after:
            self.root.after_cancel(self.anim_after); self.anim_after=None
        self.status_var.set("Run stopped.")

    def on_reset_traversal(self):
        self.on_stop()

        for line_id in self.path_line_ids + self.traveled_line_ids:
            self.canvas.delete(line_id)
        self.path_line_ids.clear()
        self.traveled_line_ids.clear()

        if self.rover_id:
            self.canvas.delete(self.rover_id)
            self.rover_id = None

        self.planned_path = []
        self.traveled = []
        self.total_cost_used = 0.0
        self.battery = BATTERY_FULL
        self.pos = self.start 
        self.last_hazard_hit = None
        self.permanently_prohibited = set() 
        
        if self.start: self.place_rover(self.start) 
        
        self.update_side_status()
        self.status_var.set("Traversal reset. Choose a new heuristic and rerun.")

    def on_global_reset(self):
        self.on_stop()
        self.canvas.delete("all")

        self.grid = [[FLAT for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
        self.start = self.goal = self.pos = None
        self.recharges.clear()

        self.planned_path = []
        self.traveled = []
        self.battery = BATTERY_FULL
        self.total_cost_used = 0.0

        self.rect_ids = [[None] * GRID_COLS for _ in range(GRID_ROWS)]
        self.recharge_ids.clear()
        self.start_id = None
        self.goal_id = None
        self.rover_id = None
        self.path_line_ids = []
        self.traveled_line_ids = []
        self.last_hazard_hit = None 
        self.permanently_prohibited = set() 

        self.draw_grid()
        self.update_side_status()
        self.clear_paths()
        self.status_var.set("Global Reset complete. Paint terrain and set Start/Goal.")

    def on_random(self):
        self.on_stop()
        weights =[0.4, 0.25, 0.1, 0.1, 0.075, 0.075]
        terrains = [FLAT, SANDY, ROCK, CLIFF, TRAP, HAZARDOUS]
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                self.grid[r][c] = random.choices(terrains, weights=weights)[0]
                self.redraw_tile(r,c)
        for k in list(self.recharge_ids.keys()):
            self.canvas.delete(self.recharge_ids[k]); del self.recharge_ids[k]
        self.recharges.clear()
        if self.start_id: self.canvas.delete(self.start_id); self.start_id=None; self.start=None
        if self.goal_id: self.canvas.delete(self.goal_id); self.goal_id=None; self.goal=None
        if self.rover_id: self.canvas.delete(self.rover_id); self.rover_id=None; self.pos=None
        self.last_hazard_hit = None 
        self.permanently_prohibited = set() 
        self.clear_paths()
        self.status_var.set("Randomized map; set Start and Goal.")

    # Utilities 
    def animate_move(self, frm, to, on_complete=None):
        if self.rover_id is None:
            self.place_rover(frm)
        frx = frm[1]*TILE_SIZE + TILE_SIZE/2; fry = frm[0]*TILE_SIZE + TILE_SIZE/2
        tox = to[1]*TILE_SIZE + TILE_SIZE/2; toy = to[0]*TILE_SIZE + TILE_SIZE/2
        dx = (tox-frx)/ANIM_STEPS; dy=(toy-fry)/ANIM_STEPS
        step = 0
        def step_anim():
            nonlocal step
            if step >= ANIM_STEPS:
                self.canvas.coords(self.rover_id, to[1]*TILE_SIZE+12, to[0]*TILE_SIZE+12,
                                   to[1]*TILE_SIZE+TILE_SIZE-12, to[0]*TILE_SIZE+TILE_SIZE-12)
                self.canvas.delete("battery_overlay") 
                self.draw_side_battery()
                if on_complete: on_complete()
                return
            self.canvas.move(self.rover_id, dx, dy)
            self.canvas.delete("battery_overlay")
            cx = frx + dx*(step+1); cy = fry + dy*(step+1)
            cc = int(cx // TILE_SIZE); rr = int(cy // TILE_SIZE)
            if in_bounds((rr,cc)):
                h = TILE_SIZE - 8
                fill_h = int((self.battery / BATTERY_FULL) * h)
                color = "#2ca02c" if self.battery/BATTERY_FULL > 0.6 else "#ffbb33" if self.battery/BATTERY_FULL > 0.25 else "#d62728"
                self.canvas.create_rectangle(cc*TILE_SIZE+4, rr*TILE_SIZE + (h - fill_h) +4,
                                             cc*TILE_SIZE+10, rr*TILE_SIZE+h+4, fill=color, tags="battery_overlay")
            step += 1
            self.anim_after = self.root.after(ANIM_DELAY_MS, step_anim)
        step_anim()

    def update_side_status(self):
        self.draw_side_battery()
        self.curr_tile_var.set(str(self.pos) if self.pos else "(-,-)")
        if self.pos:
            terr = self.grid[self.pos[0]][self.pos[1]]
            cost_value = TERRAIN_COST.get(terr, 0.0)
            if cost_value >= 1e6:
                cost_str = "Inf"
            else:
                cost_str = f"{cost_value:.0f}"

            self.curr_terrain_var.set(f"{terr} (cost {cost_str})")
        else:
            self.curr_terrain_var.set("-")
            
        self.step_cost_var.set(f"{0.0:.1f}")
        self.total_cost_var.set(f"{self.total_cost_used:.2f}")

def main():
    root = tk.Tk()
    app = RoverApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
