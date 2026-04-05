from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich.progress import BarColumn, Progress, TextColumn
import time

class BrawlTUI:
    def __init__(self):
        self.console = Console()
        self.layout = Layout()

        # Setup Layout
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )
        self.layout["main"].split_row(
            Layout(name="logs", ratio=2),
            Layout(name="stats_col", ratio=1)
        )
        self.layout["stats_col"].split_column(
            Layout(name="game_stats"),
            Layout(name="train_stats")
        )

        self.logs = []
        self.live = Live(self.layout, refresh_per_second=10, screen=True)
        
        # AI Vision State (для отображения того что видит нейронка)
        self.ai_vision_state = {
            "health_pct": 100.0,
            "health_bar": "",
            "enemy_count": 0,
            "box_count": 0,
            "enemy1_dist": "---",
            "enemy2_dist": "---",
            "box_dist": "---",
            "bush_dist": "---",
            "poison_status": "NO",
            "poison_dir": "---",
            "walls": "----",
            "cubes_destroyed": 0,
            "idle_steps": 0,
        }

    def update_header(self, match_num, total_steps):
        grid = Table.grid(expand=True)
        grid.add_column(justify="left", ratio=1)
        grid.add_column(justify="center", ratio=1)
        grid.add_column(justify="right", ratio=1)
        grid.add_row(
            Text(f" ⌛ {time.strftime('%H:%M:%S')}", style="dim cyan"),
            Text(f"BRAWL STARS AI | MATCH #{match_num}", style="bold magenta"),
            Text(f"TOTAL STEPS: {total_steps} ", style="bold yellow")
        )
        self.layout["header"].update(Panel(grid, border_style="bright_blue"))

    def add_log(self, msg):
        timestamp = time.strftime("%H:%M:%S")
        color = "white"
        if "REWARD" in msg: color = "bold green"
        elif "PENALTY" in msg: color = "bold yellow"
        elif "FATAL" in msg: color = "bold red"
        elif "MATCH" in msg: color = "bold cyan"
        
        self.logs.append(f"[dim grey][{timestamp}][/] [{color}]{msg}[/]")
        if len(self.logs) > 25:
            self.logs.pop(0)
        
        self.layout["logs"].update(
            Panel(Text.from_markup("\n".join(self.logs)), title="📜 Activity Stream", border_style="green")
        )

    def update_game_stats(self, stats):
        table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
        for key, value in stats.items():
            val_str = str(value)
            color = "white"
            if "Reward" in key: color = "green" if float(value) > 0 else "red"
            if "YES" in val_str: val_str = f"[bold red]{val_str}[/]"
            table.add_row(Text(f"{key}:", style="bold cyan"), val_str)

        self.layout["game_stats"].update(Panel(table, title="🎮 Game Data", border_style="cyan"))

    def update_game_stats_with_vision(self, stats, ai_state=None):
        """Обновляет Game Stats + AI Vision в одной панели"""
        if ai_state:
            self.ai_vision_state.update(ai_state)
        
        vs = self.ai_vision_state
        
        # Health bar с цветом
        hp = vs["health_pct"]
        if hp > 60: hp_color = "green"
        elif hp > 30: hp_color = "yellow"
        else: hp_color = "red"
        health_bar = f"[{hp_color}]{vs['health_bar']}[/{hp_color}] [{hp_color}]{hp:.0f}%[/{hp_color}]"
        
        # Enemy distances
        e1 = vs["enemy1_dist"]
        e2 = vs["enemy2_dist"]
        enemy_info = ""
        if e1 != "---":
            enemy_info += f"  E1: {e1}"
        if e2 != "---":
            enemy_info += f" | E2: {e2}"
        if not enemy_info:
            enemy_info = "  [dim]none detected[/]"
        
        # Poison status с цветом
        poison = vs["poison_status"]
        if "YES" in poison:
            poison_str = f"[bold red]⚠ {poison}[/]"
            if vs["poison_dir"] != "---":
                poison_str += f" [dim]dir: {vs['poison_dir']}[/]"
        else:
            poison_str = f"[green]✓ NO[/]"
        
        # Walls
        walls = vs["walls"]
        wall_display = ""
        for direction, blocked in zip(["W", "S", "A", "D"], walls):
            if blocked == "1":
                wall_display += f"[red]{direction}[/]"
            else:
                wall_display += f"[green]{direction}[/]"
        
        # Создаём таблицу: AI Vision сверху, Stats снизу
        table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
        
        # --- AI Vision секция ---
        table.add_row(Text(" 🧠 AI Vision", style="bold magenta"), "")
        table.add_row(Text("  ❤️  Health:", style="red"), health_bar)
        table.add_row(Text("  👾 Enemies:", style="magenta"), 
                      Text(f"{vs['enemy_count']} |{enemy_info}", style="white"))
        table.add_row(Text("  📦 Boxes:", style="yellow"),
                      Text(f"{vs['box_count']} | nearest: {vs['box_dist']}", style="white"))
        table.add_row(Text("  🌿 Bush:", style="green"),
                      Text(f"nearest: {vs['bush_dist']}", style="white"))
        table.add_row(Text("  ☠️  Poison:", style="cyan"), poison_str)
        table.add_row(Text("  🧱 Walls:", style="white"), wall_display)
        table.add_row(Text("  💎 Cubes:", style="blue"),
                      Text(f"{vs['cubes_destroyed']} this match", style="white"))
        
        # Idle indicator
        idle = vs.get("idle_steps", 0)
        if idle > 5:
            table.add_row(Text("  ⏸️  Idle:", style="yellow"), 
                          Text(f"{idle} steps ⚠️", style="bold yellow"))
        
        # Разделитель
        table.add_row(Text("  " + "─" * 28, style="dim"), "")
        
        # --- Game Stats секция ---
        for key, value in stats.items():
            val_str = str(value)
            color = "white"
            if "Reward" in key: color = "green" if float(value) > 0 else "red"
            if "YES" in val_str: val_str = f"[bold red]{val_str}[/]"
            table.add_row(Text(f"  {key}:", style="bold cyan"), Text(val_str, style=color))
        
        self.layout["game_stats"].update(
            Panel(table, title="🧠 AI Vision + Stats", border_style="magenta")
        )

    def update_train_stats(self, stats):
        table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
        for key, value in stats.items():
            # Format numbers to be pretty
            if isinstance(value, float):
                val_display = f"{value:.6f}" if abs(value) < 0.01 else f"{value:.4f}"
            else:
                val_display = str(value)
                
            table.add_row(Text(f"{key}:", style="bold yellow"), Text(val_display, style="bright_white"))
            
        self.layout["train_stats"].update(Panel(table, title="🧠 Neural Network", border_style="yellow"))

    def update_footer(self, keys, attacking, reward):
        k_str = " ".join(keys).upper() if keys else "---"

        # Color bar based on reward
        rew_color = "green" if reward > 0 else "red"
        bar_size = min(int(abs(reward) * 50), 20)
        bar = ("█" * bar_size).ljust(20)

        footer_text = Text.assemble(
            (" MOVEMENT: ", "bold cyan"), (f"[{k_str:^7}]", "bold white"),
            ("  |  ", "dim grey"),
            (" WEAPON: ", "bold red"), ("🔥 FIRING" if attacking == "True" else "💤 READY ", "bold" if attacking == "True" else "dim"),
            ("  |  ", "dim grey"),
            (" REWARD: ", f"bold {rew_color}"), (f"{reward:+.4f} ", f"bold {rew_color}"),
            (bar, rew_color)
        )

        self.layout["footer"].update(Panel(footer_text, border_style="bright_yellow"))

    def start(self):
        self.live.start()

    def stop(self):
        self.live.stop()
