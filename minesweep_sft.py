"""
Minesweeper SFT Training Pipeline
===================================
Supervised Fine-Tuning with expert-generated dataset.
Each section is labeled as a "CELL" so you can copy into a Jupyter notebook.

Usage:
  - Copy each CELL section into a notebook cell, OR
  - Run directly: python minesweep_sft.py
"""

# ============================================================
# CELL 1: Imports & Game Implementation
# ============================================================

import json
import re
import random
import copy
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set, Dict
from collections import Counter


@dataclass
class MinesweeperGame:
    rows: int
    cols: int
    num_mines: int
    seed: Optional[int] = None
    _rng: random.Random = field(init=False, repr=False)
    _board: List[List[int]] = field(init=False, repr=False)
    _revealed: Set[Tuple[int, int]] = field(init=False, repr=False, default_factory=set)
    _flagged: Set[Tuple[int, int]] = field(init=False, repr=False, default_factory=set)
    _state: str = field(default="ongoing", init=False, repr=False)

    def __post_init__(self):
        if self.num_mines >= self.rows * self.cols:
            raise ValueError("Too many mines for board size")
        self._rng = random.Random(self.seed)
        self._board = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self._place_mines()
        self._calculate_numbers()

    def _place_mines(self):
        positions = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        mine_positions = self._rng.sample(positions, self.num_mines)
        for r, c in mine_positions:
            self._board[r][c] = -1

    def _calculate_numbers(self):
        for r in range(self.rows):
            for c in range(self.cols):
                if self._board[r][c] == -1:
                    continue
                count = 0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.rows and 0 <= nc < self.cols:
                            if self._board[nr][nc] == -1:
                                count += 1
                self._board[r][c] = count

    def _reveal_cell(self, row: int, col: int) -> bool:
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return False
        if (row, col) in self._revealed or (row, col) in self._flagged:
            return False
        stack = [(row, col)]
        while stack:
            r, c = stack.pop()
            if (r, c) in self._revealed:
                continue
            self._revealed.add((r, c))
            if self._board[r][c] == -1:
                self._state = "failed"
                return True
            if self._board[r][c] == 0:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < self.rows and 0 <= nc < self.cols
                                and (nr, nc) not in self._revealed
                                and (nr, nc) not in self._flagged):
                            stack.append((nr, nc))
        return True

    def _flag_cell(self, row: int, col: int) -> bool:
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return False
        if (row, col) in self._revealed:
            return False
        if (row, col) in self._flagged:
            self._flagged.remove((row, col))
        else:
            self._flagged.add((row, col))
        return True

    def do_action(self, action: dict) -> str:
        if self._state != "ongoing":
            return "game_over"
        if not isinstance(action, dict):
            self._state = "failed"
            return "invalid_format"
        action_type = action.get("type")
        row = action.get("row")
        col = action.get("col")
        if action_type not in ["reveal", "flag"] or row is None or col is None:
            self._state = "failed"
            return "invalid_format"
        try:
            row, col = int(row), int(col)
        except (ValueError, TypeError):
            self._state = "failed"
            return "invalid_format"
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            self._state = "failed"
            return "out_of_bounds"
        if action_type == "reveal":
            if (row, col) in self._revealed:
                self._state = "failed"
                return "already_revealed"
            if (row, col) in self._flagged:
                self._state = "failed"
                return "flagged_cell"
            valid = self._reveal_cell(row, col)
        else:
            if (row, col) in self._revealed:
                self._state = "failed"
                return "invalid_flag"
            valid = self._flag_cell(row, col)
        if not valid:
            self._state = "failed"
            return "invalid_format"
        self._check_win()
        if self._state == "failed":
            return "mine"
        if self._state == "success":
            return "win"
        return "ok"

    def _check_win(self):
        total_cells = self.rows * self.cols
        safe_cells = total_cells - self.num_mines
        if len(self._revealed) == safe_cells:
            self._state = "success"

    def get_visible_board(self) -> List[List[str]]:
        visible = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                if (r, c) in self._flagged:
                    row.append('F')
                elif (r, c) in self._revealed:
                    val = self._board[r][c]
                    row.append('*' if val == -1 else str(val))
                else:
                    row.append('.')
            visible.append(row)
        return visible

    def state(self) -> str:
        return self._state

    def pretty_print(self) -> str:
        visible = self.get_visible_board()
        lines = []
        header = "   " + " ".join(f"{i:2d}" for i in range(self.cols))
        lines.append(header)
        lines.append("  " + "─" * (self.cols * 3 + 1))
        for r, row in enumerate(visible):
            line = f"{r:2d}│ " + "  ".join(row)
            lines.append(line)
        return "\n".join(lines)


# ============================================================
# CELL 2: Prompt Format (matches existing system)
# ============================================================

SYSTEM_PROMPT = "You output JSON actions for Minesweeper. No text, only JSON."


def format_state_for_llm(game: MinesweeperGame) -> str:
    """Same prompt format as existing notebook + agent."""
    state = {
        "board": game.get_visible_board(),
        "rows": game.rows,
        "cols": game.cols,
        "mines": game.num_mines,
        "flags_placed": len(game._flagged),
        "cells_revealed": len(game._revealed),
    }
    prompt = (
        "You are playing Minesweeper. Analyze the game state and output your next move.\n\n"
        "You must output ONLY a valid JSON object. No explanation, no analysis, no text.\n\n"
        "Just output section after assistantfinal and not anything before it in your output.\n\n"
        "Start your response immediately with { and end with }.\n\n"
        "Do NOT output cell which is already revealed or flagged in the current state.\n\n"
        "Game state:\n"
        f"{json.dumps(state, indent=2)}\n\n"
        "Legend:\n"
        '- "." = unrevealed cell\n'
        '- "F" = flagged cell (suspected mine)\n'
        '- "0"-"8" = number of adjacent mines\n'
        '- "*" = revealed mine (game over)\n\n'
        "Output your next action as JSON:\n"
        '{"type": "reveal", "row": <row_index>, "col": <col_index>}\n'
        "or\n"
        '{"type": "flag", "row": <row_index>, "col": <col_index>}\n\n'
        "Your action:"
    )
    return prompt


def parse_llm_action(response: str) -> Optional[dict]:
    """Extract JSON action from LLM response."""
    best = None
    for match in re.finditer(r'\{[^{}]*\}', response):
        try:
            action = json.loads(match.group())
            if ("type" in action and "row" in action and "col" in action
                    and action["type"] in ["reveal", "flag"]):
                best = action
        except json.JSONDecodeError:
            continue
    return best


# ============================================================
# CELL 3: Minesweeper Solver (Expert Move Generator)
# ============================================================

def get_neighbors(row, col, rows, cols):
    """Get all valid neighbor coordinates."""
    neighbors = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = row + dr, col + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                neighbors.append((nr, nc))
    return neighbors


def solve_step(game: MinesweeperGame) -> Optional[dict]:
    """
    Find the best expert move for the current game state.

    Strategy (priority order):
    1. Constraint propagation — find logically deducible safe reveals
    2. Constraint propagation — find logically deducible mine flags
    3. Probability estimate — pick the safest unrevealed cell
    4. Opening move — pick a corner (statistically safest for first move)

    Returns: {"type": "reveal"|"flag", "row": int, "col": int} or None
    """
    rows, cols = game.rows, game.cols

    # Collect board info
    safe_cells = set()   # Cells deduced to be safe
    mine_cells = set()   # Cells deduced to be mines

    # --- Pass 1: Constraint propagation ---
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in game._revealed:
                continue
            cell_val = game._board[r][c]
            if cell_val <= 0:
                continue

            neighbors = get_neighbors(r, c, rows, cols)
            hidden = []
            flagged_count = 0
            for nr, nc in neighbors:
                if (nr, nc) in game._flagged:
                    flagged_count += 1
                elif (nr, nc) not in game._revealed:
                    hidden.append((nr, nc))

            remaining_mines = cell_val - flagged_count

            if remaining_mines == 0 and hidden:
                # All mines accounted for — hidden neighbors are SAFE
                for h in hidden:
                    safe_cells.add(h)
            elif remaining_mines == len(hidden) and hidden:
                # All hidden neighbors must be mines
                for h in hidden:
                    mine_cells.add(h)

    # --- Pass 2: Extended constraint propagation (pairs) ---
    # Check if subsets of constraints can reveal more info
    # This catches cases simple single-cell analysis misses
    revealed_numbered = []
    for r in range(rows):
        for c in range(cols):
            if (r, c) in game._revealed and game._board[r][c] > 0:
                revealed_numbered.append((r, c))

    for i, (r1, c1) in enumerate(revealed_numbered):
        val1 = game._board[r1][c1]
        neighbors1 = get_neighbors(r1, c1, rows, cols)
        hidden1 = set()
        flagged1 = 0
        for nr, nc in neighbors1:
            if (nr, nc) in game._flagged:
                flagged1 += 1
            elif (nr, nc) not in game._revealed:
                hidden1.add((nr, nc))
        rem1 = val1 - flagged1
        if not hidden1:
            continue

        for j, (r2, c2) in enumerate(revealed_numbered):
            if i >= j:
                continue
            # Only check nearby cells (neighbors or neighbors-of-neighbors)
            if abs(r1 - r2) > 2 or abs(c1 - c2) > 2:
                continue

            val2 = game._board[r2][c2]
            neighbors2 = get_neighbors(r2, c2, rows, cols)
            hidden2 = set()
            flagged2 = 0
            for nr, nc in neighbors2:
                if (nr, nc) in game._flagged:
                    flagged2 += 1
                elif (nr, nc) not in game._revealed:
                    hidden2.add((nr, nc))
            rem2 = val2 - flagged2
            if not hidden2:
                continue

            # If hidden1 ⊂ hidden2
            if hidden1 < hidden2:
                diff = hidden2 - hidden1
                diff_mines = rem2 - rem1
                if diff_mines == 0:
                    for cell in diff:
                        safe_cells.add(cell)
                elif diff_mines == len(diff):
                    for cell in diff:
                        mine_cells.add(cell)

            # If hidden2 ⊂ hidden1
            elif hidden2 < hidden1:
                diff = hidden1 - hidden2
                diff_mines = rem1 - rem2
                if diff_mines == 0:
                    for cell in diff:
                        safe_cells.add(cell)
                elif diff_mines == len(diff):
                    for cell in diff:
                        mine_cells.add(cell)

    # --- Priority 1: Reveal a safe cell (prefer logically deduced) ---
    if safe_cells:
        # Prefer cells adjacent to more revealed cells (more informative)
        def info_score(cell):
            r, c = cell
            score = 0
            for nr, nc in get_neighbors(r, c, rows, cols):
                if (nr, nc) in game._revealed and game._board[nr][nc] > 0:
                    score += 1
            return score

        best = max(safe_cells, key=info_score)
        return {"type": "reveal", "row": best[0], "col": best[1]}

    # --- Priority 2: Flag a deduced mine ---
    if mine_cells:
        # Flag cell that will unlock the most safe reveals
        cell = next(iter(mine_cells))
        return {"type": "flag", "row": cell[0], "col": cell[1]}

    # --- Priority 3: No deduction possible — use probability heuristic ---
    unrevealed = []
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in game._revealed and (r, c) not in game._flagged:
                unrevealed.append((r, c))

    if not unrevealed:
        return None

    # If nothing revealed yet (opening move), pick a corner
    if len(game._revealed) == 0:
        corners = [(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)]
        corner = random.choice(corners)
        return {"type": "reveal", "row": corner[0], "col": corner[1]}

    # Estimate mine probability for each unrevealed cell
    # Use the constraint from each adjacent numbered cell
    mine_prob = {}
    for r, c in unrevealed:
        mine_prob[(r, c)] = 0.0

    # For each numbered cell, distribute remaining mine probability
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in game._revealed:
                continue
            cell_val = game._board[r][c]
            if cell_val <= 0:
                continue
            neighbors = get_neighbors(r, c, rows, cols)
            hidden = []
            flagged_count = 0
            for nr, nc in neighbors:
                if (nr, nc) in game._flagged:
                    flagged_count += 1
                elif (nr, nc) not in game._revealed:
                    hidden.append((nr, nc))
            remaining = cell_val - flagged_count
            if hidden and remaining > 0:
                prob = remaining / len(hidden)
                for nr, nc in hidden:
                    if (nr, nc) in mine_prob:
                        mine_prob[(nr, nc)] = max(mine_prob[(nr, nc)], prob)

    # Cells with no adjacent revealed numbered cells get base probability
    total_unrevealed_mines = game.num_mines - len(game._flagged)
    # Count mines near revealed area
    near_boundary = sum(1 for cell in unrevealed if mine_prob[cell] > 0)
    far_cells = [cell for cell in unrevealed if mine_prob[cell] == 0.0]

    if far_cells:
        remaining_far_mines = max(0, total_unrevealed_mines - sum(
            1 for cell in unrevealed if mine_prob[cell] >= 0.5))
        if len(far_cells) > 0:
            base_prob = remaining_far_mines / len(far_cells) if far_cells else 1.0
            base_prob = min(base_prob, 0.99)
            for cell in far_cells:
                mine_prob[cell] = base_prob

    # Pick cell with lowest mine probability
    safest = min(unrevealed, key=lambda c: mine_prob.get(c, 0.5))
    return {"type": "reveal", "row": safest[0], "col": safest[1]}


# ============================================================
# CELL 4: Expert Dataset Generator
# ============================================================

def play_expert_game(rows, cols, num_mines, seed, max_moves=200):
    """
    Play a full game using the solver and record all (state, action) pairs.
    Returns list of (prompt_text, action_json_str) tuples.
    """
    game = MinesweeperGame(rows=rows, cols=cols, num_mines=num_mines, seed=seed)
    examples = []

    for _ in range(max_moves):
        if game.state() != "ongoing":
            break

        # Get expert move
        action = solve_step(game)
        if action is None:
            break

        # Record the training example BEFORE executing the move
        prompt_text = format_state_for_llm(game)
        action_str = json.dumps(action, separators=(',', ':'))  # Compact JSON

        examples.append((prompt_text, action_str))

        # Execute move
        result = game.do_action(action)
        if result in ("mine", "game_over", "invalid_format"):
            break

    return examples, game.state()


def generate_expert_dataset(num_games=5000, rng_seed=42):
    """
    Generate expert dataset by playing many games with the solver.
    Returns list of chat-formatted training examples.
    """
    random.seed(rng_seed)

    board_configs = [
        (5, 5, 4),   # small easy
        (5, 5, 6),   # small hard
        (6, 6, 5),   # default (competition eval)
        (6, 6, 7),   # default harder
        (7, 7, 8),   # medium
        (7, 7, 10),  # medium hard
        (8, 8, 10),  # large
        (8, 8, 13),  # large hard
    ]
    # Weight toward 6x6 since that's likely eval
    weights = [1, 1, 4, 2, 1, 1, 1, 1]

    dataset = []
    wins = 0
    losses = 0
    game_count = 0

    for _ in range(num_games):
        config_idx = random.choices(range(len(board_configs)), weights=weights, k=1)[0]
        rows, cols, num_mines = board_configs[config_idx]
        seed = random.randint(0, 1_000_000)

        examples, final_state = play_expert_game(rows, cols, num_mines, seed)
        game_count += 1

        if final_state == "success":
            wins += 1
        elif final_state == "failed":
            losses += 1

        for prompt_text, action_str in examples:
            dataset.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_text},
                    {"role": "assistant", "content": action_str},
                ]
            })

    print(f"Expert dataset generation complete:")
    print(f"  Games played: {game_count}")
    print(f"  Solver wins:  {wins} ({wins/game_count*100:.1f}%)")
    print(f"  Solver losses: {losses} ({losses/game_count*100:.1f}%)")
    print(f"  Total examples: {len(dataset)}")

    # Stats
    action_types = Counter()
    for item in dataset:
        action = json.loads(item["messages"][2]["content"])
        action_types[action["type"]] += 1
    print(f"  Action distribution: {dict(action_types)}")

    return dataset


# ============================================================
# CELL 5: Generate & Validate Dataset
# ============================================================

def build_and_validate_dataset():
    """Generate the expert dataset and run sanity checks."""
    print("=" * 60)
    print("Generating expert dataset...")
    print("=" * 60)

    dataset = generate_expert_dataset(num_games=5000, rng_seed=42)

    # Validation
    print("\n--- Validation ---")
    invalid_count = 0
    for item in dataset:
        action_str = item["messages"][2]["content"]
        try:
            action = json.loads(action_str)
            assert "type" in action and "row" in action and "col" in action
            assert action["type"] in ["reveal", "flag"]
        except Exception:
            invalid_count += 1

    print(f"  Invalid actions: {invalid_count} / {len(dataset)}")
    print(f"  All actions valid: {invalid_count == 0}")

    # Show a few examples
    print("\n--- Sample Examples ---")
    for i in [0, 1, len(dataset) // 2]:
        msg = dataset[i]["messages"]
        user_prompt = msg[1]["content"]
        # Extract just the board from the prompt
        board_start = user_prompt.find('"board"')
        if board_start > 0:
            board_end = user_prompt.find(']', user_prompt.find(']', board_start) + 1) + 1
            print(f"  Example {i}: board snippet...  →  {msg[2]['content']}")
        else:
            print(f"  Example {i}: →  {msg[2]['content']}")

    # Save to disk for inspection
    with open("expert_dataset.json", "w") as f:
        json.dump(dataset, f, indent=1)
    print(f"\n  Dataset saved to expert_dataset.json ({len(dataset)} examples)")

    return dataset


# ============================================================
# CELL 6: Load Model & LoRA (same as existing notebook)
# ============================================================

def load_model():
    """Load model with Unsloth + LoRA. Run this cell in notebook."""
    from unsloth import FastLanguageModel
    import torch

    max_seq_length = 1024
    lora_rank = 16

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="/root/.cache/huggingface/models--Unsloth--Llama-3.1-8B-Instruct/snapshots/4699cc75b550f9c6f3173fb80f4703b62d946aa5",
        load_in_4bit=True,
        max_seq_length=max_seq_length,
        torch_dtype=torch.bfloat16,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_rank * 2,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    print(f"Model device: {model.device}")
    print("Model loaded with LoRA successfully!")
    return model, tokenizer, max_seq_length


# ============================================================
# CELL 7: SFT Training
# ============================================================

def train_sft(model, tokenizer, dataset, max_seq_length=1024):
    """Run SFT training on expert dataset."""
    from datasets import Dataset as HFDataset
    from trl import SFTConfig, SFTTrainer
    from transformers import TrainerCallback

    # Convert to HuggingFace Dataset
    hf_dataset = HFDataset.from_list(dataset)

    # Eval callback: play games every N steps
    class MinesweeperEvalCallback(TrainerCallback):
        def __init__(self, eval_every_steps=100, num_games=10):
            self.eval_every_steps = eval_every_steps
            self.num_games = num_games

        def on_step_end(self, args, state, control, model=None, processing_class=None, **kwargs):
            if state.global_step % self.eval_every_steps != 0:
                return
            tok = processing_class
            if tok is None or model is None:
                return
            was_training = model.training
            model.eval()
            wins = 0
            total_moves = 0
            for i in range(self.num_games):
                game = MinesweeperGame(rows=6, cols=6, num_mines=5, seed=10000 + i)
                moves = 0
                while game.state() == "ongoing" and moves < 50:
                    prompt = format_state_for_llm(game)
                    text = tok.apply_chat_template(
                        [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": prompt}],
                        tokenize=False, add_generation_prompt=True,
                    )
                    inputs = tok(text, return_tensors="pt", truncation=True,
                                 max_length=max_seq_length).to(model.device)
                    output = model.generate(
                        **inputs,
                        temperature=0.3, max_new_tokens=64, do_sample=True,
                    )
                    gen_tokens = output[0][inputs["input_ids"].shape[1]:]
                    response = tok.decode(gen_tokens, skip_special_tokens=True).strip()
                    action = parse_llm_action(response)
                    if action is None:
                        break
                    result = game.do_action(action)
                    if result in ("mine", "game_over", "invalid_format", "already_revealed",
                                  "out_of_bounds", "flagged_cell", "invalid_flag"):
                        break
                    moves += 1
                total_moves += moves
                if game.state() == "success":
                    wins += 1
            avg_moves = total_moves / self.num_games
            print(f"\n[Eval @ step {state.global_step}] Win rate: {wins}/{self.num_games} "
                  f"({wins/self.num_games*100:.0f}%) | Avg moves: {avg_moves:.1f}\n")
            if was_training:
                model.train()

    eval_callback = MinesweeperEvalCallback(eval_every_steps=100, num_games=10)

    # SFT Config
    sft_config = SFTConfig(
        output_dir="minesweeper_sft_outputs",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        optim="adamw_8bit",
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        max_seq_length=max_seq_length,
        report_to="none",
        bf16=True,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=hf_dataset,
        args=sft_config,
        callbacks=[eval_callback],
    )

    print("Starting SFT training...")
    print(f"  Dataset size: {len(hf_dataset)}")
    print(f"  Epochs: {sft_config.num_train_epochs}")
    print(f"  Effective batch: {sft_config.per_device_train_batch_size * sft_config.gradient_accumulation_steps}")
    print(f"  Learning rate: {sft_config.learning_rate}")

    trainer.train()
    print("Training complete!")

    return trainer


# ============================================================
# CELL 8: Post-Training Evaluation
# ============================================================

def evaluate_model(model, tokenizer, num_games=20, max_seq_length=1024):
    """Play full games and report win rate."""
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)

    print(f"\nEvaluating on {num_games} games (6x6, 5 mines)...")
    wins = 0
    total_moves = 0

    for i in range(num_games):
        game = MinesweeperGame(rows=6, cols=6, num_mines=5, seed=50000 + i)
        moves = 0

        while game.state() == "ongoing" and moves < 100:
            prompt = format_state_for_llm(game)
            text = tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=max_seq_length).to(model.device)
            output = model.generate(
                **inputs,
                temperature=0.3, max_new_tokens=64, do_sample=True,
            )
            gen_tokens = output[0][inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            action = parse_llm_action(response)

            if action is None:
                print(f"  Game {i+1}: PARSE FAIL after {moves} moves (response: '{response[:80]}')")
                break

            result = game.do_action(action)
            if result in ("mine", "game_over", "invalid_format", "already_revealed",
                          "out_of_bounds", "flagged_cell", "invalid_flag"):
                break
            moves += 1

        total_moves += moves
        status = "WIN" if game.state() == "success" else "LOSS"
        if game.state() == "success":
            wins += 1
        print(f"  Game {i+1}: {status} after {moves} moves")

    avg_moves = total_moves / num_games
    print(f"\nResults:")
    print(f"  Win rate: {wins}/{num_games} ({wins/num_games*100:.1f}%)")
    print(f"  Avg moves survived: {avg_moves:.1f}")
    print(f"  (GRPO baseline was: 0/20 wins, 1.7 avg moves)")

    return wins, num_games, avg_moves


# ============================================================
# CELL 9: Save Model
# ============================================================

def save_model(model, tokenizer):
    """Save LoRA adapters."""
    model.save_pretrained("my_minesweeper_model")
    tokenizer.save_pretrained("my_minesweeper_model")
    print("Model saved to: my_minesweeper_model/")


# ============================================================
# CELL 10: Main (run all steps)
# ============================================================

if __name__ == "__main__":
    # Step 1: Generate expert dataset
    dataset = build_and_validate_dataset()

    # Step 2: Load model (uncomment when running on GPU)
    # model, tokenizer, max_seq_length = load_model()

    # Step 3: Train
    # trainer = train_sft(model, tokenizer, dataset, max_seq_length)

    # Step 4: Evaluate
    # evaluate_model(model, tokenizer, num_games=20, max_seq_length=max_seq_length)

    # Step 5: Save
    # save_model(model, tokenizer)
