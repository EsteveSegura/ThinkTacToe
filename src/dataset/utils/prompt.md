# Introduction
You are a specialist in drafting thoughts of a professional Tic Tac Toe player.

## How to move
To indicate a move, use the format (row, column), where indexing is **0-based**:
- (0,0) means top-left
- (0,1) means top-center
- (0,2) means top-right
- (1,0) means middle-left
- (1,1) means center
- (1,2) means middle-right
- (2,0) means bottom-left
- (2,1) means bottom-center
- (2,2) means bottom-right

## Game state
The board is presented as a 3x3 array:
- Squares with an empty string `""` are free.
- Squares with `"X"` belong to player X.
- Squares with `"O"` belong to player O.

The current board state is:

```json
{{BOARD_INIT_STATE}}
```

### Game state explained with language
{{BOARD_EXPLAINED}}

### Notes

- Do not start <player_think> with phrases like "Okay" o "lets analyze" o "Right"
{{EXTRA_NOTES}}

## Mission

Player `"X"` will now move to position {{POSITION}}, leaving the board in this state:

```json
{{BOARD_WITH_MOVEMENT_APPLIED}}
```

Write the thought that player `"X"` had to make this move. Express it in a concise way, with a text of around 100 and 400 words.

And wrap the player's thought in this tag: `<player_think> PLACEHOLDER </player_think>`.