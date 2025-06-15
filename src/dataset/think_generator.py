import random
from typing import Tuple, List
import re

SPINNERS = {
    "win_verbs": [
        "place", "move", "position", "set", "put", "strategize", "execute", 
        "make", "take", "secure", "claim", "establish", "create", "form"
    ],
    "win_nouns": [
        "line", "sequence", "row", "column", "diagonal", "pattern", "combination",
        "formation", "alignment", "configuration", "arrangement", "structure"
    ],
    "win_emotions": [
        "excited", "confident", "strategic", "focused", "determined", "optimistic",
        "enthusiastic", "motivated", "inspired", "energized", "eager", "thrilled"
    ],
    "win_adjectives": [
        "perfect", "brilliant", "strategic", "decisive", "crucial", "pivotal",
        "game-changing", "winning", "victorious", "triumphant", "masterful"
    ],
    "block_verbs": [
        "block", "prevent", "stop", "thwart", "counter", "defend", "protect",
        "safeguard", "secure", "fortify", "shield", "guard", "intercept"
    ],
    "block_nouns": [
        "threat", "danger", "risk", "opportunity", "chance", "possibility",
        "vulnerability", "weakness", "opening", "loophole", "gap", "breach"
    ],
    "block_emotions": [
        "cautious", "alert", "defensive", "protective", "vigilant", "watchful",
        "careful", "prudent", "attentive", "observant", "mindful", "wary"
    ],
    "block_adjectives": [
        "defensive", "protective", "preventive", "safeguarding", "securing",
        "fortifying", "shielding", "guarding", "intercepting", "blocking"
    ],
    "neutral_verbs": [
        "choose", "select", "pick", "place", "position", "strategize", "plan",
        "develop", "build", "establish", "create", "form", "shape", "craft"
    ],
    "neutral_nouns": [
        "position", "spot", "location", "square", "space", "territory",
        "ground", "area", "zone", "region", "section", "quarter"
    ],
    "neutral_emotions": [
        "calm", "patient", "strategic", "thoughtful", "methodical", "composed",
        "collected", "balanced", "steady", "measured", "deliberate", "prudent"
    ],
    "neutral_adjectives": [
        "strategic", "tactical", "calculated", "measured", "balanced",
        "well-thought", "planned", "considered", "deliberate", "prudent"
    ],
    "game_terms": [
        "board", "game", "match", "play", "move", "turn", "position",
        "strategy", "tactic", "approach", "plan", "course"
    ],
    "identity": [
        "As X", "Playing as X", "I'm X in this match", "Being player X", "Since I play as X", "I am the X player"
    ]
}

THINK_TEMPLATES = {
    "win": [
        "I detect a winning opportunity: I {win_verbs} at {move} and complete the {win_nouns}.",
        "This {win_verbs} at {move} secures a win for me.",
        "I can win by {win_verbs}ing at {move}, completing my {win_nouns}.",
        "{win_verbs}ing at {move} allows me to finish a {win_nouns}.",
        "Victory is within reach. I {win_verbs} to {move} to win.",
        "I'm feeling {win_emotions} about this move at {move} - it will complete my {win_nouns}.",
        "The {win_adjectives} {win_verbs} at {move} will give me the win.",
        "I see a clear path to victory by {win_verbs}ing at {move}.",
        "This {win_adjectives} {win_nouns} at {move} will secure my victory.",
        "I'm {win_emotions} about this {win_adjectives} opportunity at {move}.",
        "The {game_terms} is mine if I {win_verbs} at {move}.",
        "I feel {win_emotions} as I {win_verbs} at {move} to complete my {win_nouns}."
    ],
    "block": [
        "The opponent is threatening a win. I {block_verbs} at {move}.",
        "To {block_verbs} the opponent from winning, I must move to {move}.",
        "There's a {block_nouns} of losing if I don't {block_verbs} at {move}.",
        "The opponent is one move away. I {block_verbs} them by {block_verbs}ing at {move}.",
        "To avoid defeat, I make a {block_emotions} move at {move}.",
        "I need to be {block_emotions} and {block_verbs} at {move}.",
        "This {block_nouns} requires immediate action - I {block_verbs} at {move}.",
        "I'm feeling {block_emotions} about this {block_nouns}, so I {block_verbs} at {move}.",
        "A {block_adjectives} response is needed at {move} to prevent their win.",
        "I must {block_verbs} this {block_nouns} by moving to {move}.",
        "The {game_terms} demands a {block_adjectives} move at {move}.",
        "I'm {block_emotions} as I {block_verbs} their {block_nouns} at {move}."
    ],
    "neutral": [
        "There are no immediate {block_nouns}s, so I {neutral_verbs} {move} for {neutral_nouns}.",
        "I {neutral_verbs} at {move} to control the center or setup future plays.",
        "No winning or blocking move is required, so I {neutral_verbs} {move}.",
        "I take an open {neutral_nouns} at {move} to improve my chances later.",
        "The board is balanced, I {neutral_verbs} at {move} for better {neutral_nouns}.",
        "I'm feeling {neutral_emotions} about this {neutral_nouns} at {move}.",
        "This {neutral_verbs} at {move} sets up a strong {neutral_nouns}.",
        "I {neutral_verbs} {move} with a {neutral_emotions} approach to the game.",
        "A {neutral_adjectives} {neutral_nouns} at {move} will strengthen my {game_terms}.",
        "I'm taking a {neutral_emotions} approach by {neutral_verbs}ing at {move}.",
        "This {neutral_adjectives} move at {move} improves my {game_terms} position.",
        "I {neutral_verbs} at {move} to maintain a {neutral_emotions} {game_terms}."
    ]
}

def spin_text(template: str) -> str:
    """
    Reemplaza placeholders con spinners, soportando variantes como {verb}ing, ignorando {move}.
    """
    def replace(match):
        key = match.group(1)
        suffix = match.group(2) or ""

        if key == "move":
            return f"{{{key}}}{suffix}"  # lo dejamos sin tocar

        word = random.choice(SPINNERS[key])
        if suffix == "ing":
            if word.endswith("e") and not word.endswith("ee"):
                word = word[:-1]
            word += "ing"
        return word

    return re.sub(r"{(\w+)}(ing)?", replace, template)



def generate_think_text(kind: str, move: Tuple[int, int]) -> str:
    """
    Genera texto de razonamiento con spinners y, aleatoriamente,
    agrega una frase que indica que se juega como 'X'.
    """
    template = random.choice(THINK_TEMPLATES[kind])
    move_str = f"{move[0]}-{move[1]}"
    spun_text = spin_text(template).format(move=move_str)

    if random.random() < 0.5:  # 50% de las veces añade identidad
        identity = random.choice(SPINNERS["identity"])
        spun_text = f"{identity}, {spun_text[0].lower() + spun_text[1:]}"  # minúscula tras coma

    return f"<player_think>{spun_text}</player_think>"

if __name__ == "__main__":
    for kind in ["win", "block", "neutral"]:
        print(f"\nEjemplos de {kind}:")
        for _ in range(3):
            print(generate_think_text(kind, (1, 1)))
