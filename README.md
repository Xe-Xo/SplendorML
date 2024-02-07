# SplendorML
 Experiments with Splendor


## What is Splendor?

Splendor is a multiplayer card-based board game, designed by Marc André and illustrated by Pascal Quidault. It was published in 2014 by Space Cowboys, Asmodee. Players are gem merchants of the Renaissance, developing gem mines, transportation, and shops to accumulate prestige points

### Gameplay

Splendor is an engine-building and resource management game in which two to four players compete to collect the most prestige points. The game has the following components:

40 gem tokens - seven each of emerald, sapphire, ruby, diamond, onyx, and five gold (wild). These are represented by poker-style chips.[3]
90 development cards
10 noble tiles
Each development card falls into one of three levels (•, ••, •••) indicating the difficulty to purchase that card.

Every development card also contains a gem bonus (emerald, sapphire, ruby, diamond, or onyx), which may be used for future development card purchases.

Before the game begins, 5 Noble tiles are dealt face up in the center. Four cards from each level (•, ••, •••) are dealt face up, visible to the players, and the rest are kept in separate decks of their corresponding difficulty levels.

Turns
A player's turn consists of a single action, which must be one of the following:

1) Take up to three gem tokens of different colors from the pool.
2) Take two gem tokens of the same color from the pool (provided, before the turn, there are at least four tokens left of that color).
3) Take one gold gem token and reserve one development card (if, before the turn, number of reserved cards held by that player is less than three)
4) Purchase a development card (from the table or the player's reserved cards) by spending the required gem tokens and/or using the gem bonus of the cards purchased previously by the player. Spent gems return to the pool.

After this action:

1)  If the player has earned enough development card gem bonuses to obtain Noble prestige points, that player is "visited" by the Noble, and takes that Noble tile.
    A player can only be visited by one noble in each turn; the player chooses between the eligible nobles.
    Player keeps the Noble until the end of the game.
    Noble tiles are not replenished.
2) If player possesses more than 10 gems, player returns enough gems to limit total count of gems in player's possession to 10 or fewer.
3) If a development card is purchased, the empty slot is replenished from the respective deck. If the deck of that rank has run out, the slot stays empty.

When one player reaches 15 prestige points, the players continue playing the current round until each player has taken the same number of turns. Once this occurs, the game ends.

## Game State

A Game State represents a singular action by the Player in context of the game Splendor.

The Actions a player can take are grouped by category:
1) TAKE_3_GEMS - 10 Possible Options
2) TAKE_2_GEMS - 5 Possible Options
3) RESERVE_CARD - 12 Possible Options
4) BUY_CARD - 12 Possible Options
5) BUY_RESERVED_CARD - 3 Possible Options
6) VISIT_NOBLE_CHOICE - 5 Possible Options
7) DISCARD_GEM - 6 Possible Options

An action mask is generated after each turn to validate the possible actions that can be taken.
eg. Not enough Gems/Cards to Buy a Card.

The action state helps keep track of the potential upcoming actions
1) TAKE_TURN
2) DISCARD_GEM
3) VISIT_NOBLE
4) GAME_OVER
5) RANDOM_CARD

### Evaluating a Game State

#### Static Evaluation

A Static Evaluation of whether a Game State is "good" has be simplified to:
- the number of victory points the player has compared to highest opponents victory points

The ultimate goal being to achieve 15 points before anyone else staying ahead of the pack is one way to win the game.

#### Heuristic Evaluation

A Heuristic Evaluation of whether a Game State is "good" can be further expanded to include the following features:
- The number of victory points
- The number of gems a player has (with a maximum limit of 10)
- A fraction of the victory points of the unclaimed nobles based on the completion status of the noble 
- A fraction of The victory points that can be acquired when purchasing a card that is available x The Cards Available
- The victory points that can be acquired when purchasing a reserved card x Reserved Cards the Player Owns

Generally aiming to calculate the strength of the player's resource engine based on the current state.
A better engine generally means that victory points are easier to collect than the opponent and the player will win the game.


#### Expectiminimax Search

A Tree search can be completed from the current Game State to evaluate moves ahead.
The Branching Factor for Splendor is quite high and a minimum depth search is usually +3 (to get to the current players turn again)
This means that significant pruning is need and further optimisations are required.

Futher complications involve the degree of randomness that can occur. When a Card is Reserved or Bought from the available pool it is replaced by a card remaining in the deck.
This means that the branching factor can skyrocket to +30-40.

# TODOS

- Game Over happens as soon as a player reaches 15 points
- Discard Action happens the number of times required to each 10 gems (1-3). I wish to simplify this to one action
- Futher optimisation of heuristic search
- Futher pruning optimisations eg. If the 3rd player reserving a card is bad it should be considered last on every other branch regardless of the 2nd player's actions.





