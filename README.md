# Reinforcement learning games

## Pre-requisites

Install pipenv and type:

```
pipenv install
pipenv shell
```

## Noughts and crosses

To try your hand playing noughts and crosses against the computer (trained with a Q-table):

```
python -m rl_games.games.q_learner.nac_play
```

To play the version trained with a deep Q network, use

```
python -m rl_games.games.dqn.nac_play
```

## Chopsticks

To try your hand playing chopsticks against the computer (trained with a Q-table):

```
python -m rl_games.games.q_learner.chopsticks_play
```
