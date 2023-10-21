
import os
import asyncio
import sys
import tensorflow as tf
from abc import ABC
from gym.spaces import Box, Space
import numpy as np
from tqdm import tqdm
from poke_env.environment.weather import Weather
from poke_env.environment.field import Field
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.battle_order import BattleOrder, DoubleBattleOrder, ForfeitBattleOrder
from poke_env.player import (
    EnvPlayer,
    ObservationType,
    RandomPlayer,
    wrap_for_old_gym_api,
)
from poke_env.data import GenData

from poke_env.teambuilder import Teambuilder

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from tabulate import tabulate
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers.legacy import Adam
from model_player import ModelPlayer
from rl_util import rl_action_to_move, rl_embed_battle, generate_doubles_actions

team1 = '''
Cresselia @ Safety Goggles  
Ability: Levitate  
Level: 100
Tera Type: Fairy  
EVs: 252 HP / 156 Def / 100 SpD  
Relaxed Nature  
IVs: 0 Atk  
- Moonblast  
- Helping Hand  
- Trick Room  
- Lunar Blessing  

Torkoal @ Charcoal  
Ability: Drought  
Level: 100
Tera Type: Grass  
EVs: 252 HP / 252 SpA / 4 SpD  
Quiet Nature  
IVs: 0 Atk / 0 Spe  
- Eruption  
- Heat Wave  
- Earth Power  
- Protect  

Iron Hands @ Assault Vest  
Ability: Quark Drive  
Level: 100
Tera Type: Fairy  
EVs: 252 HP / 156 Atk / 100 SpD  
Brave Nature  
IVs: 9 Spe  
- Fake Out  
- Drain Punch  
- Wild Charge  
- Heavy Slam  

Ursaluna @ Flame Orb  
Ability: Guts  
Level: 100
Tera Type: Ghost  
EVs: 252 HP / 156 Atk / 100 SpD  
Brave Nature  
IVs: 0 Spe  
- Earthquake  
- Facade  
- Swords Dance  
- Protect  
'''

team2 = '''
Landorus-Therian @ Safety Goggles  
Ability: Intimidate  
Level: 100
Tera Type: Flying  
EVs: 252 HP / 36 Atk / 84 Def / 68 SpD / 68 Spe  
Adamant Nature  
- Stomping Tantrum  
- Tera Blast  
- U-turn  
- Protect  

Urshifu-Rapid-Strike @ Mystic Water  
Ability: Unseen Fist  
Level: 100
Tera Type: Water  
EVs: 44 HP / 196 Atk / 4 Def / 84 SpD / 180 Spe  
Adamant Nature  
- Surging Strikes  
- Aqua Jet  
- Protect  
- Taunt  

Flutter Mane @ Choice Specs  
Ability: Protosynthesis  
Level: 100  
Tera Type: Grass  
EVs: 68 HP / 140 Def / 132 SpA / 4 SpD / 164 Spe  
Timid Nature  
IVs: 0 Atk  
- Shadow Ball  
- Dazzling Gleam  
- Moonblast  
- Energy Ball  

Chien-Pao @ Focus Sash  
Ability: Sword of Ruin  
Level: 100  
Tera Type: Ghost  
EVs: 252 Atk / 4 Def / 252 Spe  
Adamant Nature  
- Icicle Crash  
- Sacred Sword  
- Sucker Punch  
- Protect  
'''

class Gen9EnvDoublePlayer(EnvPlayer, ABC):
    _ACTION_SPACE = list(range(642))
    _DEFAULT_BATTLE_FORMAT = "gen9doublesubers"
    _all_possible_doubles_actions = None

    def action_to_move(self, action: int, battle: AbstractBattle) -> BattleOrder:
        if not self._all_possible_doubles_actions:
            self._all_possible_doubles_actions = generate_doubles_actions()
        doubles_action = self._all_possible_doubles_actions[action % 642]
        return rl_action_to_move(doubles_action,battle)

class SimpleRLPlayer(Gen9EnvDoublePlayer):
    def calc_reward(self, last_battle, current_battle) -> float:
        if current_battle not in self._reward_buffer:
            self._reward_buffer[current_battle] = -1
        current_value = -1

        current_faint_count = sum([mon.fainted for mon in current_battle.team.values()])
        last_faint_count = sum([mon.fainted for mon in last_battle.team.values()])
        current_value -= (current_faint_count - last_faint_count)*5

        current_faint_count = sum([mon.fainted for mon in current_battle.opponent_team.values()])
        last_faint_count = sum([mon.fainted for mon in last_battle.opponent_team.values()])
        current_value += (current_faint_count - last_faint_count)*5
        
        if current_battle.won:
            current_value += 30
        elif current_battle.lost:
            current_value -= 30

        return current_value

    def embed_battle(self, battle: AbstractBattle) -> ObservationType:
        moves_base_power = -np.ones(16)
        moves_dmg_multiplier = -np.ones(16)

        for my_id in range(2):
            my_poke = battle.active_pokemon[my_id]
            for op_id in range(2):
                op_poke = battle.active_pokemon[op_id]
                for mo_id in range(4):
                    if my_poke and op_poke and list(my_poke.moves.values())[mo_id].id in [m.id for m in battle.available_moves[my_id]]:
                        move = list(my_poke.moves.values())[mo_id]
                        moves_base_power[8*my_id + 4*op_id + mo_id] = move.base_power / 100
                        if move.type:
                            moves_dmg_multiplier[8*my_id + 4*op_id + mo_id] = move.type.damage_multiplier(
                                op_poke.type_1,
                                op_poke.type_2,
                                type_chart = GenData.from_gen(9).type_chart
                            )

        active_mons_hp_fraction = [mon.current_hp_fraction if mon else 0 for mon in battle.all_active_pokemons]

        fields = [int(f.name in battle.fields) for f in Weather]
        weathers = [int(w.name in battle.weather) for w in Field]

        final_vector = np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                active_mons_hp_fraction,
                fields,
                weathers,
                [0 if not tera else 1 for tera in battle.can_tera],
                [int(fs) for fs in battle.force_switch],
                [mon.boosts["atk"] if mon else 0 for mon in battle.active_pokemon],
            ]
        )
        return np.float32(final_vector)

    def describe_embedding(self) -> Space:
        low = [-1]*16 + [0]*16 + [0]*2 + [0]*2 + [0]*22 + [0]*2 + [0]*2 + [-6]*2
        high = [3]*16 + [8]*16 + [1]*2 + [0]*2 + [1]*22 + [1]*2 + [1]*2 + [6]*2
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )

def build_dqn(train_env):
    # Compute dimensions
    n_action = train_env.action_space.n
    input_shape = (1,) + train_env.observation_space.shape

    model = Sequential()
    model.add(Dense(128, activation="elu", input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(64, activation="elu"))
    model.add(Dense(n_action, activation="linear"))

    # Defining the DQN
    memory = SequentialMemory(limit=10000, window_length=1)

    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.05,
        value_test=0.0,
        nb_steps=10000,
    )

    dqn = DQNAgent(
        model=model,
        nb_actions=n_action,
        policy=policy,
        memory=memory,
        nb_steps_warmup=1000,
        gamma=0.5,
        target_model_update=1,
        delta_clip=0.01,
        enable_double_dqn=True,
    )
    dqn.compile(Adam(learning_rate=0.00025), metrics=["mae"])
    return dqn, model

async def main():

    p1 = RandomPlayer(
        battle_format="gen9doublesubers",
        team=team1,
        save_replays=False,
    )

    train_env = SimpleRLPlayer(
        battle_format="gen9doublesubers", opponent=p1, start_challenging=True, team=team2,
    )
    train_env = wrap_for_old_gym_api(train_env)

    # Training the model
    dqn, model = build_dqn(train_env)
    dqn.fit(train_env, nb_steps=1000)
    model.save('double-model-team2')
    train_env.close()

    p2 = ModelPlayer(
        battle_format="gen9doublesubers",
        team=team2,
        save_replays=False,
    )

    train_env = SimpleRLPlayer(
        battle_format="gen9doublesubers", opponent=p2, start_challenging=True, team=team1
    )
    train_env = wrap_for_old_gym_api(train_env)

    dqn, model = build_dqn(train_env)
    dqn.fit(train_env, nb_steps=1000)
    model.save('double-model-team1')
    train_env.close()

    p3 = ModelPlayer(
        battle_format="gen9doublesubers",
        team=team2,
        save_replays=True,
    )
    p3.load_model('double-model-team2')

    p4 = ModelPlayer(
        battle_format="gen9doublesubers",
        team=team1,
        save_replays=False,
    )
    p4.load_model('double-model-team1')

    await p3.battle_against(p4,n_battles=100)
    print(
        "P3 won %d / 100 battles"
        % (p3.n_won_battles)
    )

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())