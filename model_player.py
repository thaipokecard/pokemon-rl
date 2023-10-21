from poke_env.data import GenData
from poke_env.player import Player, ObservationType
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.weather import Weather
from poke_env.environment.field import Field
from poke_env.player.battle_order import BattleOrder, DoubleBattleOrder, DefaultBattleOrder

from poke_env.teambuilder.teambuilder import Teambuilder

from typing import Optional, Union

import tensorflow as tf

import numpy as np
from rl_util import rl_action_to_move, rl_embed_battle, generate_doubles_actions

class ModelPlayer(Player):

    model = None 
    model_path = 'double-model-team2'
    _ACTION_SPACE = list(range(642))
    _all_possible_doubles_actions = None

    def load_model(self, path):
        self.model_path = path

    def _battle_finished_callback(self, battle: AbstractBattle):
        pass

    def choose_move(self, battle):
        if not self._all_possible_doubles_actions:
            self._all_possible_doubles_actions = generate_doubles_actions()
        if not self.model:
            self.model = tf.keras.models.load_model(self.model_path)
        state = self._embed_battle(battle)
        state = np.reshape(state,(1,1,len(state)))
        action_prob = self.model.predict(state)
        action_prob = action_prob[0]
        action = np.argmax(action_prob)
        return self._action_to_move(action,battle)
    
    def _embed_battle(self, battle: AbstractBattle) -> ObservationType:
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

    def _action_to_move(self, action: int, battle: AbstractBattle) -> BattleOrder:
        doubles_action = self._all_possible_doubles_actions[action % 642]
        random_move = self.choose_random_doubles_move(battle)
        return rl_action_to_move(doubles_action,battle,default=random_move)