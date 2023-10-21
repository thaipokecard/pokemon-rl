from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.weather import Weather
from poke_env.environment.field import Field
from poke_env.player import ObservationType
from poke_env.player.battle_order import BattleOrder, DoubleBattleOrder, ForfeitBattleOrder
from poke_env.data import GenData

import numpy as np

def rl_embed_battle(battle: AbstractBattle) -> ObservationType:
    moves_base_power = -np.ones(16)
    moves_dmg_multiplier = -np.ones(16)

    for my_id in range(2):
        my_poke = battle.active_pokemon[my_id]
        for op_id in range(2):
            op_poke = battle.active_pokemon[op_id]
            if my_poke and op_poke:
                original_move_list = list(my_poke.moves.values())
                for mo_id in range(4):
                    original_move = original_move_list[mo_id]
                    embed_index = 8*my_id + 4*op_id + mo_id
                    if original_move.id in [m.id for m in battle.available_moves[my_id]]:
                        moves_base_power[embed_index] = original_move.base_power / 100
                        if original_move.type:
                            moves_dmg_multiplier[embed_index] = original_move.type.damage_multiplier(
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

def generate_single_actions() -> list[str]:
    actions = ['move {0} {1} {2}'.format(move_id, target_id, tera) for move_id in range(4) for target_id in range(4) for tera in range(2)]
    actions.extend(['switch {0}'.format(switch_id) for switch_id in range(2)])
    return actions

def generate_doubles_actions() -> list[tuple[str,str]]:
    doubles_actions = []
    p1_actions = generate_single_actions()
    p2_actions = generate_single_actions()
    for p1_action in p1_actions:
        for p2_action in p2_actions:
            doubles_action = (p1_action, p2_action)
            if p1_action.startswith('move') and p2_action.startswith('move'):
                if int(p1_action[-1]) + int(p1_action[-1]) < 2:
                    doubles_actions.append(doubles_action)
            elif p1_action.startswith('switch') and p2_action.startswith('switch'):
                if p1_action[-1] != p2_action[-1]:
                    doubles_actions.append(doubles_action)
            else:
                doubles_actions.append(doubles_action)
    return doubles_actions

def unpack_action(active_id: int, action: str, battle: AbstractBattle) -> list[any]:
    possible_targets = [2,1,-1,-2]
    moves = action.split(' ')
    if moves[0] == 'move':
        available_move = None
        move_target = possible_targets[int(moves[2])]
        will_tera = moves[3] == '1'
        for x in battle.available_moves[active_id]:
            if x.id == list(battle.active_pokemon[active_id].moves.values())[int(moves[1])].id:
                available_move = x
        if available_move and move_target in battle.get_possible_showdown_targets(available_move,battle.active_pokemon[active_id]) and not (will_tera and not battle.can_tera[active_id]):
            moves[1] = available_move
            moves[2] = move_target
            moves[3] = will_tera
        else:
            moves = ['illegal']
    else: # switch
        switch_id = int(moves[1])
        if switch_id < len(battle.available_switches[active_id]):
            moves[1] = battle.available_switches[active_id][switch_id]
        else:
            moves = ['illegal']
    return moves

def rl_action_to_move(actions: tuple[str,str], battle: AbstractBattle, default: BattleOrder = ForfeitBattleOrder()) -> BattleOrder:
    battle_moves = [[],[]]
    if sum(battle.force_switch) == 1:
        for i in range(2):
            if battle.force_switch[i]:
                action = actions[i]
                moves = unpack_action(i, action, battle)
                if moves[0] == 'switch':
                    battle_moves[i].append(BattleOrder(moves[1]))
                else:
                    return default
    elif sum(battle.force_switch) == 2:
        for i in range(2):
            action = actions[i]
            moves = unpack_action(i, action, battle)
            if moves[0] == 'switch':
                battle_moves[i].append(BattleOrder(moves[1]))
            else:
                return default
    else:
        for i in range(2):
            if not battle.active_pokemon[i]:
                continue

            action = actions[i]
            moves = unpack_action(i, action, battle)
            if moves[0] == 'move':
                battle_moves[i].append(BattleOrder(moves[1],move_target=moves[2],terastallize=moves[3]))
            elif moves[0] == 'switch':
                battle_moves[i].append(BattleOrder(moves[1]))
            else:
                return default

    orders = DoubleBattleOrder.join_orders(*battle_moves)
    if orders:
        return orders[0]
    else:
        return default

if __name__ == "__main__": 
    print(len(generate_doubles_actions()))