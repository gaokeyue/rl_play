from Warlock.class_Player_Warlock import PlayerWarlock
from game.game import Game


class GameWarlock(Game):
    def __init__(self, boss_hp):
        self.boss_hp = boss_hp
        self.boss_current_hp = self.boss_hp
        self.battle_duration = 0
        self.warlock = PlayerWarlock()
        self.is_terminal = False
        self.reward = None
        # self.state = {'ashes': self.warlock.ashes,
        #               'dark_soul_buff': 0,
        #               'dark_soul_cd': 0,
        #               'deflagration_buff_times': 0,
        #               'deflagration_buff': 0,
        #               'deflagration_cd': 0}
        self.state = (self.warlock.ashes, 0, 0, 0, 0, 0)
        self.debug_info = None

    def reset(self):
        self.boss_current_hp = self.boss_hp
        self.warlock = PlayerWarlock()
        self.is_terminal = False
        # self.state = {'ashes': self.warlock.ashes,
        #               'dark_soul_buff': 0,
        #               'dark_soul_cd': 0,
        #               'deflagration_buff_times': 0,
        #               'deflagration_buff': 0,
        #               'deflagration_cd': 0}
        self.state = (self.warlock.ashes, 0, 0, 0, 0, 0)

    def one_move(self, action):
        if action == 'burn_out':
            result = self.warlock.burn_out()
        elif action == 'chaos_bolt':
            result = self.warlock.chaos_bolt()
        elif action == 'deflagration':
            result = self.warlock.deflagration()
        elif action == 'dark_soul':
            result = self.warlock.dark_soul()
        else:
            raise ValueError('Unknow action!')
        self.boss_current_hp -= result['damage']
        self.battle_duration += max(result['cast_time'], self.warlock.gcd)
        for skill in self.warlock.cd:
            self.warlock.cd[skill] -= max(result['cast_time'], self.warlock.gcd)
            self.warlock.cd[skill] = max(0, self.warlock.cd[skill])
        for buff in self.warlock.buff:
            self.warlock.buff[buff]['remaining_duration'] -= max(result['cast_time'], self.warlock.gcd)
            self.warlock.buff[buff]['remaining_duration'] = max(0, self.warlock.buff[buff]['remaining_duration'])
        if self.warlock.buff['deflagration_buff']['remaining_duration'] == 0:
            self.warlock.buff['deflagration_buff']['available_times'] = 0
        if self.boss_current_hp <= 0:
            self.is_terminal = True
            self.reward = 999
        else:
            self.reward = -max(result['cast_time'], self.warlock.gcd)
        # self.state = {'ashes': self.warlock.ashes,
        #               'dark_soul_buff': self.warlock.buff['dark_soul_buff']['remaining_duration'],
        #               'dark_soul_cd': self.warlock.cd['dark_soul'],
        #               'deflagration_buff_times': self.warlock.buff['deflagration_buff']['available_times'],
        #               'deflagration_buff': self.warlock.buff['deflagration_buff']['remaining_duration'],
        #               'deflagration_cd': self.warlock.cd['deflagration']}
        self.state = (self.warlock.ashes,
                      self.warlock.buff['dark_soul_buff']['remaining_duration'],
                      self.warlock.cd['dark_soul'],
                      self.warlock.buff['deflagration_buff']['available_times'],
                      self.warlock.buff['deflagration_buff']['remaining_duration'],
                      self.warlock.cd['deflagration'])
        return self.state, self.reward, self.is_terminal, self.debug_info

    def available_actions(self, state=None):
        available_actions = ['burn_out']
        if self.warlock.ashes >= 20:
            available_actions.append('chaos_bolt')
        if self.warlock.cd['deflagration'] == 0:
            available_actions.append('deflagration')
        if self.warlock.cd['dark_soul'] == 0:
            available_actions.append('dark_soul')
        return available_actions



