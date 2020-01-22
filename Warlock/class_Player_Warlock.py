class PlayerWarlock:
    def __init__(self, warlock_hp=30000):
        self.life = warlock_hp
        # intellect, citical, haste, mastery, versatility,
        # self.intellect = intellect
        # self.citical = citical
        # self.haste = haste
        # self.mastery = mastery
        # self.versatility = versatility
        self.position = [0, 0]
        self.ashes = 2
        self.gcd = 5               # global cool down
        self.cd = {'deflagration': 0, 'dark_soul': 0}
        self.buff = {'deflagration_buff': {'available_times': 0, 'remaining_duration': 0},
                     'dark_soul_buff': {'remaining_duration': 0}}

    # def choice_origin_position(self):
    #     pass
    #
    # def move(self, direction):
    #     # ['forward', 'back', 'left', 'right']
    #     if direction == 'forward':
    #         self.position[1] += 1
    #     elif direction == 'back':
    #         self.position[1] -= 1
    #     elif direction == 'left':
    #         self.position[0] -= 1
    #     elif direction == 'right':
    #         self.position[0] += 1
    #     else:
    #         raise ValueError('Move in an unknown direction.')

    """脑残箭"""
    def chaos_bolt(self):
        if self.ashes >= 20:
            self.ashes -= 20
        else:
            raise ValueError('Your ashes are insufficient.')
        damage = 3000
        cast_time = 10
        if self.buff['dark_soul_buff']['remaining_duration'] > cast_time:
            damage *= 1.5
        return {'damage': damage, 'cast_time': cast_time}

    """爆燃"""
    def deflagration(self):
        self.ashes = min(50, self.ashes+3)
        damage = 500
        cast_time = 0
        self.buff['deflagration_buff'] = {'available_times': 2, 'remaining_duration': 150}
        self.cd['deflagration'] = 40
        return {'damage': damage, 'cast_time': cast_time}

    """烧尽"""
    def burn_out(self):
        self.ashes = min(50, self.ashes + 2)
        damage = 500
        cast_time = 5
        if self.buff['deflagration_buff']['available_times'] > 0:
            cast_time = 3
            self.buff['deflagration_buff']['available_times'] -= 1
            if self.buff['deflagration_buff']['available_times'] == 0:
                self.buff['deflagration_buff']['remaining_duration'] = 0
        return {'damage': damage, 'cast_time': cast_time}

    def dark_soul(self):
        damage = 0
        cast_time = 0
        self.cd['dark_soul'] = 600
        self.buff['dark_soul_buff']['remaining_duration'] = 125
        return {'damage': damage, 'cast_time': cast_time}

    # """献祭 dot"""
    # def immolate(self):
    #     pass
    #
    # """火雨 aoe"""
    # def rain_of_fire(self):
    #     pass
    #
    # """吸血"""
    # def drain_life(self):
    #     pass
    # def interrupt_cast(self):
    #     pass
