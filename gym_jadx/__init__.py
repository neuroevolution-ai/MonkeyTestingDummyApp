from gym.envs.registration import register

register(
    id='jadx-v0',
    entry_point='gym_jadx.envs:JadxEnv',
)
