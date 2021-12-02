#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.
#
""" Interface to all environments. """

def make_environment(config, train=True, seed=None):
    env_name = config.get('name', 'dcs')
    if env_name == 'dcs':
        from environment_container_dcs import EnvironmentContainerDCS
        return EnvironmentContainerDCS(config, train=train, seed=seed)
    elif env_name == 'robosuite':
        from environment_container_robosuite import EnvironmentContainerRobosuite
        return EnvironmentContainerRobosuite(config, train=train, seed=seed)
    elif env_name == 'dcs_dmc_paired':
        from environment_container_dcs import EnvironmentContainerDCS_DMC_paired
        return EnvironmentContainerDCS_DMC_paired(config, train=train, seed=seed)
    else:
        raise ValueError
