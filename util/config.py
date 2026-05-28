from mmengine import Config

def get_config_from_path(config_path):
    config = Config.fromfile(config_path)
    return config
