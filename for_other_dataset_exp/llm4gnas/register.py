model_factory = {

}


def build(model_name, config):
    assert model_name in model_factory, f"{model_name} not in model_factory"
    return model_factory[model_name](config)


def register(func, name=None):
    name = func.__name__ if name is None else name
    model_factory[name] = func
    return func