import importlib


def try_plugins(plugin_name, settings, env):
    for plugin in plugin_name:
        ##Create error checking here
        spec = importlib.util.spec_from_file_location("ComaCreator", plugin)
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)                 
        foo.run(settings, env)
