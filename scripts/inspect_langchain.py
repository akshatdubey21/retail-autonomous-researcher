import importlib, sys, pkgutil
print('python', sys.executable)
try:
    import langchain
    print('langchain version:', getattr(langchain, '__version__', 'unknown'))
except Exception as e:
    print('langchain import error:', e)

spec = importlib.util.find_spec('langchain.schema')
print('langchain.schema spec:', spec)

try:
    modules = list(m.name for m in pkgutil.iter_modules(langchain.__path__))
    print('langchain modules sample:', modules[:100])
except Exception as e:
    print('error listing modules', e)
