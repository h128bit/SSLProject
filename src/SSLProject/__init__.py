from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("sslproject")
except PackageNotFoundError:
    # Пакет не установлен (например, запуск из исходников без pip install)
    __version__ = "unknown"