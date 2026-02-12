import logging 


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True 
        elif shell == 'TerminalInteractiveShell':
            return False 
        else:
            return False 
    except NameError:
        return False     


def get_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-s | %(message)s',
        datefmt='%H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
