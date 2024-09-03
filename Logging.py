import logging


def open_log(log_path):
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=log_path,
                        level=logging.INFO,
                        format="%(asctime)s %(levelname)-8s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")