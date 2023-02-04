import os

os.environ["HOME_PATH"] = os.path.abspath('../bluemist')
os.environ["ARTIFACT_PATH"] = os.environ["HOME_PATH"] + '/' + 'artifacts'
print('HOME_PATH', os.environ["HOME_PATH"])
print('ARTIFACT_PATH', os.environ["ARTIFACT_PATH"])


def initialize():
    return None