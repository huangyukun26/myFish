import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fishnet.env import print_environment_report


if __name__ == "__main__":
    print_environment_report()
