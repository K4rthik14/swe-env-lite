import shlex
import sys
from pathlib import Path

from swe_env import SWEEnv
from swe_env.types import SWEAction


def main() -> None:
    repo_path = Path(__file__).resolve().parents[1]
    env = SWEEnv(repo_path=repo_path, max_steps=1)
    observation, info = env.reset()

    print("Initial workspace entries")
    print(observation.stdout)
    print("Info:", info)

    command = f"{shlex.quote(sys.executable)} -m pytest -q"
    result = env.step(SWEAction(command))
    print("\nCommand output:")
    print(result.observation.stdout)
    print("Reward:", result.reward)
    print("Terminated:", result.terminated)
    print("Truncated:", result.truncated)
    print("Info:", result.info)


if __name__ == "__main__":
    main()
