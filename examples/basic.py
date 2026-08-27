from swe_env import SWEEnv

def main():
  env = SWEEnv(repo_path="examples/tasks/buggy_repo")
  obs,info = env.reset()

  print("Initial Observation")
  print(obs)
  obs,reward,terminated, truncated,info = env.step("pytest")
  print("\nReward",reward)
  print("\nTerminated",terminated)
  print("\nTruncated",truncated)
  print("\nInfo",info)

if __name__ == "__main__":
  main()


