from dataclasses import dataclass


@dataclass
class SWEAction:
  command: str

@dataclass
class SWEObservation:
  stdout: str
  stderr: str
  exit_code: int
  diff: str

@dataclass
class StepResult:
  observation: SWEObservation
  reward: float
  terminated: bool
  truncated: bool
  info: dict
