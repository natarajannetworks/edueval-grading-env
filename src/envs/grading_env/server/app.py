from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.envs.grading_env.models import GradingAction, GradingObservation, GradingState
from src.envs.grading_env.server.environment import GradingEnvironment

app = FastAPI(
    title="EduEval Grading Environment",
    description="OpenEnv-compliant answer sheet grading environment",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

environments = {
    1: GradingEnvironment(task_id=1),
    2: GradingEnvironment(task_id=2),
    3: GradingEnvironment(task_id=3),
}

@app.post("/reset", response_model=GradingObservation)
def reset(task_id: int = 1):
    if task_id not in environments:
        raise HTTPException(400, "task_id must be 1, 2, or 3")
    return environments[task_id].reset()

@app.post("/step")
def step(action: GradingAction, task_id: int = 1):
    if task_id not in environments:
        raise HTTPException(400, "task_id must be 1, 2, or 3")
    env = environments[task_id]
    if env.state().is_complete:
        raise HTTPException(400, "Episode complete. Call /reset first.")
    obs, reward, done = env.step(action)
    return {
        "observation": obs.dict(),
        "reward": reward,
        "done": done,
        "info": {
            "episode_id": env.episode_id,
            "step": env.current_index
        }
    }

@app.get("/state", response_model=GradingState)
def state(task_id: int = 1):
    if task_id not in environments:
        raise HTTPException(400, "task_id must be 1, 2, or 3")
    return environments[task_id].state()

@app.get("/health")
def health():
    return {"status": "ok", "env": "EduEval"}