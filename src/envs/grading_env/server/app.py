from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from envs.grading_env.models import GradingAction, GradingObservation, GradingState
from .environment import GradingEnvironment

app = FastAPI(
    title="EduEval Grading Environment",
    description="OpenEnv environment for automated answer sheet grading and evaluation",
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

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
        <head>
            <title>EduEval Grading Environment</title>
            <script src="https://cdn.tailwindcss.com"></script>
        </head>
        <body class="bg-gradient-to-br from-indigo-50 to-blue-100 min-h-screen flex items-center justify-center p-6">
            <div class="max-w-4xl w-full">
                <div class="text-center mb-10">
                    <h1 class="text-6xl font-extrabold text-indigo-700 mb-3">📚 EduEval</h1>
                    <p class="text-xl text-gray-600">Automated Answer Sheet Grading Environment</p>
                    <p class="text-sm text-indigo-400 mt-2">OpenEnv · Reinforcement Learning · Education AI</p>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">

                    <div class="bg-white rounded-2xl p-6 shadow-md border border-green-100">
                        <div class="text-3xl mb-2">🟢</div>
                        <h3 class="font-bold text-lg text-gray-800 mb-1">Task 1 — Factual Grading</h3>
                        <p class="text-gray-500 text-sm">
                            Grade objective questions with clear right or wrong answers.
                            Strict binary-leaning reward function.
                        </p>
                        <div class="mt-3 text-green-600 font-semibold text-sm">Baseline Score: 0.97</div>
                    </div>

                    <div class="bg-white rounded-2xl p-6 shadow-md border border-yellow-100">
                        <div class="text-3xl mb-2">🟡</div>
                        <h3 class="font-bold text-lg text-gray-800 mb-1">Task 2 — Conceptual Grading</h3>
                        <p class="text-gray-500 text-sm">
                            Grade concept-based answers with partial credit.
                            Graduated rewards based on concept coverage.
                        </p>
                        <div class="mt-3 text-yellow-600 font-semibold text-sm">Baseline Score: 0.88</div>
                    </div>

                    <div class="bg-white rounded-2xl p-6 shadow-md border border-red-100">
                        <div class="text-3xl mb-2">🔴</div>
                        <h3 class="font-bold text-lg text-gray-800 mb-1">Task 3 — Essay Grading</h3>
                        <p class="text-gray-500 text-sm">
                            Grade complex descriptive answers with structured feedback,
                            coherence evaluation, and nuanced scoring.
                        </p>
                        <div class="mt-3 text-red-500 font-semibold text-sm">Baseline Score: 0.77</div>
                    </div>

                </div>

                <div class="bg-white rounded-2xl p-6 shadow-md mb-6">
                    <h2 class="text-xl font-bold text-gray-800 mb-4">🔗 API Endpoints</h2>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm font-mono">
                        <div class="bg-green-50 rounded-lg p-3"><span class="text-green-700 font-bold">POST</span> /reset?task_id=1</div>
                        <div class="bg-blue-50 rounded-lg p-3"><span class="text-blue-700 font-bold">POST</span> /step?task_id=1</div>
                        <div class="bg-purple-50 rounded-lg p-3"><span class="text-purple-700 font-bold">GET</span> /state?task_id=1</div>
                        <div class="bg-gray-50 rounded-lg p-3"><span class="text-gray-700 font-bold">GET</span> /health</div>
                    </div>
                </div>

                <div class="flex justify-center gap-4">
                    <a href="/docs"
                       class="bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-3 px-8 rounded-xl transition shadow-lg">
                        📄 API Docs
                    </a>
                    <a href="https://github.com/natarajannetworks/edueval-grading-env"
                       target="_blank"
                       class="bg-gray-800 hover:bg-black text-white font-semibold py-3 px-8 rounded-xl transition shadow-lg">
                        🔗 GitHub
                    </a>
                    <a href="/health"
                       class="bg-green-600 hover:bg-green-700 text-white font-semibold py-3 px-8 rounded-xl transition shadow-lg">
                        ✅ Health Check
                    </a>
                </div>
            </div>
        </body>
    </html>
    """

@app.post("/reset", response_model=GradingObservation)
def reset(task_id: int = 1):
    if task_id not in environments:
        raise HTTPException(400, "task_id must be 1, 2, or 3")
    environments[task_id] = GradingEnvironment(task_id=task_id)
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
    return {
        "status": "ok",
        "env": "EduEval",
        "version": "1.0.0",
        "tasks": 3,
        "message": "Environment is running successfully"
    }

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)