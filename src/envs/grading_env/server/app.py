from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

# ✅ Correct relative imports
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

# Initialize environments
environments = {
    1: GradingEnvironment(task_id=1),
    2: GradingEnvironment(task_id=2),
    3: GradingEnvironment(task_id=3),
}

# -------------------- ROOT UI --------------------
@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
        <head>
            <title>EduEval Grading Environment</title>
            <script src="https://cdn.tailwindcss.com"></script>
        </head>
        <body class="bg-gray-50 flex items-center justify-center min-h-screen">
            <div class="max-w-3xl w-full p-8 text-center text-gray-800">
                <h1 class="text-5xl font-extrabold mb-4 text-indigo-600">EduEval</h1>
                <p class="text-lg text-gray-600 mb-10">
                    OpenEnv environment for intelligent answer sheet grading and evaluation
                </p>
                <div class="bg-white rounded-2xl p-8 text-left shadow-md border border-gray-200">
                    <h2 class="text-2xl font-bold mb-6 text-gray-800">Tasks</h2>
                    <ul class="space-y-5">
                        <li><span class="font-semibold text-indigo-500">Easy Level</span><br/>Basic answer grading using simple evaluation rules.</li>
                        <li><span class="font-semibold text-indigo-500">Medium Level</span><br/>Multi-step answer validation and scoring logic.</li>
                        <li><span class="font-semibold text-indigo-500">Hard Level</span><br/>Complex descriptive answer grading with detailed feedback.</li>
                    </ul>
                </div>
                <div class="flex justify-center gap-6 mt-10">
                    <a href="/docs" 
                       class="bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-3 px-8 rounded-xl transition shadow">
                        📄 API Docs
                    </a>
                    <a href="https://github.com/YOUR_USERNAME/YOUR_REPO" 
                       target="_blank"
                       class="bg-gray-800 hover:bg-black text-white font-semibold py-3 px-8 rounded-xl transition shadow">
                        🔗 GitHub
                    </a>
                </div>
            </div>
        </body>
    </html>
    """

# -------------------- API --------------------
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
    return {
        "status": "ok",
        "env": "EduEval",
        "message": "Environment is running successfully"
    }
def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)