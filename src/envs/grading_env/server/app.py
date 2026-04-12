from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from envs.grading_env.models import GradingAction, GradingObservation, GradingState
from .environment import GradingEnvironment

app = FastAPI(
    title="EduEval Grading Environment",
    description="OpenEnv environment for automated answer sheet grading",
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
        <title>EduEval</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>

    <body class="bg-gradient-to-br from-indigo-50 to-blue-100 min-h-screen flex items-center justify-center p-6">
        <div class="max-w-4xl w-full">

            <!-- HEADER -->
            <div class="text-center mb-10">
                <h1 class="text-6xl font-extrabold text-indigo-700 mb-3">📚 EduEval</h1>
                <p class="text-xl text-gray-600">AI Answer Sheet Grading System</p>
            </div>

            <!-- TASK CARDS -->
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">

                <div class="bg-white p-5 rounded-xl shadow">
                    <h3 class="font-bold text-green-600">Task 1 — Factual</h3>
                    <p class="text-sm text-gray-500">Binary grading</p>
                </div>

                <div class="bg-white p-5 rounded-xl shadow">
                    <h3 class="font-bold text-yellow-600">Task 2 — Conceptual</h3>
                    <p class="text-sm text-gray-500">Partial credit</p>
                </div>

                <div class="bg-white p-5 rounded-xl shadow">
                    <h3 class="font-bold text-red-600">Task 3 — Essay</h3>
                    <p class="text-sm text-gray-500">Detailed evaluation</p>
                </div>

            </div>

            <!-- BUTTONS -->
            <div class="flex justify-center gap-4 mb-6">
                <button onclick="startTask(1)" class="bg-green-600 text-white px-5 py-2 rounded-lg">Task 1</button>
                <button onclick="startTask(2)" class="bg-yellow-600 text-white px-5 py-2 rounded-lg">Task 2</button>
                <button onclick="startTask(3)" class="bg-red-600 text-white px-5 py-2 rounded-lg">Task 3</button>
            </div>

            <!-- ANSWER BOX -->
            <div class="flex flex-col items-center gap-3">
                <textarea id="answer" placeholder="Enter your answer..."
                    class="w-full max-w-xl p-3 border rounded-lg"></textarea>

                <button onclick="submitAnswer()"
                    class="bg-indigo-600 text-white px-6 py-2 rounded-lg">
                    Submit Answer
                </button>

                <p id="result" class="text-gray-700 text-sm text-center"></p>
            </div>

        </div>

        <!-- SCRIPT -->
        <script>
            let currentTask = 1;

            async function startTask(taskId) {
                currentTask = taskId;

                const res = await fetch(`/reset?task_id=${taskId}`, {
                    method: "POST"
                });
                const data = await res.json();

                // ✅ FIXED: correct path to question
                const question = data.question || data.observation?.question || "No question found";

                document.getElementById("result").innerText =
                    "Question: " + question;
            }

            async function submitAnswer() {
                const answer = document.getElementById("answer").value;

                const res = await fetch(`/step?task_id=${currentTask}`, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        answer: answer
                    })
                });

                const data = await res.json();

                // ✅ FIXED: include feedback if available
                const feedback = data.observation?.feedback || "";

                document.getElementById("result").innerText =
                    "Score: " + data.reward + 
                    " | Completed: " + data.done + 
                    (feedback ? " | Feedback: " + feedback : "");
            }
        </script>

    </body>
    </html>
    """

@app.post("/reset")
def reset(task_id: int = 1):
    if task_id not in environments:
        raise HTTPException(400, "Invalid task_id")

    environments[task_id] = GradingEnvironment(task_id=task_id)
    obs = environments[task_id].reset()

    # 🔍 Try all possible fields
    question = None

    if hasattr(obs, "question"):
        question = obs.question
    elif hasattr(obs, "current_question"):
        question = obs.current_question
    elif isinstance(obs, dict):
        question = obs.get("question")

    return {
        "question": question if question else "⚠️ Question not found",
        "observation": obs.dict() if hasattr(obs, "dict") else obs
    }

@app.post("/step")
def step(request: dict, task_id: int = 1):
    if task_id not in environments:
        raise HTTPException(400, "Invalid task_id")

    env = environments[task_id]

    if env.state().is_complete:
        raise HTTPException(400, "Episode complete")

    # ✅ Extract answer from frontend
    answer = request.get("answer", "")

    # ✅ Create action object correctly
    action = GradingAction(answer=answer)

    obs, reward, done = env.step(action)

    return {
        "observation": obs.dict(),
        "reward": reward,
        "done": done
    }

@app.get("/state", response_model=GradingState)
def state(task_id: int = 1):
    if task_id not in environments:
        raise HTTPException(400, "Invalid task_id")
    return environments[task_id].state()

@app.get("/health")
def health():
    return {
        "status": "ok",
        "env": "EduEval",
        "tasks": 3
    }

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)