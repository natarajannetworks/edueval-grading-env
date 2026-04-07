# EduEval Client (OpenEnv Compatible)

from typing import Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from src.envs.grading_env.models import GradingAction, GradingObservation


class GradingEnvClient(EnvClient[GradingAction, GradingObservation, State]):

    # 🔹 What we send to server
    def _step_payload(self, action: GradingAction) -> Dict:
        return {
            "marks_awarded": action.marks_awarded,
        }

    # 🔹 How we read server response
    def _parse_result(self, payload: Dict) -> StepResult[GradingObservation]:
        obs_data = payload.get("observation", {})

        observation = GradingObservation(
            question_text=obs_data.get("question_text", ""),
            student_answer=obs_data.get("student_answer", ""),
            answer_summary=obs_data.get("answer_summary", ""),
            answer_key=obs_data.get("answer_key", ""),
            semantic_similarity=obs_data.get("semantic_similarity", 0.0),
            concept_coverage=obs_data.get("concept_coverage", 0.0),
            question_number=obs_data.get("question_number", 0),
            total_questions=obs_data.get("total_questions", 0),
            max_marks=obs_data.get("max_marks", 1.0),
            task_id=obs_data.get("task_id", 1),
            done=payload.get("done", False),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    # 🔹 State parser
    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )


# 🔥 MAIN EXECUTION (This is your intelligent grading logic)
if __name__ == "__main__":
    with GradingEnvClient(base_url="http://localhost:8000") as client:

        for task_id in [1, 2, 3]:
            print(f"\n===== STARTING TASK {task_id} =====")

            result = client.reset(task_id=task_id)
            obs = result.observation

            total_reward = 0.0
            done = False

            while not done:
                print("\nQuestion:", obs.question_text)
                print("Student Answer:", obs.student_answer)

                # ✅ SMART GRADING LOGIC
                similarity = obs.semantic_similarity
                coverage = obs.concept_coverage

                # Combine both (core intelligence)
                marks = (similarity + coverage) / 2

                # Clamp between 0 and 1
                marks = max(0.0, min(1.0, marks))

                # Create action
                action = GradingAction(marks_awarded=marks)

                # Step
                result = client.step(action)
                obs = result.observation
                reward = result.reward
                done = result.done

                total_reward += reward

                print(f"Marks Given: {round(marks, 2)}")
                print(f"Reward: {reward}")

            print(f"\n✅ Total Reward for Task {task_id}: {round(total_reward, 2)}")