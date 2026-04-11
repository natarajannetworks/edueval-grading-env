import sys
sys.path.insert(0, 'src')
from envs.grading_env.server.environment import GradingEnvironment
env = GradingEnvironment(task_id=1)
obs = env.reset()
print('concept_coverage:', obs.concept_coverage)
print('question:', obs.question_text)
print('summary:', obs.answer_summary)