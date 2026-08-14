import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from chatbot import ChatbotAgent

class TestChatbot(unittest.TestCase):
    @patch('chatbot.GcseAssistant')
    def setUp(self, mock_assistant_cls):
        self.mock_assistant = MagicMock()
        mock_assistant_cls.return_value = self.mock_assistant
        self.agent = ChatbotAgent("Physics", "Edexcel")

    def test_initial_state(self):
        self.assertEqual(self.agent.subject, "Physics")
        self.assertEqual(self.agent.examiner, "Edexcel")
        self.assertEqual(len(self.agent.history), 0)
        self.assertEqual(len(self.agent.preferences), 0)

    def test_preferences_text(self):
        self.assertEqual(self.agent.get_preferences_text(), "No custom preferences set yet.")
        self.agent.preferences.append("Exclude radioactivity")
        self.agent.preferences.append("Limit marks to 20")
        expected = "- Exclude radioactivity\n- Limit marks to 20"
        self.assertEqual(self.agent.get_preferences_text(), expected)

    @patch('chatbot.ChatbotAgent.run_agent_loop_followup')
    def test_run_agent_loop_none(self, mock_followup):
        # Mock invoke_json to return a conversational response (action='none')
        self.mock_assistant.llm_client.invoke_json.return_value = {
            "thought": "Greeting the user",
            "action": "none",
            "params": {},
            "message": "Hello there! How can I help you?"
        }
        
        self.agent.run_agent_loop("Hi")
        self.assertEqual(len(self.agent.history), 2)
        self.assertEqual(self.agent.history[0], {"role": "user", "content": "Hi"})
        self.assertEqual(self.agent.history[1], {"role": "assistant", "content": "Hello there! How can I help you?"})
        mock_followup.assert_not_called()

    @patch('chatbot.ChatbotAgent.run_agent_loop_followup')
    def test_run_agent_loop_query_past_papers(self, mock_followup):
        # Mock invoke_json to return query_past_papers action
        self.mock_assistant.llm_client.invoke_json.return_value = {
            "thought": "Querying for velocity definition",
            "action": "query_past_papers",
            "params": {"query": "velocity definition"},
            "message": "Searching database..."
        }
        self.mock_assistant.llm_client.invoke_qa.return_value = "Velocity is rate of change of displacement."

        self.agent.run_agent_loop("what is velocity?")
        self.mock_assistant.llm_client.invoke_qa.assert_called_with(self.mock_assistant.ms_qa_chain, "velocity definition")
        self.assertEqual(self.agent.history[-1]["role"], "system")
        self.assertIn("Velocity is rate of change of displacement.", self.agent.history[-1]["content"])
        mock_followup.assert_called_once()

    @patch('chatbot.ChatbotAgent.run_agent_loop_followup')
    def test_run_agent_loop_save_preferences(self, mock_followup):
        # Mock invoke_json to return save_preferences action
        self.mock_assistant.llm_client.invoke_json.return_value = {
            "thought": "Saving user instruction",
            "action": "save_preferences",
            "params": {"preferences": "Ignore radioactivity"},
            "message": "Saving rule..."
        }

        self.agent.run_agent_loop("Never generate radioactivity questions")
        self.assertIn("Ignore radioactivity", self.agent.preferences)
        mock_followup.assert_called_once()

    @patch('chatbot.ChatbotAgent.run_agent_loop_followup')
    def test_generate_exam_validation_failure(self, mock_followup):
        # Mock invoke_json to return generate_exam with missing topics
        self.mock_assistant.llm_client.invoke_json.return_value = {
            "thought": "Generating exam but missing topics",
            "action": "generate_exam",
            "params": {"exam_type": "Higher", "total_marks": 20},
            "message": "Generating..."
        }

        self.agent.run_agent_loop("generate a higher exam for 20 marks")
        # Check that error was recorded in system role in history
        self.assertEqual(self.agent.history[-1]["role"], "system")
        self.assertIn("Validation Error: Cannot generate exam. Missing required parameter(s): topics.", self.agent.history[-1]["content"])
        mock_followup.assert_called_once()
        self.mock_assistant.make_exam.assert_not_called()

    @patch('chatbot.ChatbotAgent.run_agent_loop_followup')
    def test_mark_answer_validation_failure(self, mock_followup):
        # Mock invoke_json to return mark_answer with missing student_answer
        self.mock_assistant.llm_client.invoke_json.return_value = {
            "thought": "Marking answer but missing student_answer",
            "action": "mark_answer",
            "params": {"question": "What is speed?", "marks": 2, "mark_scheme": "speed=distance/time"},
            "message": "Marking..."
        }

        self.agent.run_agent_loop("mark my answer for speed question")
        # Check that error was recorded
        self.assertEqual(self.agent.history[-1]["role"], "system")
        self.assertIn("Validation Error: Cannot mark answer. Missing required parameter(s): student_answer.", self.agent.history[-1]["content"])
        mock_followup.assert_called_once()
        self.mock_assistant.exam_marker.mark_answer.assert_not_called()

    @patch('chatbot.ChatbotAgent.run_agent_loop_followup')
    def test_single_item_actions(self, mock_followup):
        # 1. Single question generation
        self.mock_assistant.generate_single_question.return_value = {
            "number": "1)", "text": "Explain photosynthesis.", "marks": 4, "subtopic": "Plants"
        }
        self.mock_assistant.llm_client.invoke_json.return_value = {
            "thought": "Generate single question",
            "action": "generate_single_question",
            "params": {"topic": "Photosynthesis", "marks": 4}
        }
        self.agent.run_agent_loop("Give me a 4 mark question on photosynthesis")
        self.mock_assistant.generate_single_question.assert_called_once()
        self.assertEqual(self.agent.history[-1]["role"], "system")
        self.assertIn("Single question generated", self.agent.history[-1]["content"])

        # 3. Mark scheme generation
        self.mock_assistant.generate_mark_scheme.return_value = "1 mark for light, 1 mark for chlorophyll"
        self.mock_assistant.llm_client.invoke_json.return_value = {
            "thought": "Generate mark scheme",
            "action": "generate_mark_scheme",
            "params": {"topic": "Photosynthesis", "question": "Explain photosynthesis", "marks": 4}
        }
        self.agent.run_agent_loop("Give me the mark scheme")
        self.mock_assistant.generate_mark_scheme.assert_called_once()

        # 4. Model answer generation
        self.mock_assistant.generate_model_answer.return_value = "Light intensity increases the rate of photosynthesis..."
        self.mock_assistant.llm_client.invoke_json.return_value = {
            "thought": "Generate model answer",
            "action": "generate_model_answer",
            "params": {"question": "Explain photosynthesis", "marks": 4}
        }
        self.agent.run_agent_loop("Write a model answer")
        self.mock_assistant.generate_model_answer.assert_called_once()

        # 6. Spec breakdown
        self.mock_assistant.get_spec_breakdown.return_value = {
            "topic": "Photosynthesis", "spec_code": "4.1.2", "subtopics": ["Light intensity", "Chlorophyll"]
        }
        self.mock_assistant.llm_client.invoke_json.return_value = {
            "thought": "Get spec breakdown",
            "action": "get_spec_breakdown",
            "params": {"topic": "Photosynthesis"}
        }
        self.agent.run_agent_loop("Show spec codes")
        self.mock_assistant.get_spec_breakdown.assert_called_once()

        # 8. Command word explanation
        self.mock_assistant.explain_command_word.return_value = "Explain requires giving reasons..."
        self.mock_assistant.llm_client.invoke_json.return_value = {
            "thought": "Explain command word",
            "action": "explain_command_word",
            "params": {"command_word": "Explain"}
        }
        self.agent.run_agent_loop("Explain the command word Explain")
        self.mock_assistant.explain_command_word.assert_called_once()

    @patch('chatbot.ChatbotAgent.run_agent_loop_followup')
    def test_multi_action_complex_query(self, mock_followup):
        self.mock_assistant.generate_single_question.return_value = {
            "number": "1)", "text": "Describe Newton's laws.", "marks": 4
        }
        self.mock_assistant.generate_mark_scheme.return_value = "1 mark per law"
        self.mock_assistant.get_spec_breakdown.return_value = {"topic": "Forces", "spec_code": "P2.1", "subtopics": ["Newton"]}

        self.mock_assistant.llm_client.invoke_json.return_value = {
            "thought": "Complex query combining question, mark scheme, and spec breakdown",
            "action": "generate_single_question",
            "actions": [
                {"action": "generate_single_question", "params": {"topic": "Forces", "marks": 4}},
                {"action": "generate_mark_scheme", "params": {"topic": "Forces", "marks": 4}},
                {"action": "get_spec_breakdown", "params": {"topic": "Forces"}}
            ]
        }

        self.agent.run_agent_loop("Give me a question, its mark scheme, and the spec code for Forces")
        self.mock_assistant.generate_single_question.assert_called_once()
        self.mock_assistant.generate_mark_scheme.assert_called_once()
        self.mock_assistant.get_spec_breakdown.assert_called_once()
        mock_followup.assert_called_once()

    @patch('chatbot.ChatbotAgent.run_agent_loop_followup')
    def test_exam_generation_isolation(self, mock_followup):
        self.agent.valid_exam_types = ["Higher"]
        self.mock_assistant.llm_client.invoke_json.return_value = {
            "thought": "Generate exam request with stray secondary actions",
            "action": "generate_exam",
            "actions": [
                {"action": "generate_exam", "params": {"exam_type": "Higher", "total_marks": 20, "topics": ["Motion"]}},
                {"action": "generate_single_question", "params": {"topic": "Motion"}}
            ],
            "params": {"exam_type": "Higher", "total_marks": 20, "topics": ["Motion"]}
        }
        self.mock_assistant.make_exam.return_value = {"questions": {}, "spec_trees": {}}

        self.agent.run_agent_loop("Generate a higher exam for 20 marks on Motion")
        self.mock_assistant.make_exam.assert_called_once()
        self.mock_assistant.generate_single_question.assert_not_called()


if __name__ == '__main__':
    unittest.main()

