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
    def test_run_agent_loop_query_content(self, mock_followup):
        # Mock invoke_json to return query_content action
        self.mock_assistant.llm_client.invoke_json.return_value = {
            "thought": "Querying for velocity definition",
            "action": "query_content",
            "params": {"query": "velocity definition"},
            "message": "Searching database..."
        }
        self.mock_assistant.llm_client.invoke_qa.return_value = "Velocity is rate of change of displacement."

        self.agent.run_agent_loop("what is velocity?")
        self.mock_assistant.llm_client.invoke_qa.assert_called_with(self.mock_assistant.spec_qa_chain, "velocity definition")
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
            "params": {"exam_topic": "Higher", "total_marks": 20},
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


if __name__ == '__main__':
    unittest.main()
