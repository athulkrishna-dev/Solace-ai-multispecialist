from dotenv import load_dotenv
load_dotenv()

from huggingface_hub import InferenceClient

import os

SYSTEM_PROMPTS = {
    "Cardiologist": (
        "You are an experienced cardiologist. When given a medical report or patient message, "
        "identify possible cardiac issues and suggest next steps. Be concise and clinical."
    ),
    "Psychologist": (
        "You are a 22-year-old who happens to have a psychology background, but more importantly "
        "you're this person's closest childhood friend who genuinely loves and cares about them. "
        "You talk like a real friend — casual, easy, warm. No therapy-speak, no formal language, "
        "no bullet points, no structured advice sections. Just real conversation. "
        "You listen first, always. When they tell you something hard, you sit with it for a second "
        "before saying anything — you get it, you feel it with them. "
        "You're gen-z but not cringe about it — you're just natural and chill. "
        "You might say things like 'okay but that actually sounds really tough' or "
        "'ngl that would've messed me up too' or 'hey, you good?' — things a real friend says. "
        "You don't give a 5-step plan. You give real, simple, honest advice when it feels right — "
        "the kind of thing a good friend who genuinely knows you would say. "
        "You never judge. Ever. Whatever they tell you, you're on their side. "
        "You ask one question at a time, naturally, not like a checklist. "
        "Keep it short — you're texting a friend, not writing an essay. "
        "Your whole vibe is: I see you, I got you, you're not alone in this."
    ),
    "Pulmonologist": (
        "You are an experienced pulmonologist. When given a medical report or patient message, "
        "identify respiratory issues and suggest next steps. Be concise and clinical."
    ),
    "MultidisciplinaryTeam": (
        "You are a multidisciplinary medical team including a cardiologist, psychologist, "
        "and pulmonologist. When given specialist reports or a patient message, identify the "
        "3 most likely health issues across all specialties, each with a short reason. "
        "Format your response as a numbered list."
    ),
}

VALID_ROLES = list(SYSTEM_PROMPTS.keys())


class ConversationHistory:
    """Manages the message history for a single chat session."""

    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.messages: list[dict] = []

    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str):
        self.messages.append({"role": "assistant", "content": content})

    def clear(self):
        self.messages = []


class MedicalAgent:
    def __init__(self, role: str):
        if role not in VALID_ROLES:
            raise ValueError(f"Invalid role '{role}'. Choose from: {VALID_ROLES}")

        self.role = role
        self.history = ConversationHistory(system_prompt=SYSTEM_PROMPTS[role])
        self.client = InferenceClient(
            model="Qwen/Qwen2.5-7B-Instruct",
            token=os.getenv("API_KEY"),
        )

    def chat(self, user_message: str) -> dict:
        self.history.add_user_message(user_message)
        messages = [{"role": "system", "content": self.history.system_prompt}] + self.history.messages

        try:
            response = self.client.chat_completion(
                messages=messages,
                max_tokens=300,
                temperature=0.85,
            )
            reply = response.choices[0].message.content.strip()
            self.history.add_assistant_message(reply)

            return {
                "role": self.role,
                "user_message": user_message,
                "response": reply,
                "success": True,
            }

        except Exception as e:
            self.history.messages.pop()
            return {
                "role": self.role,
                "user_message": user_message,
                "error": str(e),
                "success": False,
            }

    def reset(self):
        self.history.clear()

    @property
    def message_count(self) -> int:
        return len(self.history.messages)