from Utils.agent import MedicalAgent, VALID_ROLES


class MedicalSession:
    """
    Manages a full multi-specialist session.

    Typical flows:
        1. Single specialist  — create one agent, call chat() in a loop
        2. Panel review       — run_panel() sends the same message to all specialists,
                                then pipes their reports into the MDT agent
    """

    def __init__(self):
        self._agents: dict[str, MedicalAgent] = {}

    def get_agent(self, role: str) -> MedicalAgent:
        """Lazily creates and caches agents by role."""
        if role not in self._agents:
            self._agents[role] = MedicalAgent(role=role)
        return self._agents[role]

    def chat(self, role: str, message: str) -> dict:
        """Send a message to a specific specialist agent."""
        return self.get_agent(role).chat(message)

    def run_panel(self, patient_report: str) -> dict:
        """
        Full pipeline:
            1. Send the patient report to each specialist
            2. Combine their responses into a single MDT prompt
            3. Return the MDT synthesis

        Returns:
            {
                "specialist_reports": { "Cardiologist": "...", ... },
                "mdt_synthesis": "...",
                "success": bool,
            }
        """
        specialists = ["Cardiologist", "Psychologist", "Pulmonologist"]
        specialist_reports = {}

        for role in specialists:
            result = self.chat(role, patient_report)
            if not result["success"]:
                return {
                    "specialist_reports": specialist_reports,
                    "mdt_synthesis": None,
                    "success": False,
                    "error": f"{role} failed: {result['error']}",
                }
            specialist_reports[role] = result["response"]

        # Build a combined prompt for the MDT agent
        mdt_prompt = (
            f"Cardiologist Report:\n{specialist_reports['Cardiologist']}\n\n"
            f"Psychologist Report:\n{specialist_reports['Psychologist']}\n\n"
            f"Pulmonologist Report:\n{specialist_reports['Pulmonologist']}"
        )

        mdt_result = self.chat("MultidisciplinaryTeam", mdt_prompt)
        if not mdt_result["success"]:
            return {
                "specialist_reports": specialist_reports,
                "mdt_synthesis": None,
                "success": False,
                "error": f"MDT failed: {mdt_result['error']}",
            }

        return {
            "specialist_reports": specialist_reports,
            "mdt_synthesis": mdt_result["response"],
            "success": True,
        }

    def reset_all(self):
        """Clear history for every agent in the session."""
        for agent in self._agents.values():
            agent.reset()

    def reset_agent(self, role: str):
        """Clear history for a single agent."""
        if role in self._agents:
            self._agents[role].reset()

    @property
    def available_roles(self) -> list[str]:
        return VALID_ROLES