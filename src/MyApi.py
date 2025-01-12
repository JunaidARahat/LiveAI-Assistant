import enum
from typing import Annotated
from livekit.agents import llm
import logging

logger = logging.getLogger("voice-assistant")
logger.setLevel(logging.INFO)


class Zone(enum.Enum):
    LIVING_ROOM = "living_room"
    BEDROOM = "bedroom"
    KITCHEN = "kitchen"
    BATHROOM = "bathroom"
    OFFICE = "office"


class AssistantFnc(llm.FunctionContext):
    def __init__(self) -> None:
        super().__init__()
        self._temperature = {
            Zone.LIVING_ROOM: 22,
            Zone.BEDROOM: 20,
            Zone.KITCHEN: 24,
            Zone.BATHROOM: 23,
            Zone.OFFICE: 21,
        }
        self._lights = {zone: False for zone in Zone}
        self._reminders = []

    @llm.ai_callable(description="Get the temperature in a specific room.")
    def get_temperature(
        self, zone: Annotated[Zone, llm.TypeInfo(description="The specific zone")]
    ):
        logger.info("Get temperature - zone: %s", zone)
        # Convert string input to Zone enum
        if isinstance(zone, str):
            zone = Zone(zone)
        temp = self._temperature[zone]
        return f"The temperature in the {zone.value.replace('_', ' ')} is {temp}°C."

    @llm.ai_callable(description="Set the temperature in a specific room.")
    def set_temperature(
        self,
        zone: Annotated[Zone, llm.TypeInfo(description="The specific zone")],
        temp: Annotated[int, llm.TypeInfo(description="The temperature to set")],
    ):
        logger.info("Set temperature - zone: %s, temp: %s", zone, temp)
        # Convert string input to Zone enum
        if isinstance(zone, str):
            zone = Zone(zone)
        self._temperature[zone] = temp
        return f"The temperature in the {zone.value.replace('_', ' ')} is now set to {temp}°C."

    @llm.ai_callable(description="Control the lights in a specific room.")
    def control_lights(
        self,
        zone: Annotated[Zone, llm.TypeInfo(description="The specific zone")],
        state: Annotated[str, llm.TypeInfo(description="The state (on/off) to set")],
    ):
        logger.info("Control lights - zone: %s, state: %s", zone, state)
        # Convert string input to Zone enum
        if isinstance(zone, str):
            zone = Zone(zone)
        self._lights[zone] = state.lower() == "on"
        state_text = "on" if self._lights[zone] else "off"
        return f"The lights in the {zone.value.replace('_', ' ')} are now {state_text}."

    @llm.ai_callable(description="Set a reminder for a specific task.")
    def set_reminder(
        self,
        task: Annotated[str, llm.TypeInfo(description="The task to be reminded about")],
    ):
        logger.info("Set reminder - task: %s", task)
        self._reminders.append(task)
        return f"Reminder set: '{task}'."

    @llm.ai_callable(description="Get the current weather for a location.")
    def get_weather(
        self,
        location: Annotated[str, llm.TypeInfo(description="The location for the weather update")],
    ):
        logger.info("Get weather - location: %s", location)
        # Dummy weather data
        weather = f"Foggy, 10°C in {location}"
        return f"The current weather in {location} is: {weather}."
