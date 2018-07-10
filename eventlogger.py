from datetime import datetime

import pandas as pd

class EventLogger:
    event_id = 0
    def __init__(self):
        self.starttime = datetime.now()
        self.eventsMessage = {}
        self.eventsTime = {}

    def addEvent(self, event_message, event_code):
        EventLogger.event_id += 1
        self.eventsTime[EventLogger.event_id] = datetime.now()
        self.eventsMessage[EventLogger.event_id] = {event_code: event_message}


if __name__ == "__main__":
    pass