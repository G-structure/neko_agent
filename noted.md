Notes
- [ ] Scroll up / down is reversed.
- [ ] manual.py somtimes disconnects and does not reconnect forcing the user to quit the app
- help command does not work in manual_refactored.py

- [ ] Go through an dedup code...
  - [ ] remote_agent.py and showui_agent.py (maybe gpu_agent.py)

for agent_refactored.py we for some reason always think the screen is 1920x1080

for agent_refactored.py when the agent has finished a task in online mode (--online) and it is waiting for a new task it should relase the controls so that anyone can now use the neko browser controls again. This should be how agent.py works so if you need a refrence look to there.
