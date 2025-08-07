# Neko Agent Justfile

# Start the nix development shell
shell:
    nix develop

# Launch the agent with a default task
agent:
    NEKO_LOGFILE="/Users/luc/projects/neko_agent/tmp/neko1.log" python src/agent.py --task "search for pics of cats"
#    #!/usr/bin/env bash
#    cd "{{justfile_directory()}}"
#    nix develop --command bash -c 'NEKO_LOGFILE="/Users/luc/projects/neko_agent/tmp/neko1.log" python src/agent.py --task "search for pics of cats"'

# Launch the agent with a custom task
agent-task task:
    #!/usr/bin/env bash
    cd "{{justfile_directory()}}"
    nix develop --command bash -c 'NEKO_LOGFILE="/Users/luc/projects/neko_agent/tmp/neko1.log" python src/agent.py --task "{{task}}"'

# Force kill the agent script
kill-agent:
    #!/usr/bin/env bash
    echo "Killing agent processes..."
    ps aux | grep -E "(src/agent|just agent)" | grep -v grep
    pkill -9 -f "python src/agent" || echo "No python agent process found"
    pkill -9 -f "just agent" || echo "No just agent process found"
    pkill -9 -f "src/agent.py" || echo "No agent.py process found"
    sleep 1
    echo "Remaining processes:"
    ps aux | grep -E "(src/agent|just agent)" | grep -v grep || echo "No remaining agent processes"

# Launch the manual control script
manual:
    NEKO_LOGFILE="/Users/luc/projects/neko_agent/tmp/neko_manual.log" python src/manual.py
#    #!/usr/bin/env bash
#    cd "{{justfile_directory()}}"
#    nix develop --command bash -c 'NEKO_LOGFILE="/Users/luc/projects/neko_agent/tmp/neko1.log" python src/manual.py'

# Force kill the manual control script
kill-manual:
    #!/usr/bin/env bash
    echo "Killing manual processes..."
    ps aux | grep -E "(src/manual|just manual)" | grep -v grep
    pkill -9 -f "python src/manual" || echo "No python manual process found"
    pkill -9 -f "just manual" || echo "No just manual process found"
    pkill -9 -f "src/manual.py" || echo "No manual.py process found"
    sleep 1
    echo "Remaining processes:"
    ps aux | grep -E "(src/manual|just manual)" | grep -v grep || echo "No remaining manual processes"

# Force kill both agent and manual control scripts
kill-all:
    #!/usr/bin/env bash
    echo "Killing all neko processes..."
    ps aux | grep -E "(src/agent|src/manual|just agent|just manual)" | grep -v grep
    pkill -9 -f "python src/agent" || echo "No python agent process found"
    pkill -9 -f "just agent" || echo "No just agent process found"
    pkill -9 -f "python src/manual" || echo "No python manual process found"
    pkill -9 -f "just manual" || echo "No just manual process found"
    pkill -9 -f "src/agent.py" || echo "No agent.py process found"
    pkill -9 -f "src/manual.py" || echo "No manual.py process found"
    sleep 1
    echo "Remaining processes:"
    ps aux | grep -E "(src/agent|src/manual|just agent|just manual)" | grep -v grep || echo "No remaining neko processes"

# Launch default browser at NEKO_URL
browser:
    open "$NEKO_URL"

# Create necessary directories
setup:
    mkdir -p tmp/actions

# View the agent log
log:
    tail -f /Users/luc/projects/neko_agent/tmp/neko_agent.log

# Show running neko processes
ps:
    ps aux | grep -E "(src/agent.py|src/manual.py)" | grep -v grep

# Clean up temporary files
clean:
    rm -rf tmp/actions/*
    rm -f tmp/frame.png
    rm -f tmp/neko1.log
    rm -f tmp/*.tmp
