# Manual Control Guide

The Manual Control CLI is an interactive tool that lets you remotely control any Neko server through your terminal. Think of it as a command-line remote desktop that you can use for testing, administration, or automating tasks on hosted desktop environments.

## Quick Start

### Prerequisites

- Python 3.8 or newer
- Access to a Neko server (either local or hosted)
- Network connectivity to the Neko server

### Installation

The manual control tool is included with the Neko Agent package:

```bash
# If you have the project locally
cd /path/to/neko_agent_private
python src/manual.py --help

# Or if installed as a package
neko-manual --help
```

### First Connection

The easiest way to connect is using your Neko server credentials:

```bash
python src/manual.py \
  --neko-url "https://your-neko-server.com" \
  --username "your-username" \
  --password "your-password"
```

If successful, you'll see:
```text
Starting Neko manual CLI. Type 'help' for commands. Ctrl+D or 'quit' to exit.
neko> 
```

## Basic Usage

### Getting Help

Type `help` at any time to see all available commands:

```text
neko> help
```

This displays a comprehensive list of all commands with their syntax.

### Essential Commands

#### Mouse Control
```bash
# Move mouse cursor to specific coordinates
neko> move 100 200

# Click at current mouse position
neko> click

# Move and click in one command
neko> tap 300 400

# Right-click
neko> rclick

# Double-click
neko> dblclick
```

#### Keyboard Input
```bash
# Type text
neko> text "Hello, World!"

# Press specific keys
neko> key Enter
neko> key Escape
neko> key F5

# Click somewhere and then type
neko> input 500 300 "username@example.com"
```

#### Navigation
```bash
# Scroll in different directions
neko> scroll down 3
neko> scroll up
neko> scroll left 2

# Drag/swipe gestures
neko> swipe 100 100 300 300
```

### Coordinate Systems

By default, the tool uses pixel coordinates based on your screen resolution. You can also use normalized coordinates (0.0 to 1.0) for resolution-independent control:

```bash
# Start with normalized coordinates
python src/manual.py --norm \
  --neko-url "https://your-server.com" \
  --username "admin" --password "secret"

# Now coordinates are between 0 and 1
neko> move 0.5 0.5    # Center of screen
neko> tap 0.1 0.9     # Bottom-left area
```

## Common Use Cases

### Website Testing

Test web applications by automating browser interactions:

```bash
# Navigate to a website
neko> tap 400 60              # Click address bar
neko> text "https://example.com"
neko> key Enter

# Fill out a form
neko> tap 300 200             # Click username field
neko> text "testuser"
neko> key Tab                 # Move to next field
neko> text "password123"
neko> key Enter               # Submit form

# Test navigation
neko> tap 500 300             # Click a link
neko> key F5                  # Refresh page
neko> key Alt+Left            # Go back
```

### Application Testing

Test desktop applications:

```bash
# Open application menu
neko> key Super               # Windows/Super key
neko> text "calculator"
neko> key Enter

# Use the application
neko> tap 200 300             # Click number 5
neko> tap 250 350             # Click plus
neko> tap 200 300             # Click number 5
neko> tap 300 400             # Click equals
```

### System Administration

Perform administrative tasks:

```bash
# Open terminal
neko> key Ctrl+Alt+t

# Run system commands
neko> text "sudo apt update"
neko> key Enter
neko> text "your-password"    # If prompted
neko> key Enter

# Navigate file system
neko> text "cd /var/log"
neko> key Enter
neko> text "ls -la"
neko> key Enter
```

### File Management

Work with files and folders:

```bash
# Open file manager
neko> key Super
neko> text "files"
neko> key Enter

# Navigate and create files
neko> key Ctrl+n              # New folder
neko> text "TestFolder"
neko> key Enter
neko> dblclick                # Enter folder
neko> rclick                  # Right-click for context menu
neko> text "n"                # New file (if available)
```

## Advanced Features

### Clipboard Operations

```bash
# Copy, cut, and paste
neko> select_all              # Select all text
neko> copy                    # Copy selection
neko> tap 400 500             # Click elsewhere
neko> paste                   # Paste content

# Paste specific text
neko> paste "This is custom text to paste"
```

### Session Management

If you have admin rights, you can manage other users:

```bash
# List all connected users
neko> sessions

# Force take control from another user
neko> force-take

# Kick a specific user (use session ID from 'sessions' command)
neko> kick abc123-def456-789

# Release control
neko> force-release
```

### Screen Size Management

```bash
# Check current screen size
neko> size

# Set virtual screen size (affects coordinate scaling)
neko> size 1920x1080
```

### Raw Protocol Access

For advanced users, send custom commands:

```bash
# Send custom JSON messages to the server
neko> raw '{"event":"system/stats"}'
neko> raw '{"event":"control/scroll","payload":{"delta_x":0,"delta_y":240}}'
```

## Configuration

### Environment Variables

Set these to avoid typing credentials each time:

```bash
export NEKO_URL="https://your-neko-server.com"
export NEKO_USER="your-username"  
export NEKO_PASS="your-password"
export NEKO_SIZE="1920x1080"

# Now you can just run:
python src/manual.py
```

### Command Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `--neko-url` | Neko server URL | `https://neko.example.com` |
| `--username` | Login username | `admin` |
| `--password` | Login password | `secretpass` |
| `--norm` | Use 0-1 coordinates | (flag only) |
| `--size` | Virtual screen size | `1920x1080` |
| `--no-auto-host` | Don't auto-request control | (flag only) |
| `--no-media` | Skip video/audio setup | (flag only) |
| `--no-audio` | Disable audio | (flag only) |

### Logging

Enable detailed logging for troubleshooting:

```bash
export NEKO_LOGLEVEL="DEBUG"
export NEKO_LOGFILE="/tmp/neko-manual.log"
python src/manual.py --neko-url "..." --username "..." --password "..."

# In another terminal, watch the logs:
tail -f /tmp/neko-manual.log
```

## Tips and Best Practices

### Timing and Delays

For automation scripts, you may need to add delays between actions:

```bash
# The tool doesn't have built-in delays, but you can use shell scripting:
echo -e "tap 300 200\ntext hello\nkey Enter" | python src/manual.py --neko-url "..." --username "..." --password "..."
```

### Screen Resolution

Always check the screen size when starting:

```bash
neko> size
size 1920x1080  normalized=false
```

This helps you understand the coordinate system you're working with.

### Connection Issues

If you have connection problems:

1. **Check credentials**: Make sure username/password are correct
2. **Verify URL**: Ensure the Neko server URL is accessible
3. **Network connectivity**: Test basic HTTP access to the server
4. **Enable debug logging**: Use `NEKO_LOGLEVEL=DEBUG` for detailed logs

### Multiple Users

Be aware that Neko servers can have multiple users connected:

- Only one user can have "host" control at a time
- Use `sessions` command to see who else is connected
- Use `host` and `unhost` to request/release control
- Admin users can use `force-take` and `force-release`

### Automation Scripts

For repetitive tasks, consider creating shell scripts:

```bash
#!/bin/bash
# test-website.sh

export NEKO_URL="https://neko.example.com"
export NEKO_USER="admin"
export NEKO_PASS="password"

{
    echo "tap 400 60"                    # Click address bar
    echo "text https://example.com"      # Type URL
    echo "key Enter"                     # Navigate
    sleep 3                             # Wait for page load
    echo "tap 300 200"                  # Click login button
    echo "text testuser"                # Username
    echo "key Tab"                      # Next field
    echo "text testpass"                # Password  
    echo "key Enter"                    # Submit
} | python src/manual.py
```

## Troubleshooting

### Common Issues

**"Connection failed"**
- Verify the Neko server is running and accessible
- Check that the URL, username, and password are correct
- Ensure no firewall is blocking the connection

**"Command not found: manual.py"**
- Make sure you're in the correct directory
- Use the full path: `python /path/to/src/manual.py`

**"No host control"**
- Another user may have control; use `sessions` to check
- Try `host` command to request control
- Admin users can use `force-take`

**Commands not working**
- Verify you have host control
- Check if coordinates are correct for the screen size
- Use `size` command to verify resolution

**High latency**
- Network delay between you and the Neko server
- Try reducing the number of rapid commands
- Consider using the server closer to your location

### Getting Help

If you encounter issues:

1. Enable debug logging with `NEKO_LOGLEVEL=DEBUG`
2. Check the log output for error messages
3. Verify basic connectivity to the Neko server
4. Review the [Developer Guide](../developer-guide/components/manual.md) for technical details

The Manual Control CLI is a powerful tool for remote desktop automation and testing. With practice, you can efficiently control any Neko server environment from your terminal.