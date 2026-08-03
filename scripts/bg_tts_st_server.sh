#!/usr/bin/env bash
#
# Start (or restart) the persistent bg-tts-st TTS server on port 8002.
#
# bg-tts-st is the Ani-Voice-API two-stage Bulgarian TTS (Supertonic reference
# generator -> BgTTS synthesizer). The server keeps both models resident across
# requests so the tts-router doesn't reload them on every mixed BG/EN prompt.
# The models auto-unload after 5 minutes idle and reload lazily on the next
# request.
#
# Usage:
#   ~/.hermes/scripts/bg_tts_st_server.sh [start|stop|restart|status]
#
set -euo pipefail

SERVER_PY="$HOME/.hermes/scripts/bg_tts_st_server.py"
PYTHON="$HOME/.hermes/venvs/ani-voice/bin/python"
PORT=8002
IDLE=300
VOICE="M5"
LOG="$HOME/.hermes/logs/bg-tts-st-server.log"
PIDFILE="$HOME/.hermes/logs/bg-tts-st-server.pid"

mkdir -p "$HOME/.hermes/logs"

is_running() {
    [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null
}

start() {
    if is_running; then
        echo "bg-tts-st server already running (pid $(cat "$PIDFILE"))"
        return 0
    fi
    if [ ! -x "$SERVER_PY" ] || [ ! -x "$PYTHON" ]; then
        echo "ERROR: server script or venv python missing."
        echo "  server: $SERVER_PY"
        echo "  python: $PYTHON"
        exit 1
    fi
    echo "Starting bg-tts-st server on port $PORT (voice=$VOICE, idle unload ${IDLE}s)..."
    # Launch with cleared PYTHONPATH so the dedicated venv stays self-contained.
    env -u PYTHONPATH -u PYTHONHOME nohup "$PYTHON" "$SERVER_PY" \
        --port "$PORT" --idle-timeout "$IDLE" --voice-style "$VOICE" \
        >>"$LOG" 2>&1 &
    echo $! > "$PIDFILE"
    sleep 1
    echo "Started (pid $(cat "$PIDFILE")). Log: $LOG"
}

stop() {
    if is_running; then
        kill "$(cat "$PIDFILE")" 2>/dev/null || true
        rm -f "$PIDFILE"
        echo "Stopped."
    else
        echo "Not running."
    fi
}

status() {
    if is_running; then
        echo "bg-tts-st server running (pid $(cat "$PIDFILE"))"
        curl -s http://127.0.0.1:$PORT/health || true
        echo ""
    else
        echo "bg-tts-st server not running."
    fi
}

case "${1:-start}" in
    start)   start ;;
    stop)    stop ;;
    restart) stop; start ;;
    status)  status ;;
    *) echo "Usage: $0 [start|stop|restart|status]"; exit 1 ;;
esac