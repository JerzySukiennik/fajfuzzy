#!/bin/bash
# Install and start the LaunchAgent that keeps G-Decide pretraining. Remove with:
#   launchctl bootout gui/$(id -u)/fun.gzowo.g-decide.chain && rm ~/Library/LaunchAgents/fun.gzowo.g-decide.chain.plist
set -e
cp "$(dirname "$0")/fun.gzowo.g-decide.chain.plist" ~/Library/LaunchAgents/
rm -f "$(dirname "$0")/STOP"
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/fun.gzowo.g-decide.chain.plist
echo "chain started, log: ~/Library/Logs/g-decide-chain.log"
