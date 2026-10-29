#!/bin/bash
# Install and start the LaunchAgent that keeps Fajfuzzy pretraining. Remove with:
#   launchctl bootout gui/$(id -u)/fun.gzowo.fajfuzzy.chain && rm ~/Library/LaunchAgents/fun.gzowo.fajfuzzy.chain.plist
set -e
cp "$(dirname "$0")/fun.gzowo.fajfuzzy.chain.plist" ~/Library/LaunchAgents/
rm -f "$(dirname "$0")/STOP"
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/fun.gzowo.fajfuzzy.chain.plist
echo "chain started, log: ~/Library/Logs/fajfuzzy-chain.log"
