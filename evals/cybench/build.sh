#!/usr/bin/bash

# Stop if any command fails.
set -e

# Change to the directory of this script so that relative paths work.
cd "$(dirname "${BASH_SOURCE[0]}")"

# Agent environment.
docker build -t agent-environment:1.0.0 ./agent_execution_environment/

# Challenge environments.
docker build -t avatar-victim:1.0.0 ./challenges/avatar/images/victim
docker build -t back_to_the_past-victim:1.0.0 ./challenges/back_to_the_past/images/victim
docker build -t chunky-blog:1.0.0 ./challenges/chunky/images/blog
docker build -t chunky-rcache:1.0.0 ./challenges/chunky/images/rcache
docker build -t chunky-nginx:1.0.0 ./challenges/chunky/images/nginx
