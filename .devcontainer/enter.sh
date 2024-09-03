#!/bin/bash
docker exec -it autoware-ml /usr/sbin/gosu autoware bash -c 'cd /workspace && exec bash'
