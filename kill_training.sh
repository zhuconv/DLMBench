#!/bin/bash
# Script to kill training processes on nodes from the last gh-dev job

# Get the last gh-dev running job's node list
NODELIST=$(squeue -u $USER -p gh-dev -t R -h -o "%N" | tail -1)

if [ -z "$NODELIST" ]; then
    echo "No running gh-dev jobs found!"
    echo "Trying to find processes on any nodes..."
    # If no job found, try to kill processes on current node
    echo "Killing training processes on current node..."
    pkill -f "torchrun.*train.py" || echo "No torchrun processes found"
    pkill -f "train.py" || echo "No train.py processes found"
    exit 0
fi

echo "Found node list: $NODELIST"

# Expand the node list
NODES=$(scontrol show hostnames "$NODELIST")

if [ -z "$NODES" ]; then
    echo "Failed to expand node list!"
    exit 1
fi

# Convert to array
NODE_ARRAY=($NODES)
NNODES=${#NODE_ARRAY[@]}

echo "Expanded to $NNODES nodes:"
echo "$NODES"
echo ""

# Ask for confirmation
read -p "Kill all training processes on these nodes? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Kill processes on each node
echo "Killing training processes on all nodes..."

for NODE in "${NODE_ARRAY[@]}"; do
    echo "Killing on node $NODE..."
    
    ssh $NODE "pkill -f 'torchrun.*train.py' 2>/dev/null; \
               pkill -f 'train.py' 2>/dev/null; \
               pkill -f 'python.*train.py' 2>/dev/null" || true
    
    # Check if processes are still running
    REMAINING=$(ssh $NODE "pgrep -f 'train.py' 2>/dev/null | wc -l" || echo "0")
    if [ "$REMAINING" -gt 0 ]; then
        echo "  Warning: $REMAINING process(es) still running on $NODE"
        # Force kill if needed
        ssh $NODE "pkill -9 -f 'train.py' 2>/dev/null" || true
    else
        echo "  All processes killed on $NODE"
    fi
done

echo ""
echo "Done! All training processes should be stopped."


