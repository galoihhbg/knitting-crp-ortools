import json
import logging
import sys

# Configure logging to stdout
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(levelname)s - %(name)s - %(message)s')

from app.engine.model import Engine

def main():
    payload_file = "/home/anya/anya/knitting-crp-ortools/logs/solver_input_CP_1783406986016920762.json"
    with open(payload_file, "r") as f:
        payload = json.load(f)
    
    engine = Engine(payload)
    result = engine.solve()
    print(f"Status: {result.get('status')}")

if __name__ == "__main__":
    main()
