import argparse
import zmq
import json
import time

def main(port=5555):
    print(f"Connecting to RLBench ZMQ Server at localhost:{port}...")
    
    # Setup ZMQ Context
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://localhost:{port}")
    
    # Test 1: PING
    print("\n--- TEST: PING ---")
    socket.send_string(json.dumps({"type": "PING"}))
    reply = socket.recv_string()
    print(f"Reply: {reply}")
    
    # Test 2: Extraction
    print("\n--- TEST: DYNAMIC STATE EXTRACTION ---")
    socket.send_string(json.dumps({"type": "GET_STATE", "goal": "clean the table"}))
    reply = socket.recv_string()
    
    data = json.loads(reply)
    if data.get("status") == "success":
        print("Successfully extracted state over network!\n")
        print("=== STATE TEXT ===")
        print(data.get("state_text", ""))
        print("==================\n")
    else:
        print(f"Server returned error: {data}")
        
    # Example 3: Send Action Command (uncomment to test movement)
    # print("\n--- OPTIONAL: SENDING ACTION ---")
    # action_req = {
    #     "type": "EXECUTE_ACTION", 
    #     "actions": [
    #         {"name": "pick", "args": ["mug_table", "table"]}
    #     ]
    # }
    # socket.send_string(json.dumps(action_req))
    # reply = socket.recv_string()
    # print(f"Action Reply: {reply}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5555, help="ZMQ Port")
    args = parser.parse_args()
    
    main(port=args.port)
