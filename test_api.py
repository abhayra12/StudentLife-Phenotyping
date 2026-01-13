
import urllib.request
import json
import time
import sys

def test_api():
    print("Testing API Health...")
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/health") as response:
            data = json.loads(response.read().decode())
            print(f"Health Check: {data}")
            if data['status'] == 'ok' and len(data['models_loaded']) > 0:
                print("SUCCESS: Models loaded.")
            else:
                print("FAILURE: Models not loaded.")
                sys.exit(1)
    except Exception as e:
        print(f"Health Check Failed: {e}")
        sys.exit(1)
        
    print("\nTesting /predict endpoint...")
    # Dummy features: 11 floats * 24 steps = 264 values? 
    # OR if model expects [seq, feat] flattened?
    # verify_phase4.py generated [3362, 30] so 11 is likely feature count per step?
    # But wait, create_sequences in 09_transformer uses:
    # data[i : i+seq_len] -> [24, features]
    # So input to model forward is [24, 1, 11]
    # My reshape in main.py does: np.array(data.features).reshape(24, -1)
    # So if features=11, total length must be 24 * 11 = 264.
    
    # Let's send 24 * 11 floats
    payload = {
        "features": [0.5] * (24 * 11),
        "participant_id": "test_user"
    }
    req = urllib.request.Request(
        "http://127.0.0.1:8000/predict",
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            print(f"Prediction: {data}")
    except Exception as e:
        print(f"Prediction Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Retry loop for server startup
    for i in range(10):
        try:
            test_api()
            break
        except SystemExit:
            if i == 9: raise
            print("Server not ready, retrying...")
            time.sleep(2)
